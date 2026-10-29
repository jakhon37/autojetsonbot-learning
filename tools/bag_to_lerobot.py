#!/usr/bin/env python3
"""Convert autoJetsonBot rosbag2 recordings into a LeRobot v3 dataset.

    python tools/bag_to_lerobot.py bags/ --root data/jbl_dock --repo-id jakhon37/jbl_dock
    python tools/bag_to_lerobot.py bags/run_01 bags/run_02 --root ... --repo-id ... --push

Inputs are bag directories (metadata.yaml), bare .db3 / .mcap files, or parent directories
searched recursively. Foxy bags (sqlite3, no message definitions inside) are read with the
--typestore definitions; Jazzy mcap bags carry their own.

Each episode is resampled at --fps on bag log (receive) time, because the stock camera node does
not stamp its frames (docs/INTERFACES.md, "Dataset"). For an episode starting at t0 and ending
at t1, tick k is at T = t0 + k / fps for T < t1, and

    image   the latest camera frame received at or before T; ticks before the episode's first
            frame are skipped
    state   the latest /odom twist (v, w) at or before T, zero before the first odometry
    action  the latest /cmd_vel_in (v, w) at or before T, zero if older than --cmd-timeout

Leading ticks whose action is zero (the robot standing at the start pose while the operator
gets to the teleop terminal) are dropped unless --keep-idle-start: a memoryless policy would
otherwise learn "stand still" for the start views. Their length is recorded as idle_start_s.

Episodes come from /episode/event ("start <method> <pose_id> <round>" ... "end <outcome>
[reason]"; "intervention [note]" events are skipped) when the bag has events, else the whole
bag is one episode. A start without an end is
treated as an abort. Aborted episodes are dropped unless --keep-abort. Per-episode metadata is
written to meta/jbl_episodes.jsonl (see jbl_episodes.py).

The dataset is built in a staging directory beside --root and moved into place only when the
conversion succeeded, so a failed run never leaves a half-written dataset at --root.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "jbl_nodes"))
from jbl_nodes import imgconv, preprocess  # noqa: E402

import jbl_episodes  # noqa: E402

log = logging.getLogger("bag_to_lerobot")

TASK = "drive to the box with the marker and stop in front of it"
ROBOT_TYPE = "autojetsonbot"
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_FPS = 10
CMD_TIMEOUT_S = 0.3
STALE_IMAGE_WARN_S = 0.5

COMPRESSED_TOPIC = "/camera/image_raw/compressed"
RAW_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"
CMD_TOPIC = "/cmd_vel_in"
EVENT_TOPIC = "/episode/event"
COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
RAW_TYPE = "sensor_msgs/msg/Image"
ODOM_TYPE = "nav_msgs/msg/Odometry"
TWIST_TYPE = "geometry_msgs/msg/Twist"
STRING_TYPE = "std_msgs/msg/String"
OUTCOMES = ("success", "fail", "abort")
TYPESTORES = ("foxy", "humble", "jazzy")
BAG_SUFFIXES = (".db3", ".mcap")


def dataset_features(use_video: bool) -> dict[str, dict[str, Any]]:
    """LeRobot feature spec of docs/INTERFACES.md (image stored HWC, served as 3x240x320)."""
    h, w = preprocess.DATASET_HW
    return {
        IMAGE_KEY: {
            "dtype": "video" if use_video else "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (2,), "names": ["v", "w"]},
        ACTION_KEY: {"dtype": "float32", "shape": (2,), "names": ["v", "w"]},
    }


# --------------------------------------------------------------------------------------------
# /episode/event


class EpisodeEvent(NamedTuple):
    kind: str  # "start", "end" or "intervention"
    method: str | None = None
    pose_id: int | None = None
    round: int | None = None
    outcome: str | None = None
    reason: str = ""


def parse_event(text: str) -> EpisodeEvent:
    """/episode/event payload -> EpisodeEvent; ValueError if malformed."""
    words = text.split()
    if not words:
        raise ValueError("empty event")
    kind, rest = words[0], words[1:]
    if kind == "start":
        if len(rest) != 3:
            raise ValueError("start takes <method> <pose_id> <round>")
        try:
            pose_id, round_ = int(rest[1]), int(rest[2])
        except ValueError:
            raise ValueError("pose_id and round must be integers") from None
        if pose_id < 0 or round_ < 0:
            raise ValueError("pose_id and round must be >= 0")
        return EpisodeEvent("start", method=rest[0], pose_id=pose_id, round=round_)
    if kind == "end":
        if not rest or rest[0] not in OUTCOMES:
            raise ValueError("end takes <%s> [reason]" % "|".join(OUTCOMES))
        return EpisodeEvent("end", outcome=rest[0], reason=" ".join(rest[1:]))
    if kind == "intervention":  # an operator note; dock_judge ends the episode with "end fail ..."
        return EpisodeEvent("intervention", reason=" ".join(rest))
    raise ValueError("event kind must be start, end or intervention, got %r" % kind)


# --------------------------------------------------------------------------------------------
# Resampling (pure: no bag or dataset I/O)


class Msg(NamedTuple):
    kind: str  # "image", "odom", "cmd" or "event"
    t_ns: int  # bag log time
    value: Any  # image: opaque payload; odom/cmd: (v, w); event: str


@dataclass
class Episode:
    start_ns: int
    end_ns: int | None = None  # exclusive
    method: str | None = None
    pose_id: int | None = None
    round: int | None = None
    outcome: str | None = None
    reason: str = ""
    num_frames: int = 0
    max_image_age_ns: int = 0
    idle_start_ticks: int = 0  # leading zero-action ticks dropped (see keep_idle_start)


class Tick(NamedTuple):
    t_ns: int
    image: Any  # payload of the camera message used
    image_t_ns: int
    state: tuple[float, float]
    action: tuple[float, float]


class EpisodeStarted(NamedTuple):
    episode: Episode


class EpisodeEnded(NamedTuple):
    episode: Episode


ResampleEvent = EpisodeStarted | Tick | EpisodeEnded


def tick_time(start_ns: int, k: int, fps: int) -> int:
    """T_k in integer nanoseconds (no float drift over long episodes)."""
    return start_ns + (k * 1_000_000_000) // fps


def resample(
    messages: Iterable[Msg],
    fps: int = DEFAULT_FPS,
    cmd_timeout_s: float = CMD_TIMEOUT_S,
    whole_bag: tuple[int, int] | None = None,
    warn: Callable[[str], None] = log.warning,
    keep_idle_start: bool = False,
) -> Iterator[ResampleEvent]:
    """Stream of messages in log-time order -> episode starts, ticks and ends.

    ``whole_bag`` = (start_ns, end_ns exclusive) makes the whole span one episode and ignores
    events; otherwise episodes come from "event" messages. A tick at T is emitted once a message
    later than T arrives (or the stream ends), so every message at or before T is accounted for.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    cmd_timeout_ns = int(round(cmd_timeout_s * 1e9))
    image: tuple[int, Any] | None = None
    odom: tuple[int, float, float] | None = None
    cmd: tuple[int, float, float] | None = None
    episode: Episode | None = None
    k = 0
    last_t: int | None = None

    def ticks_before(t_ns: int) -> Iterator[Tick]:
        nonlocal k
        assert episode is not None
        while True:
            T = tick_time(episode.start_ns, k, fps)
            if T >= t_ns:
                return
            k += 1
            if image is None or image[0] < episode.start_ns:
                continue
            state = (odom[1], odom[2]) if odom is not None else (0.0, 0.0)
            fresh = cmd is not None and T - cmd[0] <= cmd_timeout_ns
            action = (cmd[1], cmd[2]) if fresh else (0.0, 0.0)
            if not keep_idle_start and episode.num_frames == 0 and action == (0.0, 0.0):
                episode.idle_start_ticks += 1
                continue
            episode.num_frames += 1
            episode.max_image_age_ns = max(episode.max_image_age_ns, T - image[0])
            yield Tick(T, image[1], image[0], state, action)

    def close(end_ns: int, outcome: str | None, reason: str) -> EpisodeEnded:
        nonlocal episode
        assert episode is not None
        episode.end_ns, episode.outcome, episode.reason = end_ns, outcome, reason
        ended, episode = episode, None
        return EpisodeEnded(ended)

    if whole_bag is not None:
        episode = Episode(start_ns=whole_bag[0])
        yield EpisodeStarted(episode)

    for msg in messages:
        if last_t is not None and msg.t_ns < last_t:
            raise ValueError("messages are not in log-time order (%d after %d)" % (msg.t_ns, last_t))
        last_t = msg.t_ns
        if episode is not None:
            limit = msg.t_ns if whole_bag is None else min(msg.t_ns, whole_bag[1])
            yield from ticks_before(limit)
        if msg.kind == "image":
            image = (msg.t_ns, msg.value)
        elif msg.kind == "odom":
            odom = (msg.t_ns, float(msg.value[0]), float(msg.value[1]))
        elif msg.kind == "cmd":
            cmd = (msg.t_ns, float(msg.value[0]), float(msg.value[1]))
        elif msg.kind == "event":
            if whole_bag is not None:
                continue
            try:
                event = parse_event(msg.value)
            except ValueError as exc:
                warn("ignoring malformed %s %r at %d: %s" % (EVENT_TOPIC, msg.value, msg.t_ns, exc))
                continue
            if event.kind == "intervention":
                continue  # not a boundary: the end event that follows decides the outcome
            if event.kind == "start":
                if episode is not None:
                    yield close(msg.t_ns, "abort", "no end event before the next start")
                episode = Episode(msg.t_ns, method=event.method, pose_id=event.pose_id, round=event.round)
                k = 0
                yield EpisodeStarted(episode)
            elif episode is None:
                warn("ignoring '%s' at %d: no episode was started" % (msg.value.strip(), msg.t_ns))
            else:
                yield close(msg.t_ns, event.outcome, event.reason)
        else:
            raise ValueError("unknown message kind %r" % msg.kind)

    if whole_bag is not None:
        yield from ticks_before(whole_bag[1])
        yield close(whole_bag[1], None, "")
    elif episode is not None:
        end_ns = (last_t if last_t is not None else episode.start_ns) + 1
        yield from ticks_before(end_ns)
        yield close(end_ns, "abort", "recording ended before the end event")


# --------------------------------------------------------------------------------------------
# Bags


def find_bags(inputs: Sequence[str | Path]) -> list[tuple[Path, str]]:
    """(bag path, display name) for bag dirs / files given directly or found under parents."""
    found: list[tuple[Path, str]] = []
    for item in inputs:
        path = Path(item)
        if _is_bag(path):
            found.append((path, path.name))
            continue
        if not path.is_dir():
            raise FileNotFoundError("not a bag directory, .db3/.mcap file or directory: %s" % path)
        below = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames.sort()
            here = Path(dirpath)
            if "metadata.yaml" in filenames:
                below.append(here)
                dirnames[:] = []
                continue
            below.extend(here / f for f in filenames if Path(f).suffix in BAG_SUFFIXES)
        if not below:
            raise FileNotFoundError("no rosbag2 recordings under %s" % path)
        found.extend((b, b.relative_to(path).as_posix()) for b in sorted(below))
    unique: dict[Path, tuple[Path, str]] = {}
    for bag, name in found:
        unique.setdefault(bag.resolve(), (bag, name))
    return list(unique.values())


def _is_bag(path: Path) -> bool:
    if path.is_dir():
        return (path / "metadata.yaml").is_file()
    return path.is_file() and path.suffix in BAG_SUFFIXES


class BagSource:
    """One rosbag2 recording opened with rosbags; yields Msg in log-time order."""

    def __init__(self, path: Path, typestore: str = "foxy", image_topic: str | None = None) -> None:
        if typestore not in TYPESTORES:
            raise ValueError("typestore must be one of %s" % ", ".join(TYPESTORES))
        self.path = Path(path)
        self._typestore_name = typestore
        self._requested_image_topic = image_topic
        self._reader: Any = None
        self.topics: dict[str, str] = {}  # kind -> topic
        self.image_type = ""
        self.has_events = False
        self.start_ns = 0
        self.end_ns = 0

    def __enter__(self) -> BagSource:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore

        store = {"foxy": Stores.ROS2_FOXY, "humble": Stores.ROS2_HUMBLE, "jazzy": Stores.ROS2_JAZZY}
        self._reader = AnyReader([self.path], default_typestore=get_typestore(store[self._typestore_name]))
        self._reader.open()
        try:
            self._select_topics()
        except BaseException:
            self._reader.close()
            raise
        self.start_ns, self.end_ns = self._reader.start_time, self._reader.end_time
        return self

    def __exit__(self, *exc: Any) -> None:
        self._reader.close()

    def _select_topics(self) -> None:
        info = {name: t for name, t in self._reader.topics.items() if t.msgcount > 0}

        def require_type(topic: str, allowed: Sequence[str]) -> str:
            msgtype = info[topic].msgtype
            if msgtype not in allowed:
                raise ValueError("%s: %s has type %s, expected %s" % (self.path, topic, msgtype, " or ".join(allowed)))
            return msgtype

        candidates = [self._requested_image_topic] if self._requested_image_topic else [COMPRESSED_TOPIC, RAW_TOPIC]
        image_topic = next((t for t in candidates if t in info), None)
        if image_topic is None:
            raise ValueError("%s: no camera messages on %s" % (self.path, " or ".join(candidates)))
        self.image_type = require_type(image_topic, (COMPRESSED_TYPE, RAW_TYPE))
        self.topics = {"image": image_topic}
        for kind, topic, msgtype in (("odom", ODOM_TOPIC, ODOM_TYPE), ("cmd", CMD_TOPIC, TWIST_TYPE),
                                     ("event", EVENT_TOPIC, STRING_TYPE)):
            if topic in info:
                require_type(topic, (msgtype,))
                self.topics[kind] = topic
        self.has_events = "event" in self.topics

    def messages(self) -> Iterator[Msg]:
        kind_of = {topic: kind for kind, topic in self.topics.items()}
        connections = [c for c in self._reader.connections if c.topic in kind_of]
        deserialize = self._reader.deserialize
        for conn, t_ns, raw in self._reader.messages(connections=connections):
            kind = kind_of[conn.topic]
            if kind == "image":
                # Deserialised and decoded only if a tick uses it (see decode_frame).
                yield Msg(kind, t_ns, (raw, conn.msgtype))
            elif kind == "odom":
                twist = deserialize(raw, conn.msgtype).twist.twist
                yield Msg(kind, t_ns, (twist.linear.x, twist.angular.z))
            elif kind == "cmd":
                twist = deserialize(raw, conn.msgtype)
                yield Msg(kind, t_ns, (twist.linear.x, twist.angular.z))
            else:
                yield Msg(kind, t_ns, deserialize(raw, conn.msgtype).data)

    def decode_frame(self, payload: tuple[bytes, str]) -> np.ndarray:
        """Camera message payload -> 240x320x3 uint8 RGB dataset frame."""
        raw, msgtype = payload
        msg = self._reader.deserialize(raw, msgtype)
        bgr = imgconv.compressed_to_bgr(msg) if msgtype == COMPRESSED_TYPE else imgconv.image_to_bgr(msg)
        return preprocess.to_dataset_frame(bgr, is_bgr=True)


# --------------------------------------------------------------------------------------------
# Dataset writing


@dataclass
class ConversionReport:
    root: Path
    storage: str  # "video (<codec>)" or "images (...)"
    episodes: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return sum(e["num_frames"] for e in self.episodes)


def choose_storage(video: str = "auto", vcodec: str | None = None) -> tuple[bool, Any, str]:
    """(use_video, RGBEncoderConfig or None, description) for --video auto|on|off."""
    if video not in ("auto", "on", "off"):
        raise ValueError("video must be auto, on or off")
    if video == "off":
        return False, None, "images (--video off)"
    try:
        from lerobot.configs import RGBEncoderConfig

        encoder = RGBEncoderConfig(vcodec=vcodec) if vcodec else RGBEncoderConfig()
    except (ImportError, ValueError) as exc:
        if video == "on":
            raise RuntimeError("video encoding requested but unavailable: %s" % exc) from exc
        return False, None, "images (video encoding unavailable: %s)" % exc
    return True, encoder, "video (%s)" % encoder.vcodec


def episode_record(episode: Episode, bag_name: str, fps: int) -> dict[str, Any]:
    assert episode.end_ns is not None
    return {
        "bag": bag_name,
        "method": episode.method,
        "pose_id": episode.pose_id,
        "round": episode.round,
        "outcome": episode.outcome,
        "reason": episode.reason,
        "duration_s": round((episode.end_ns - episode.start_ns) / 1e9, 3),
        "num_frames": episode.num_frames,
        "start_time_ns": episode.start_ns,
        "max_image_age_s": round(episode.max_image_age_ns / 1e9, 3),
        "idle_start_s": round(episode.idle_start_ticks / fps, 3),
    }


def _check_output_root(root: Path, overwrite: bool) -> None:
    if not root.exists():
        return
    if not overwrite:
        raise FileExistsError("%s exists; pass --overwrite to replace it" % root)
    is_dataset = (root / "meta" / "info.json").is_file()
    if not root.is_dir() or not (is_dataset or not any(root.iterdir())):
        raise FileExistsError("refusing to overwrite %s: it is not a LeRobot dataset directory" % root)


def convert(
    bags: Sequence[tuple[Path, str]],
    root: Path,
    repo_id: str,
    *,
    fps: int = DEFAULT_FPS,
    cmd_timeout_s: float = CMD_TIMEOUT_S,
    keep_abort: bool = False,
    video: str = "auto",
    vcodec: str | None = None,
    typestore: str = "foxy",
    image_topic: str | None = None,
    task: str = TASK,
    overwrite: bool = False,
    image_writer_threads: int = 4,
    keep_idle_start: bool = False,
) -> ConversionReport:
    """Write every kept episode of ``bags`` (from find_bags) to a new dataset at ``root``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    _check_output_root(root, overwrite)
    use_video, encoder, storage = choose_storage(video, vcodec)
    if not use_video:
        log.warning("storing frames as %s", storage)
    report = ConversionReport(root=root, storage=storage)
    staging = root.parent / (".%s.partial-%d" % (root.name, os.getpid()))
    if staging.exists():
        shutil.rmtree(staging)
    root.parent.mkdir(parents=True, exist_ok=True)

    def warn(text: str) -> None:
        report.warnings.append(text)
        log.warning(text)

    dataset = LeRobotDataset.create(
        repo_id,
        fps=fps,
        features=dataset_features(use_video),
        root=staging,
        robot_type=ROBOT_TYPE,
        use_videos=use_video,
        image_writer_threads=image_writer_threads,
        rgb_encoder=encoder,
    )
    try:
        for bag_path, bag_name in bags:
            _convert_bag(dataset, bag_path, bag_name, report, warn, fps=fps, cmd_timeout_s=cmd_timeout_s,
                         keep_abort=keep_abort, typestore=typestore, image_topic=image_topic, task=task,
                         keep_idle_start=keep_idle_start)
        dataset.finalize()
        if not report.episodes:
            raise RuntimeError("no episode was kept; nothing to write (see warnings above)")
        jbl_episodes.write_episode_meta(staging, report.episodes)
        if root.exists():
            shutil.rmtree(root)
        staging.rename(root)
    except BaseException:
        with contextlib.suppress(Exception):
            dataset.finalize()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _convert_bag(
    dataset: Any,
    bag_path: Path,
    bag_name: str,
    report: ConversionReport,
    warn: Callable[[str], None],
    *,
    fps: int,
    cmd_timeout_s: float,
    keep_abort: bool,
    typestore: str,
    image_topic: str | None,
    task: str,
    keep_idle_start: bool = False,
) -> None:
    t_start = time.monotonic()
    kept = dropped = 0
    with BagSource(bag_path, typestore, image_topic) as bag:
        for kind in ("odom", "cmd"):
            if kind not in bag.topics:
                warn("%s: no %s messages; %s is zero throughout" % (
                    bag_name, ODOM_TOPIC if kind == "odom" else CMD_TOPIC, "state" if kind == "odom" else "action"))
        whole_bag = None if bag.has_events else (bag.start_ns, bag.end_ns)
        last_payload: Any = None
        last_frame: np.ndarray | None = None
        for item in resample(bag.messages(), fps, cmd_timeout_s, whole_bag,
                             warn=lambda text: warn("%s: %s" % (bag_name, text)), keep_idle_start=keep_idle_start):
            if isinstance(item, Tick):
                if item.image is not last_payload:
                    try:
                        last_frame = bag.decode_frame(item.image)
                    except ValueError as exc:
                        raise ValueError("%s: camera frame received at %d: %s"
                                         % (bag_name, item.image_t_ns, exc)) from exc
                    last_payload = item.image
                dataset.add_frame({
                    IMAGE_KEY: last_frame,
                    STATE_KEY: np.asarray(item.state, dtype=np.float32),
                    ACTION_KEY: np.asarray(item.action, dtype=np.float32),
                    "task": task,
                })
            elif isinstance(item, EpisodeEnded):
                episode = item.episode
                record = episode_record(episode, bag_name, fps)
                reason = None
                if episode.num_frames == 0:
                    reason = ("no non-zero command in the episode" if episode.idle_start_ticks
                              else "no camera frame inside the episode")
                elif episode.outcome == "abort" and not keep_abort:
                    reason = "outcome abort (%s)" % (episode.reason or "no reason given")
                if reason is None:
                    dataset.save_episode()
                    record = dict(episode_index=dataset.meta.total_episodes - 1, **record)
                    report.episodes.append(record)
                    kept += 1
                    if episode.max_image_age_ns > STALE_IMAGE_WARN_S * 1e9:
                        warn("%s: episode %d used a camera frame %.2f s old (camera stall?)"
                             % (bag_name, record["episode_index"], record["max_image_age_s"]))
                else:
                    if dataset.has_pending_frames():
                        dataset.clear_episode_buffer()
                    report.dropped.append(dict(record, dropped=reason))
                    dropped += 1
    log.info("%s: %d episode(s) kept, %d dropped, %.1f s", bag_name, kept, dropped, time.monotonic() - t_start)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("inputs", nargs="+", help="bag directories, .db3/.mcap files or parent directories")
    parser.add_argument("--root", required=True, type=Path, help="output dataset directory (must not exist)")
    parser.add_argument("--repo-id", required=True, help="dataset id, e.g. jakhon37/jbl_dock")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--cmd-timeout", type=float, default=CMD_TIMEOUT_S, metavar="S",
                        help="action is zero when the latest command is older (default %(default)s s)")
    parser.add_argument("--keep-abort", action="store_true", help="keep aborted and unterminated episodes")
    parser.add_argument("--keep-idle-start", action="store_true",
                        help="keep the zero-action ticks before an episode's first command")
    parser.add_argument("--video", choices=("auto", "on", "off"), default="auto",
                        help="store frames as video (auto: when PyAV can encode), else PNG images")
    parser.add_argument("--vcodec", default=None, help="video codec (default: LeRobot's, libsvtav1)")
    parser.add_argument("--typestore", choices=TYPESTORES, default="foxy",
                        help="message definitions for bags without embedded ones (Foxy/Humble sqlite3)")
    parser.add_argument("--image-topic", default=None,
                        help="camera topic (default: %s, else %s)" % (COMPRESSED_TOPIC, RAW_TOPIC))
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset at --root")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--push", action="store_true", help="push the dataset to the Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="with --push: create a private repo")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1 prints its whole config per episode otherwise
    import datasets

    datasets.disable_progress_bars()

    bags = find_bags(args.inputs)
    log.info("%d bag(s): %s", len(bags), ", ".join(name for _, name in bags))
    report = convert(bags, args.root, args.repo_id, fps=args.fps, cmd_timeout_s=args.cmd_timeout,
                     keep_abort=args.keep_abort, video=args.video, vcodec=args.vcodec,
                     typestore=args.typestore, image_topic=args.image_topic, task=args.task,
                     overwrite=args.overwrite, image_writer_threads=args.image_writer_threads,
                     keep_idle_start=args.keep_idle_start)
    print("wrote %s: %d episodes, %d frames at %d fps, %s; %d episode(s) dropped, %d warning(s)" % (
        report.root, len(report.episodes), report.num_frames, args.fps, report.storage,
        len(report.dropped), len(report.warnings)))
    for record in report.dropped:
        print("  dropped %s %s/%s/%s: %s" % (record["bag"], record["method"], record["pose_id"],
                                             record["round"], record["dropped"]))
    if args.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(args.repo_id, root=args.root).push_to_hub(
            private=args.private, tags=["autojetsonbot", "docking", "ros2"])
        print("pushed to https://huggingface.co/datasets/%s" % args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
