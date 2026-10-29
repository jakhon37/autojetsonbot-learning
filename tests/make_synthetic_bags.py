#!/usr/bin/env python3
"""Small, realistic rosbag2 recordings of the robot's topics, written without ROS installed.

    python tests/make_synthetic_bags.py OUT_DIR      # writes the default set and prints it

Two on-disk layouts, both written with the rosbags Writer:

    foxy    sqlite3, serialised with the Foxy typestore, then reduced to what rosbag2 on Foxy
            writes: metadata.yaml version 4, ``<name>_0.db3``, topics table without type hashes
            and no message definitions, so a reader needs a default (Foxy) typestore.
    jazzy   mcap with embedded ros2msg definitions, serialised with the Jazzy typestore.

Every value in a bag is a known function of its log time (``command_at``, ``odom_at``) and every
camera frame encodes its own index (``index_image`` / ``decode_index``), so tests can check what
the converter picked at each tick against an independent oracle (``RecordedBag``). Header stamps
lag log time on purpose: the converter must resample on log time (docs/INTERFACES.md).
"""

import argparse
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

COMPRESSED_TOPIC = "/camera/image_raw/compressed"
RAW_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"
CMD_TOPIC = "/cmd_vel_in"
EVENT_TOPIC = "/episode/event"
LATENCY_TOPIC = "/policy/latency"

COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
RAW_TYPE = "sensor_msgs/msg/Image"
ODOM_TYPE = "nav_msgs/msg/Odometry"
TWIST_TYPE = "geometry_msgs/msg/Twist"
STRING_TYPE = "std_msgs/msg/String"

CAMERA_HW = (480, 640)
JPEG_QUALITY = 85
BAG_EPOCH_NS = 1_761_600_000 * 10**9
CAMERA_STAMP_LAG_NS = 40_000_000  # capture -> receive latency of the camera pipeline
ODOM_STAMP_LAG_NS = 5_000_000

INDEX_BITS = 12
_BIT_GRID = (2, 6)  # rows x cols of bit cells in the top two thirds of the frame
_ON, _OFF = 230, 25
RED_RGB = (220, 30, 30)
BLUE_RGB = (30, 30, 220)

# rosbag2 on Foxy stores the offered QoS as a YAML string inside metadata.yaml.
_FOXY_QOS = (
    "- history: 3\n  depth: 0\n  reliability: 1\n  durability: 2\n  deadline:\n    sec: 2147483647\n"
    "    nsec: 4294967295\n  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n  liveliness: 1\n"
    "  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  avoid_ros_namespace_conventions: false"
)


def command_at(t_s: float) -> Tuple[float, float]:
    """Operator command (v m/s, w rad/s) sent at bag time t_s (seconds since bag start)."""
    return 0.2 * math.sin(2.0 * math.pi * t_s / 5.0), 1.2 * math.sin(2.0 * math.pi * t_s / 3.0 + 0.5)


def odom_at(t_s: float) -> Tuple[float, float]:
    """Measured twist published at t_s: the command 0.2 s earlier, slightly attenuated."""
    v, w = command_at(t_s - 0.2)
    return 0.9 * v, 0.9 * w


def index_image(index: int, encoding: str = "bgr8") -> np.ndarray:
    """640x480x3 uint8 frame whose top two thirds encode ``index`` in 12 big binary cells.

    The bottom third is a red patch (left) and a blue patch (right) so channel order can be
    checked after decoding. Cells are large enough to survive JPEG, the 320x240 resize and
    lossy video encoding.
    """
    if not 0 <= index < 2**INDEX_BITS:
        raise ValueError("index %d does not fit in %d bits" % (index, INDEX_BITS))
    h, w = CAMERA_HW
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    top = (2 * h) // 3
    rows, cols = _BIT_GRID
    ys = np.linspace(0, top, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    for bit in range(INDEX_BITS):
        r, c = divmod(bit, cols)
        rgb[ys[r] : ys[r + 1], xs[c] : xs[c + 1]] = _ON if (index >> bit) & 1 else _OFF
    rgb[top:, : w // 2] = RED_RGB
    rgb[top:, w // 2 :] = BLUE_RGB
    if encoding == "rgb8":
        return rgb
    if encoding == "bgr8":
        return np.ascontiguousarray(rgb[:, :, ::-1])
    raise ValueError("unsupported encoding %r" % encoding)


def _cell_mean(img: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    """Mean colour of the central half of a cell given as fractions of the image size."""
    h, w = img.shape[:2]
    cy0, cy1 = y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0)
    cx0, cx1 = x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0)
    patch = img[int(cy0 * h) : int(math.ceil(cy1 * h)), int(cx0 * w) : int(math.ceil(cx1 * w))]
    return patch.reshape(-1, img.shape[2]).mean(axis=0)


def decode_index(rgb: np.ndarray) -> int:
    """Frame index encoded by ``index_image`` (any resolution, RGB or BGR)."""
    rows, cols = _BIT_GRID
    top = 2.0 / 3.0
    index = 0
    for bit in range(INDEX_BITS):
        r, c = divmod(bit, cols)
        mean = _cell_mean(rgb, r * top / rows, (r + 1) * top / rows, c / cols, (c + 1) / cols).mean()
        if mean > 0.5 * (_ON + _OFF):
            index |= 1 << bit
    return index


def patch_colours(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean colour (in the image's channel order) of the left and right bottom patches."""
    return _cell_mean(rgb, 2.0 / 3.0, 1.0, 0.0, 0.5), _cell_mean(rgb, 2.0 / 3.0, 1.0, 0.5, 1.0)


@dataclass
class BagSpec:
    name: str
    layout: str  # "foxy" (sqlite3, Foxy schema) or "jazzy" (mcap)
    duration_s: float
    image: str = "compressed"  # "compressed", "bgr8" or "rgb8" (raw sensor_msgs/Image)
    camera_hz: float = 15.0
    odom_hz: float = 20.0
    cmd_hz: float = 20.0
    camera_start_s: float = 0.0
    cmd_start_s: float = 0.0
    camera_stalls: Sequence[Tuple[float, float]] = ()  # [t0, t1) without frames
    cmd_gaps: Sequence[Tuple[float, float]] = ()  # [t0, t1) without commands
    latency_hz: float = 0.0  # policy_runner's /policy/latency (one per fresh inference); 0 = no topic
    latency_gaps: Sequence[Tuple[float, float]] = ()  # [t0, t1) without inferences (stale frames)
    zero_cmds: Sequence[float] = ()  # extra commands of exactly (0, 0) at these times: policy_runner's fallback
    events: Sequence[Tuple[float, str]] = ()
    row_padding: int = 0  # extra bytes per row of raw images (Image.step > width * 3)
    seed: int = 0
    offset_s: float = 0.0  # bag start relative to BAG_EPOCH_NS


@dataclass
class RecordedBag:
    """What was written, in bag log time (ns): the oracle tests compare against."""

    spec: BagSpec
    path: Path
    images: List[Tuple[int, int]] = field(default_factory=list)  # (t_ns, encoded frame index)
    odom: List[Tuple[int, float, float]] = field(default_factory=list)
    cmd: List[Tuple[int, float, float]] = field(default_factory=list)
    latency: List[int] = field(default_factory=list)  # /policy/latency times
    events: List[Tuple[int, str]] = field(default_factory=list)
    origin_ns: int = 0  # time zero of the spec (command_at / odom_at / events are relative to it)
    start_ns: int = 0  # first message
    end_ns: int = 0  # last message time + 1 (exclusive), like rosbags' end_time

    def rel_s(self, t_ns: int) -> float:
        """Seconds since the spec's time zero."""
        return (t_ns - self.origin_ns) / 1e9


_LATENCY_JSON = json.dumps(
    {"preprocess_ms": 3.0, "infer_ms": 25.0, "total_ms": 29.0, "image_age_ms": 40.0, "backend": "onnxruntime"}
)


def _in_any(t: float, windows: Sequence[Tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in windows)


def _schedule(spec: BagSpec, origin_ns: int) -> RecordedBag:
    rng = np.random.default_rng(spec.seed)
    bag = RecordedBag(spec=spec, path=Path(), origin_ns=origin_ns)

    def ns(t_s: float) -> int:
        return origin_ns + int(round(t_s * 1e9))

    def rel(t_ns: int) -> float:
        return (t_ns - origin_ns) / 1e9

    t, index = spec.camera_start_s, 0
    while t < spec.duration_s:
        jittered = t + float(rng.uniform(-0.004, 0.004))
        if jittered >= 0.0 and not _in_any(jittered, spec.camera_stalls):
            bag.images.append((ns(jittered), index))
            index += 1
        t += 1.0 / spec.camera_hz
    for k in range(int(math.ceil((spec.duration_s - 0.01) * spec.odom_hz))):
        t_ns = ns(0.01 + k / spec.odom_hz)
        bag.odom.append((t_ns,) + odom_at(rel(t_ns)))
    t = spec.cmd_start_s
    while t < spec.duration_s:
        jittered = t + float(rng.uniform(-0.003, 0.003))
        if jittered >= 0.0 and not _in_any(jittered, spec.cmd_gaps):
            t_ns = ns(jittered)
            bag.cmd.append((t_ns,) + command_at(rel(t_ns)))
        t += 1.0 / spec.cmd_hz
    if spec.zero_cmds:  # no random draws, so the other streams stay as they were
        bag.cmd = sorted(bag.cmd + [(ns(t_s), 0.0, 0.0) for t_s in spec.zero_cmds])
    if spec.latency_hz > 0.0:  # its own generator, so the other streams stay as they were
        lat_rng, t = np.random.default_rng(spec.seed + 1000), 0.0
        while t < spec.duration_s:
            jittered = t + 0.002 + float(lat_rng.uniform(0.0, 0.003))  # just after the tick's command
            if not _in_any(jittered, spec.latency_gaps):
                bag.latency.append(ns(jittered))
            t += 1.0 / spec.latency_hz
    bag.events = [(ns(t_s), text) for t_s, text in spec.events]
    times = [x[0] for x in bag.images + bag.odom + bag.cmd + bag.events] + bag.latency
    bag.start_ns, bag.end_ns = min(times), max(times) + 1
    return bag


class _MessageFactory:
    def __init__(self, store: Stores) -> None:
        self.typestore = get_typestore(store)
        self._t = self.typestore.types

    def _header(self, stamp_ns: int, frame_id: str):
        stamp = self._t["builtin_interfaces/msg/Time"](sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9)
        return self._t["std_msgs/msg/Header"](stamp=stamp, frame_id=frame_id)

    def compressed(self, index: int, stamp_ns: int):
        ok, buf = cv2.imencode(".jpg", index_image(index, "bgr8"), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        assert ok
        return self._t[COMPRESSED_TYPE](
            header=self._header(stamp_ns, "camera_link"), format="jpeg", data=buf.reshape(-1)
        )

    def raw(self, index: int, stamp_ns: int, encoding: str, row_padding: int):
        img = index_image(index, encoding)
        h, w = img.shape[:2]
        rows = np.zeros((h, w * 3 + row_padding), dtype=np.uint8)
        rows[:, : w * 3] = img.reshape(h, w * 3)
        return self._t[RAW_TYPE](
            header=self._header(stamp_ns, "camera_link"),
            height=h,
            width=w,
            encoding=encoding,
            is_bigendian=0,
            step=w * 3 + row_padding,
            data=rows.reshape(-1),
        )

    def odom(self, v: float, w: float, stamp_ns: int):
        t = self._t
        vec = t["geometry_msgs/msg/Vector3"]
        pose = t["geometry_msgs/msg/Pose"](
            position=t["geometry_msgs/msg/Point"](x=0.0, y=0.0, z=0.0),
            orientation=t["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        )
        twist = t[TWIST_TYPE](linear=vec(x=v, y=0.0, z=0.0), angular=vec(x=0.0, y=0.0, z=w))
        return t[ODOM_TYPE](
            header=self._header(stamp_ns, "odom"),
            child_frame_id="base_link",
            pose=t["geometry_msgs/msg/PoseWithCovariance"](pose=pose, covariance=np.zeros(36)),
            twist=t["geometry_msgs/msg/TwistWithCovariance"](twist=twist, covariance=np.zeros(36)),
        )

    def twist(self, v: float, w: float):
        vec = self._t["geometry_msgs/msg/Vector3"]
        return self._t[TWIST_TYPE](linear=vec(x=v, y=0.0, z=0.0), angular=vec(x=0.0, y=0.0, z=w))

    def string(self, text: str):
        return self._t[STRING_TYPE](data=text)


def _reduce_to_foxy_layout(bag_dir: Path, recorded: Dict[str, Tuple[str, int]]) -> None:
    """Rewrite a rosbags sqlite3 bag into the schema and metadata rosbag2 writes on Foxy."""
    src = bag_dir / ("%s.db3" % bag_dir.name)
    db = bag_dir / ("%s_0.db3" % bag_dir.name)
    src.rename(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            """
            DROP TABLE schema;
            DROP TABLE metadata;
            DROP TABLE message_definitions;
            CREATE TABLE topics_foxy(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                                     serialization_format TEXT NOT NULL,
                                     offered_qos_profiles TEXT NOT NULL);
            """
        )
        conn.execute(
            "INSERT INTO topics_foxy SELECT id, name, type, serialization_format, ? FROM topics", (_FOXY_QOS,)
        )
        conn.executescript("DROP TABLE topics; ALTER TABLE topics_foxy RENAME TO topics;")
        first, last, count = conn.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages").fetchone()
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    topics = "".join(
        "    - topic_metadata:\n"
        "        name: %s\n"
        "        type: %s\n"
        "        serialization_format: cdr\n"
        "        offered_qos_profiles: %s\n"
        "      message_count: %d\n" % (topic, msgtype, json.dumps(_FOXY_QOS), n)
        for topic, (msgtype, n) in recorded.items()
    )
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  version: 4\n"
        "  storage_identifier: sqlite3\n"
        "  relative_file_paths:\n"
        "    - %s\n"
        "  duration:\n"
        "    nanoseconds: %d\n"
        "  starting_time:\n"
        "    nanoseconds_since_epoch: %d\n"
        "  message_count: %d\n"
        "  topics_with_message_count:\n"
        "%s"
        '  compression_format: ""\n'
        '  compression_mode: ""\n' % (db.name, last - first, first, count, topics)
    )


def write_bag(spec: BagSpec, out_dir: Path) -> RecordedBag:
    """Write one bag directory ``out_dir/spec.name`` and return what it contains."""
    if spec.layout not in ("foxy", "jazzy"):
        raise ValueError("layout must be foxy or jazzy, got %r" % spec.layout)
    bag = _schedule(spec, BAG_EPOCH_NS + int(round(spec.offset_s * 1e9)))
    bag.path = Path(out_dir) / spec.name
    if bag.path.exists():
        shutil.rmtree(str(bag.path))
    factory = _MessageFactory(Stores.ROS2_FOXY if spec.layout == "foxy" else Stores.ROS2_JAZZY)
    if spec.image == "compressed":
        image_topic, image_type = COMPRESSED_TOPIC, COMPRESSED_TYPE
    else:
        image_topic, image_type = RAW_TOPIC, RAW_TYPE

    def image_msg(index: int, t_ns: int):
        if spec.image == "compressed":
            return factory.compressed(index, t_ns - CAMERA_STAMP_LAG_NS)
        return factory.raw(index, t_ns - CAMERA_STAMP_LAG_NS, spec.image, spec.row_padding)

    # (t_ns, topic order for ties, topic, msgtype, message builder); built lazily while writing
    messages = (
        [(t, 0, EVENT_TOPIC, STRING_TYPE, (lambda s=s: factory.string(s))) for t, s in bag.events]
        + [(t, 1, ODOM_TOPIC, ODOM_TYPE, (lambda t=t, v=v, w=w: factory.odom(v, w, t - ODOM_STAMP_LAG_NS)))
           for t, v, w in bag.odom]
        + [(t, 2, CMD_TOPIC, TWIST_TYPE, (lambda v=v, w=w: factory.twist(v, w))) for t, v, w in bag.cmd]
        + [(t, 3, image_topic, image_type, (lambda i=i, t=t: image_msg(i, t))) for t, i in bag.images]
        + [(t, 4, LATENCY_TOPIC, STRING_TYPE, (lambda: factory.string(_LATENCY_JSON))) for t in bag.latency]
    )
    messages.sort(key=lambda m: (m[0], m[1]))
    storage = StoragePlugin.SQLITE3 if spec.layout == "foxy" else StoragePlugin.MCAP
    counts: Dict[str, Tuple[str, int]] = {}
    with Writer(bag.path, version=8 if spec.layout == "foxy" else 9, storage_plugin=storage) as writer:
        connections = {}
        for t_ns, _, topic, msgtype, build in messages:
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msgtype, typestore=factory.typestore)
            writer.write(connections[topic], t_ns, factory.typestore.serialize_cdr(build(), msgtype))
            counts[topic] = (msgtype, counts.get(topic, (msgtype, 0))[1] + 1)
    if spec.layout == "foxy":
        _reduce_to_foxy_layout(bag.path, counts)
    return bag


def default_specs() -> List[BagSpec]:
    """The bag set the tests use; see each spec for the case it covers."""
    return [
        # Whole bag = one episode; odom before the camera starts (ticks before the first
        # frame are skipped), no command during the first second (action zero).
        BagSpec("foxy_teleop", "foxy", duration_s=4.0, camera_start_s=0.25, cmd_start_s=1.0, seed=1),
        # Three episodes split by /episode/event (success, abort, fail), a camera stall, a
        # command gap longer than the 0.3 s timeout and one malformed event.
        BagSpec(
            "jazzy_trials",
            "jazzy",
            duration_s=9.0,
            camera_stalls=[(2.0, 2.45)],
            cmd_gaps=[(6.0, 7.0)],
            events=[
                (0.50, "start teleop 3 1"),
                (3.55, "end success"),
                (4.00, "start teleop 4 1"),
                (5.00, "end abort operator stopped"),
                (5.40, "start bc 5 2"),
                (8.25, "end fail timeout"),
                (8.50, "hello"),
            ],
            seed=2,
            offset_s=100.0,
        ),
        # A start without an end: incomplete, handled like an abort.
        BagSpec("foxy_unterminated", "foxy", duration_s=2.0, events=[(0.3, "start teleop 1 2")], seed=3,
                offset_s=200.0),
        # Raw rgb8 frames with padded rows instead of the compressed topic.
        BagSpec("jazzy_raw_rgb", "jazzy", duration_s=1.5, image="rgb8", camera_hz=10.0, row_padding=16, seed=4,
                offset_s=300.0),
        # Raw bgr8 in the Foxy layout; a 5 Hz camera, so each frame serves about two ticks.
        BagSpec("foxy_raw_bgr", "foxy", duration_s=1.2, image="bgr8", camera_hz=5.0, seed=5, offset_s=400.0),
    ]


def write_default_bags(out_dir: Path, specs: Optional[Sequence[BagSpec]] = None) -> Dict[str, RecordedBag]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {spec.name: write_bag(spec, out_dir) for spec in (specs or default_specs())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    for name, bag in write_default_bags(args.out_dir).items():
        print(
            "%-18s %-5s %5.2f s  images %3d  odom %3d  cmd %3d  events %d  -> %s"
            % (name, bag.spec.layout, (bag.end_ns - bag.start_ns) / 1e9, len(bag.images), len(bag.odom),
               len(bag.cmd), len(bag.events), bag.path)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
