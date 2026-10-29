#!/usr/bin/env python3
"""Ground-truth episode metrics and the summary table of the harder simulation evaluation.

    python scripts/sim_eval_summary.py episodes --bags sim_work/eval_v2/jazzy/noise/bags \
        --out results/sim_eval_v2/noise/episodes_gt.csv
    python scripts/sim_eval_summary.py table --results results/sim_eval_v2 \
        --conditions clean_ref noise wide hfov62 --vs bc --out results/sim_eval_v2/SUMMARY_table.md
    python scripts/sim_eval_summary.py merge --out results/sim_eval_v3_bcfix/noise \
        results/sim_eval_v2/noise sim_work/v2/eval_v2/jazzy/noise
    python scripts/sim_eval_summary.py stops --results results/sim_eval_v4_cleanstop \
        --conditions clean_ref noise --method bc_v4

episodes: one row per episode in the bags (every start ... end on /episode/event, aborts
included) with what only the simulator knows, from /fake_robot/ground_truth, plus the judge's
and the controller's view from /dock_judge/status and /cmd_vel_in, and the inputs that reached
the controller in the end (camera frames, policy_runner's /policy/latency):
    collided          the footprint touched the box or a wall (fake_robot's latched flag; the
                      first 0.3 s after the start are skipped, the teleport may still be pending)
    final_*           at the last ground-truth sample before the end: base_link distance to the
                      box face plane, lateral offset from the marker centre line (+ = left when
                      facing the box), heading off square (deg, + = turned left) and bumper gap
    final_range_gt_m  at that pose: the lidar front-sector median range dock_judge would measure
                      without noise (sim_world's ray cast, judge_logic.front_range)
    final_bearing_gt_deg  at that pose: the marker centre's true angle from the camera axis
                      (deg, + = left), where dock_judge estimates it from the detected marker
    zone_reached      the judge's in_band and aligned held together for >= 1.0 s at some point,
                      whatever "stopped" said (the success rule without the stop requirement);
                      time_to_zone_s = when that first happened, from the start
    end_in_zone       in_band and aligned on the last judge status before the end
    cmd_v_end         median requested /cmd_vel_in v over the last 2 s before the end (m/s)
    cmd_v_min_last1s  smallest |requested v| in the last 1 s before the end, over every
                      request: 0 when the controller commanded a stop (the servo does once
                      inside its band), and also when policy_runner sent a fallback zero
    own_v_min_last1s  the same over the controller's own requests: for a policy arm without
                      policy_runner's fallback zeros (below), for the servo (servo /
                      servo_demo) every request. Empty when there was none. "Stop commanded"
                      (table, stops) reads this column; in an episodes_gt.csv written before
                      it existed (v2, v3), cmd_v_min_last1s
    camera_frames_last1s  camera frames (/camera/image_raw/compressed) recorded in the last
                      1 s before the end; fake_robot publishes 15 Hz (config/sim.yaml).
                      Empty when the bag has no camera topic
    fresh_inferences_last1s  policy arms (any method but servo / servo_demo): /policy/latency
                      messages recorded in the last 1 s before the end. policy_runner publishes
                      one per inference on a frame at most max_image_age (0.2 s) old and none
                      on a tick that publishes a fallback zero; about 10 at its 10 Hz. Empty
                      for the servo and when the bag has no such topic
    fallback_zeros_last1s  policy arms: requests in the last 1 s before the end that were
                      exactly (0, 0). policy_runner sends exactly (0, 0) instead of its
                      network's output on a tick whose newest frame is missing, older than
                      max_image_age or stamped in the future, and after an inference error.
                      An inference that overruns the 100 ms period is followed at once by
                      another tick (the timer catches up), which is such a tick when the
                      newest frame is by then too old. The network's own output is
                      never exactly (0, 0) (a linear head, no deadband in policy_runner).
                      dock_judge reads the latest request at each tick, so a fallback zero
                      can stand in for the policy's request. Empty for the servo
    stale_last1s      the validity check of results/sim_eval_v4_cleanstop/PREREG.md: 1 when
                      the last 1 s had fewer than 5 camera frames or, for a policy arm, fewer
                      than 5 fresh inferences or any fallback zero, so a stop there may come
                      from stale input (servo_baseline also requests (0, 0) when no frame
                      arrived for 0.5 s); 0 otherwise; empty when a count it needs was not
                      recorded
The default box of config/sim.yaml is assumed (front face centre (1.5, 0), facing -x).

table: per condition directory (trials.csv, episodes_gt.csv), per method: success k/n with the
Wilson 95% CI, the exact McNemar test against servo on pose x round pairs (scoring exactly as
tools/analyze_trials.py), median time to dock, failure reasons, final errors and the
ground-truth columns above, and the validity check: successes flagged stale_last1s and k/n with
them counted as failures. --vs M adds a table pairing M with each method listed after it.

stops: one method's scored trials pooled over the given conditions: how many commanded a stop
with their own requests (the table's "Stop commanded") and the ground-truth final lidar range of
those, median (IQR); then the same over every request (cmd_v_min_last1s, fallback zeros
included) as a sensitivity line. Exit status 2 when a condition has no trials.csv or a scored
trial has no ground-truth row (the lines are still written, saying so).

merge: condition folders of separate sessions -> one folder whose trials.csv, episodes_gt.csv
and schedule.csv hold the rows of each input in turn, so analyze_trials and table pair the
arms on pose_id x round. A method may have rows in one input only. An input without
schedule.csv contributes the slots of its trials.csv instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JBL_NODES = REPO / "ros2_ws" / "src" / "jbl_nodes"  # sim_world and judge_logic, for the ground truth
sys.path.insert(0, str(REPO / "tools"))
import analyze_trials  # noqa: E402

BOX_X, BOX_Y = 1.5, 0.0  # config/sim.yaml; the face looks along -x, so facing it squarely is yaw 0
TELEPORT_SETTLE_S = 0.3
ZONE_HOLD_S = 1.0
CMD_TAIL_S = 2.0
TARGET_RANGE_M = 0.25
STOP_COMMANDED_V = 0.005  # m/s: "commanded a stop" = min |requested v| in the last 1 s below this
CAMERA_TOPIC = "/camera/image_raw/compressed"  # what the judge, the servo and policy_runner read
LATENCY_TOPIC = "/policy/latency"  # policy_runner: one message per inference on a fresh frame
TOPICS = ("/episode/event", "/fake_robot/ground_truth", "/dock_judge/status", "/cmd_vel_in")
COUNTED_TOPICS = (CAMERA_TOPIC, LATENCY_TOPIC)  # only their times are read
SERVO_METHODS = ("servo", "servo_demo")  # servo_baseline's arms; every other method is a policy arm
STALE_WINDOW_S = 1.0  # the validity check: in the last 1 s before the end, fewer than
STALE_MIN_FRAMES = 5  # 5 camera frames or (policy arms) 5 fresh inferences flag the episode,
STALE_MIN_INFERENCES = 5
STALE_MAX_FALLBACK_ZEROS = 0  # and so does (policy arms) any fallback zero
EPISODE_COLUMNS = (
    "method",
    "pose_id",
    "round",
    "outcome",
    "reason",
    "duration_s",
    "collided",
    "final_dist_m",
    "final_lateral_m",
    "final_heading_deg",
    "final_bumper_m",
    "final_range_gt_m",
    "final_bearing_gt_deg",
    "zone_reached",
    "time_to_zone_s",
    "end_in_zone",
    "cmd_v_end",
    "cmd_v_min_last1s",
    "own_v_min_last1s",
    "camera_frames_last1s",
    "fresh_inferences_last1s",
    "fallback_zeros_last1s",
    "stale_last1s",
    "bag",
)


def wrap_deg(angle_rad: float) -> float:
    return math.degrees((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


_SIM: Any = None


def gt_range_and_bearing(x: float, y: float, yaw: float) -> tuple[float | None, float | None]:
    """At a ground-truth pose in fake_robot's default world (config/sim.yaml): the lidar front
    range dock_judge would measure without noise (the median of the valid beams within its 0.17 rad
    sector) and the marker centre's true bearing (deg, + = left; None when either is undefined)."""
    global _SIM
    if str(JBL_NODES) not in sys.path:
        sys.path.insert(0, str(JBL_NODES))
    from jbl_nodes.judge_logic import front_range
    from jbl_nodes.sim_world import Pose2D, SimWorld

    if _SIM is None:
        _SIM = SimWorld()
    pose, lidar = Pose2D(x, y, yaw), _SIM.lidar
    ranges, _ = _SIM.cast(_SIM.lidar_origin(pose), yaw + _SIM.beam_angles)
    r = front_range(ranges, lidar.angle_min, lidar.angle_increment, lidar.range_min, lidar.range_max)
    return r, _SIM.marker_bearing_deg(pose)


def stale_flag(method: str, frames: int | None, inferences: int | None, fallback_zeros: int | None) -> int | str:
    """The validity check: 1 when the last STALE_WINDOW_S had fewer than STALE_MIN_FRAMES camera
    frames or, for a policy arm, fewer than STALE_MIN_INFERENCES fresh inferences or more than
    STALE_MAX_FALLBACK_ZEROS fallback zeros; 0 when none of that; "" when a count it needs is
    unknown (None) and the known ones do not flag it. The servo's arms need only the frames."""
    policy = method not in SERVO_METHODS
    if (frames is not None and frames < STALE_MIN_FRAMES) or (
        policy
        and (
            (inferences is not None and inferences < STALE_MIN_INFERENCES)
            or (fallback_zeros is not None and fallback_zeros > STALE_MAX_FALLBACK_ZEROS)
        )
    ):
        return 1
    if frames is None or (policy and (inferences is None or fallback_zeros is None)):
        return ""
    return 0


@dataclass
class Episode:
    method: str
    pose_id: int
    round: int
    t0: float
    bag: str
    outcome: str = "abort"
    reason: str = "no end event"
    t1: float | None = None
    gt: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    status: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    # /cmd_vel_in requests (t, v, w); (t, v) when w is unknown, which is never a fallback zero
    cmd: list[tuple[float, ...]] = field(default_factory=list)
    frames: list[float] | None = None  # camera frame times; None: the bag has no camera topic
    inferences: list[float] | None = None  # /policy/latency times; None: the bag has no such topic

    @property
    def policy(self) -> bool:
        """policy_runner's arm (every method but the servo's)."""
        return self.method not in SERVO_METHODS

    def is_fallback(self, request: tuple[float, ...]) -> bool:
        """policy_runner's fallback zero: a policy arm's request of exactly (0, 0)."""
        return self.policy and len(request) > 2 and request[1] == 0.0 and request[2] == 0.0

    def row(self) -> dict[str, Any]:
        t1 = self.t1 if self.t1 is not None else (self.gt[-1][0] if self.gt else self.t0)
        gt = [(t, g) for t, g in self.gt if t >= self.t0 + TELEPORT_SETTLE_S]
        last = gt[-1][1] if gt else None
        zone_since = None
        time_to_zone = None
        for t, s in self.status:
            if s.get("in_band") and s.get("aligned"):
                zone_since = t if zone_since is None else zone_since
                if time_to_zone is None and t - zone_since >= ZONE_HOLD_S:
                    time_to_zone = t - self.t0
            else:
                zone_since = None
        gt_range, gt_bearing = gt_range_and_bearing(last["x"], last["y"], last["yaw"]) if last else (None, None)
        tail = [c[1] for c in self.cmd if c[0] >= t1 - CMD_TAIL_S]
        recent = [c for c in self.cmd if c[0] >= t1 - 1.0]
        last1 = [abs(c[1]) for c in recent]
        own = [abs(c[1]) for c in recent if not self.is_fallback(c)]
        frames = sum(1 for t in self.frames if t >= t1 - STALE_WINDOW_S) if self.frames is not None else None
        inferences = fallback_zeros = None
        if self.policy:
            fallback_zeros = sum(1 for c in self.cmd if c[0] >= t1 - STALE_WINDOW_S and self.is_fallback(c))
            if self.inferences is not None:
                inferences = sum(1 for t in self.inferences if t >= t1 - STALE_WINDOW_S)
        end_status = self.status[-1][1] if self.status else None
        return {
            "method": self.method,
            "pose_id": self.pose_id,
            "round": self.round,
            "outcome": self.outcome,
            "reason": self.reason,
            "duration_s": "%.3f" % (t1 - self.t0),
            "collided": int(any(bool(g.get("collided")) for _, g in gt)),
            "final_dist_m": "%.4f" % (BOX_X - last["x"]) if last else "",
            "final_lateral_m": "%.4f" % (last["y"] - BOX_Y) if last else "",
            "final_heading_deg": "%.2f" % wrap_deg(last["yaw"]) if last else "",
            "final_bumper_m": "%.4f" % last["bumper_to_box_m"] if last else "",
            "final_range_gt_m": "%.4f" % gt_range if gt_range is not None else "",
            "final_bearing_gt_deg": "%.2f" % gt_bearing if gt_bearing is not None else "",
            "zone_reached": int(time_to_zone is not None),
            "time_to_zone_s": "%.2f" % time_to_zone if time_to_zone is not None else "",
            "end_in_zone": int(bool(end_status and end_status.get("in_band") and end_status.get("aligned"))),
            "cmd_v_end": "%.4f" % statistics.median(tail) if tail else "",
            "cmd_v_min_last1s": "%.4f" % min(last1) if last1 else "",
            "own_v_min_last1s": "%.4f" % min(own) if own else "",
            "camera_frames_last1s": frames if frames is not None else "",
            "fresh_inferences_last1s": inferences if inferences is not None else "",
            "fallback_zeros_last1s": fallback_zeros if fallback_zeros is not None else "",
            "stale_last1s": stale_flag(self.method, frames, inferences, fallback_zeros),
            "bag": self.bag,
        }


def read_episodes(bag: Path) -> list[Episode]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    episodes: list[Episode] = []
    current: Episode | None = None
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as reader:
        connections = [c for c in reader.connections if c.topic in TOPICS + COUNTED_TOPICS]
        recorded = {c.topic for c in connections}
        for connection, t_ns, raw in reader.messages(connections=connections):
            t = t_ns * 1e-9
            topic = connection.topic
            if topic in COUNTED_TOPICS:  # only when they arrived matters
                if current is not None:
                    (current.frames if topic == CAMERA_TOPIC else current.inferences).append(t)
                continue
            msg = reader.deserialize(raw, connection.msgtype)
            if topic == "/episode/event":
                words = msg.data.split()
                if not words:
                    continue
                if words[0] == "start" and len(words) == 4:
                    if current is not None:
                        current.reason = "no end before the next start"
                        current.t1 = t
                        episodes.append(current)
                    current = Episode(
                        words[1], int(words[2]), int(words[3]), t, bag.name,
                        frames=[] if CAMERA_TOPIC in recorded else None,
                        inferences=[] if LATENCY_TOPIC in recorded else None,
                    )
                elif words[0] == "end" and current is not None and len(words) >= 2:
                    current.outcome = words[1]
                    current.reason = " ".join(words[2:])
                    current.t1 = t
                    episodes.append(current)
                    current = None
                continue
            if current is None:
                continue
            if topic == "/fake_robot/ground_truth":
                current.gt.append((t, json.loads(msg.data)))
            elif topic == "/dock_judge/status":
                current.status.append((t, json.loads(msg.data)))
            elif topic == "/cmd_vel_in":
                current.cmd.append((t, float(msg.linear.x), float(msg.angular.z)))
    if current is not None:
        episodes.append(current)
    return episodes


def cmd_episodes(args: argparse.Namespace) -> int:
    bags = [bag for path in args.bags for bag in analyze_trials.find_bags(path)]
    if not bags:
        print("no bags under %s" % " ".join(map(str, args.bags)), file=sys.stderr)
        return 2
    rows = [ep.row() for bag in bags for ep in read_episodes(bag)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d episodes from %d bag(s) to %s" % (len(rows), len(bags), args.out), file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------- table


def _median_iqr(values: Sequence[float], fmt: str = "%.1f") -> str:
    """median (IQR) with numpy's linear percentiles, as tools/analyze_trials.py reports them."""
    if not values:
        return "—"
    q1, med, q3 = np.percentile(np.asarray(values, dtype=np.float64), [25.0, 50.0, 75.0])
    return (fmt + " (" + fmt + "–" + fmt + ")") % (med, q1, q3)


def _median_min(counts: Sequence[int]) -> str:
    return "%g (min %d)" % (float(np.median(counts)), min(counts)) if counts else "—"


def own_v_min(row: dict[str, str]) -> str:
    """The smallest |v| of the controller's own requests in the last 1 s (own_v_min_last1s, without
    policy_runner's fallback zeros), or of every request (cmd_v_min_last1s) in an episodes_gt.csv
    written before own_v_min_last1s existed (v2, v3). "" when there was no request."""
    return (row["own_v_min_last1s"] if "own_v_min_last1s" in row else row.get("cmd_v_min_last1s")) or ""


def stop_commanded(row: dict[str, str], column: str | None = None) -> bool:
    """The smallest requested |v| in the last 1 s was below STOP_COMMANDED_V (episodes_gt.csv): over
    the controller's own requests (own_v_min), or over the given column (cmd_v_min_last1s: every
    request, fallback zeros included)."""
    value = (row.get(column) or "") if column else own_v_min(row)
    return bool(value) and float(value) < STOP_COMMANDED_V


def scored_gt(path: Path, scoring: analyze_trials.Scoring) -> dict[tuple[str, int, int], dict[str, str]]:
    """Ground-truth row of each scored slot: the first success/fail episode of the slot."""
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.is_file():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["outcome"] not in ("success", "fail"):
                continue
            key = (row["method"], int(row["pose_id"]), int(row["round"]))
            if key in scoring.scored and key not in out:
                out[key] = row
    return out


CAPTION_PRIMARY = (
    "**Primary: the judge's success rule** (in band, marker within 10°, robot and request still "
    "for 1 s; 30 s limit). Times are median (IQR) over successes."
)
CAPTION_SECONDARY = (
    "**Secondary: docking zone reached, stop ignored** (the judge's in_band and aligned held "
    "together for ≥ 1 s at some point, from `/dock_judge/status`; not the success rule). Time to "
    "zone and the requested speeds (`/cmd_vel_in`, before the gate) are median (IQR) over trials; "
    "the smallest requested |v| of the last 1 s and stop commanded (that |v| below 0.005 m/s; the "
    "judge calls a request still at |v| <= 0.005 m/s from v4 on, <= 0.02 m/s in v2/v3) count the "
    "controller's own requests: a policy arm's fallback zeros (exactly (0, 0), sent by policy_runner "
    "on a stale or missing frame or an inference error) are left out, except in files written before "
    "v4, which have every request only; collisions from fake_robot's ground truth."
)
CAPTION_ERRORS = (
    "**Final errors** at the end of each scored trial, median (IQR). Range error and bearing from "
    "the judge (trials.csv); GT = fake_robot ground truth (heading off square to the box face, "
    "sideways offset from the marker centre line, bumper-to-box gap, and at that pose the lidar "
    "front range the judge would measure without noise and the marker's true bearing)."
)
CAPTION_VALIDITY = (
    "**Validity check: stale inputs at the end** (results/sim_eval_v4_cleanstop/PREREG.md). Per "
    "scored trial, in the last 1 s before the end: camera frames recorded on "
    "`/camera/image_raw/compressed` (fake_robot sends 15 Hz) and, for a policy arm, fresh inferences "
    "(`/policy/latency`: one per policy_runner inference on a frame at most 0.2 s old, at 10 Hz), as "
    "median (min); and the trials with a fallback zero (a policy arm's request of exactly (0, 0): "
    "policy_runner sends it instead of the policy's output on a stale or missing frame, as on the tick "
    "that follows at once an inference that overran its period, or after an inference error). A "
    "success with fewer than 5 frames or fresh inferences, or with a fallback zero, is flagged: its "
    "stop may not be the controller's decision (servo_baseline also requests zero when no frame "
    "arrived for 0.5 s). The primary table scores flagged successes as successes; the last column "
    "counts them as failures. A success the check cannot read (no ground-truth row, or a bag without "
    "a counted topic) is listed as not checked and stays a success there."
)
VALIDITY_HEADER = (
    "| Condition | Method | Camera frames, last 1 s | Fresh inferences, last 1 s | Trials with a fallback zero, "
    "last 1 s | Successes flagged stale | Success k/n, flagged as failures |"
)


CAPTION_VS = (
    "**Paired with `%s`** on the same pose × round slots. b = `%s` succeeded and the method failed, "
    "c = the reverse; exact McNemar p; Δ = method − `%s` success rate with Newcombe's 95%% CI; the "
    "last two columns pair the zone-reached outcome the same way. Arms from different sessions "
    "share each condition's seed and schedule rule, not its noise draws."
)
VS_HEADER = (
    "| Condition | Pair | Pairs | b / c | McNemar exact p | Δ success (95% CI) | Zone b / c | Zone McNemar p |"
)


def _ci(k: int, n: int) -> str:
    if not n:
        return "—"
    lo, hi = analyze_trials.stats.wilson_interval(k, n, 0.95)
    return "%.0f–%.0f%%" % (100 * lo, 100 * hi)


def _paired(base: dict[tuple[int, int], bool], other: dict[tuple[int, int], bool]) -> tuple[str, str]:
    """'b / c' and the exact McNemar p of other against base over their common slots."""
    slots = sorted(set(base) & set(other))
    if not slots:
        return "—", "—"
    table = analyze_trials.stats.discordant_counts(
        np.array([base[s] for s in slots], dtype=np.int64),
        np.array([other[s] for s in slots], dtype=np.int64),
    )
    return "%d / %d" % (table.b, table.c), "%.3g" % analyze_trials.stats.mcnemar_exact(
        table.b, table.c
    ).p_exact


def cmd_table(args: argparse.Namespace) -> int:
    primary = [
        "| Condition | Method | Success k/n | Wilson 95% CI | b / c vs servo | McNemar exact p | "
        "Time to dock, s | Fails by reason |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    secondary = [
        "| Condition | Method | Zone reached k/n | Wilson 95% CI | b / c vs servo | McNemar exact p | "
        "Time to zone, s | Requested v, last 2 s, m/s | Min requested \\|v\\|, last 1 s, m/s | "
        "Stop commanded | Collisions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    vs_rows: dict[str, list[str]] = {m: [] for m in args.vs}
    errors = [
        "| Condition | Method | Range error, cm | \\|Bearing\\|, deg | GT \\|lateral\\|, cm | "
        "GT \\|heading off square\\|, deg | GT bumper gap, cm | GT lidar range, m | GT \\|bearing\\|, deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    validity = [VALIDITY_HEADER, "|---|---|---:|---:|---:|---:|---:|"]
    for cond in args.conditions:
        directory = args.results / cond
        trials_csv = directory / "trials.csv"
        if not trials_csv.is_file():
            continue
        trials, _ = analyze_trials.load_trials(trials_csv)
        scoring = analyze_trials.score_trials(trials)
        methods = analyze_trials.method_order({t.method for t in trials}, "servo")
        gt = scored_gt(directory / "episodes_gt.csv", scoring)
        zone_by_method = {
            m: {(p, r): bool(int(row["zone_reached"])) for (mm, p, r), row in gt.items() if mm == m}
            for m in methods
        }
        for method in methods:
            s = analyze_trials.summarize_method(method, scoring, TARGET_RANGE_M)
            if method == "servo":
                bc, p = "—", "—"
            else:
                comp = analyze_trials.compare_with_baseline(method, "servo", scoring)
                bc = "%d / %d" % (comp.table.b, comp.table.c) if comp.table else "—"
                p = ("%.3g" % comp.test.p_exact) if comp.test else "—"
            fails = Counter(c for c in s.categories.elements() if c not in ("success", "abort"))
            fail_text = ", ".join("%s %d" % (k, v) for k, v in sorted(fails.items())) or "none"
            if s.categories["abort"]:
                fail_text += " (+%d aborted, not scored)" % s.categories["abort"]
            primary.append(
                "| %s | %s | %d/%d | %s | %s | %s | %s | %s |"
                % (cond, method, s.k, s.n, _ci(s.k, s.n), bc, p, _median_iqr(s.dock_times_s), fail_text)
            )
            rows = [r for (m, _, _), r in gt.items() if m == method]
            zone = zone_by_method[method]
            zbc, zp = ("—", "—") if method == "servo" else _paired(zone_by_method.get("servo", {}), zone)
            to_zone = [float(r["time_to_zone_s"]) for r in rows if r["time_to_zone_s"]]
            cmd_v = [float(r["cmd_v_end"]) for r in rows if r["cmd_v_end"]]
            cmd_min = [float(own_v_min(r)) for r in rows if own_v_min(r)]
            collisions = sum(int(r["collided"]) for r in rows)
            stops = sum(1 for r in rows if stop_commanded(r))
            k_zone = sum(zone.values())
            secondary.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    cond,
                    method,
                    "%d/%d" % (k_zone, len(zone)) if zone else "—",
                    _ci(k_zone, len(zone)),
                    zbc,
                    zp,
                    _median_iqr(to_zone),
                    _median_iqr(cmd_v, "%.3f"),
                    _median_iqr(cmd_min, "%.3f"),
                    "%d/%d" % (stops, len(rows)) if rows else "—",
                    "%d/%d" % (collisions, len(rows)) if rows else "—",
                )
            )
            lateral = [abs(float(r["final_lateral_m"])) * 100 for r in rows if r["final_lateral_m"]]
            heading = [abs(float(r["final_heading_deg"])) for r in rows if r["final_heading_deg"]]
            bumper = [float(r["final_bumper_m"]) * 100 for r in rows if r["final_bumper_m"]]
            gt_range = [float(r["final_range_gt_m"]) for r in rows if r.get("final_range_gt_m")]
            gt_bearing = [abs(float(r["final_bearing_gt_deg"])) for r in rows if r.get("final_bearing_gt_deg")]
            errors.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    cond,
                    method,
                    _median_iqr(s.range_errors_cm),
                    _median_iqr(s.bearing_errors_deg),
                    _median_iqr(lateral),
                    _median_iqr(heading),
                    _median_iqr(bumper),
                    _median_iqr(gt_range, "%.4f"),  # 4 decimals: P5 of the v4 pre-registration reads it
                    _median_iqr(gt_bearing),
                )
            )
            # The validity check: a scored success is flagged when its row says stale_last1s 1;
            # one without the check (no row, or a bag without the counted topics) is listed as
            # not checked and stays a success in the last column.
            flags = [gt[key].get("stale_last1s", "") if key in gt else ""
                     for key, t in scoring.scored.items() if key[0] == method and t.success]
            flagged, unchecked = flags.count("1"), sum(1 for f in flags if f not in ("0", "1"))
            frames = [int(r["camera_frames_last1s"]) for r in rows if r.get("camera_frames_last1s")]
            fresh = [int(r["fresh_inferences_last1s"]) for r in rows if r.get("fresh_inferences_last1s")]
            fallback = [int(r["fallback_zeros_last1s"]) for r in rows if r.get("fallback_zeros_last1s")]
            validity.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    cond,
                    method,
                    _median_min(frames),
                    _median_min(fresh),
                    "%d/%d" % (sum(1 for f in fallback if f), len(fallback)) if fallback else "—",
                    "%d%s" % (flagged, " (%d not checked)" % unchecked if unchecked else ""),
                    "%d/%d" % (s.k - flagged, s.n) if s.n else "—",
                )
            )
            for base in args.vs:  # each later method in the table order, so no pair is listed twice
                if method == "servo" or base not in methods or methods.index(method) <= methods.index(base):
                    continue
                comp = analyze_trials.compare_with_baseline(method, base, scoring)
                if comp.table is None:
                    continue
                zbc, zp = _paired(zone_by_method[base], zone)
                vs_rows[base].append(
                    "| %s | %s vs %s | %d | %d / %d | %.3g | %+.1f pp (%+.1f to %+.1f) | %s | %s |"
                    % (cond, method, base, comp.table.n_pairs, comp.table.b, comp.table.c, comp.test.p_exact,
                       100 * comp.diff.estimate, 100 * comp.diff.low, 100 * comp.diff.high, zbc, zp)
                )
    vs_tables: list[str] = []
    for base, rows in vs_rows.items():
        if rows:
            vs_tables += ["", CAPTION_VS % (base, base, base), "", VS_HEADER, "|---|---|---:|---:|---:|---:|---:|---:|"]
            vs_tables += rows
    text = "\n".join(
        ["<!-- generated by scripts/sim_eval_summary.py table; SIMULATION results -->", ""]
        + [CAPTION_PRIMARY, ""]
        + primary
        + ["", CAPTION_SECONDARY, ""]
        + secondary
        + ["", CAPTION_ERRORS, ""]
        + errors
        + vs_tables
        + ["", CAPTION_VALIDITY, ""]
        + validity
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(text)
    return 0


# ---------------------------------------------------------------------------- stops


def _where(ranges: Sequence[str]) -> str:
    """median (IQR, range) of the final_range_gt_m values given, at 4 decimals; "none" without any."""
    values = [float(r) for r in ranges if r]
    if not values:
        return "none"
    q1, med, q3 = np.percentile(np.asarray(values, dtype=np.float64), [25.0, 50.0, 75.0])
    return "median %.4f m (IQR %.4f-%.4f, range %.4f-%.4f; %d with ground truth)" % (
        med, q1, q3, min(values), max(values), len(values))


def cmd_stops(args: argparse.Namespace) -> int:
    """One method's scored trials pooled over conditions: stop commanded by its own requests (as in
    the table) and the ground-truth final lidar range of those trials; then the same over every
    request as a sensitivity line. n is every scored trial of the method in trials.csv."""
    n, missing, no_gt = 0, [], []
    own: list[str] = []  # final_range_gt_m of each trial that commanded a stop with its own requests
    every: list[str] = []  # the same over every request, policy_runner's fallback zeros included
    for cond in args.conditions:
        trials_csv = args.results / cond / "trials.csv"
        if not trials_csv.is_file():
            missing.append(cond)
            continue
        trials, _ = analyze_trials.load_trials(trials_csv)
        scoring = analyze_trials.score_trials(trials)
        gt = scored_gt(args.results / cond / "episodes_gt.csv", scoring)
        for key in scoring.scored:
            if key[0] != args.method:
                continue
            n += 1
            row = gt.get(key)
            if row is None:
                no_gt.append("%s pose %d round %d" % (cond, key[1], key[2]))
                continue
            if stop_commanded(row):
                own.append(row.get("final_range_gt_m") or "")
            if stop_commanded(row, "cmd_v_min_last1s"):
                every.append(row.get("final_range_gt_m") or "")
    if missing:
        print("stops: no trials.csv for %s under %s" % (", ".join(missing), args.results), file=sys.stderr)
        return 2

    def share(k: int) -> str:
        return "%.1f %%" % (100.0 * k / n) if n else "n/a"

    lines = [
        "%s, %s: stop commanded in %d of %d scored trials (%s)"
        % (args.method, " + ".join(args.conditions), len(own), n, share(len(own))),
        "ground-truth final lidar range of those %d trials: %s" % (len(own), _where(own)),
        "sensitivity, every request counted (cmd_v_min_last1s, policy_runner's fallback zeros included): "
        "stop commanded in %d of %d scored trials (%s); ground-truth final lidar range of those: %s"
        % (len(every), n, share(len(every)), _where(every)),
    ]
    if no_gt:
        lines.append(
            "INCOMPLETE: %d of the %d scored trials have no ground-truth row in episodes_gt.csv and are not "
            "counted as stops above: %s" % (len(no_gt), n, "; ".join(no_gt))
        )
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if no_gt:
        print("stops: %d scored trial(s) without a ground-truth row" % len(no_gt), file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------- merge

SCHEDULE_COLUMNS = ("trial", "round", "block", "method", "pose_id", "event")  # tools/make_schedule.py


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_conditions(inputs: Sequence[Path], out: Path) -> dict[str, int]:
    """Concatenate trials.csv, episodes_gt.csv and schedule.csv of the input folders into out.

    ValueError if a method has trial rows in more than one input or the trial headers differ.
    Returns the number of rows written per file name.
    """
    trial_header: list[str] | None = None
    trials: list[dict[str, str]] = []
    schedule: list[dict[str, str]] = []
    episodes: list[dict[str, str]] = []
    owner: dict[str, Path] = {}
    gt_missing = []
    for folder in inputs:
        header, rows = _read_csv(folder / "trials.csv")
        if trial_header is None:
            trial_header = header
        elif header != trial_header:
            raise ValueError("%s/trials.csv has columns %s, not %s" % (folder, header, trial_header))
        for method in dict.fromkeys(r["method"] for r in rows):
            if method in owner:
                raise ValueError("method %r has trials in both %s and %s" % (method, owner[method], folder))
            owner[method] = folder
        trials += rows
        if (folder / "schedule.csv").is_file():
            schedule += _read_csv(folder / "schedule.csv")[1]
        else:  # the slots that were run
            slots = dict.fromkeys((r["method"], r["pose_id"], r["round"]) for r in rows)
            schedule += [
                {"trial": "", "round": r, "block": "", "method": m, "pose_id": p, "event": "start %s %s %s" % (m, p, r)}
                for m, p, r in slots
            ]
        if (folder / "episodes_gt.csv").is_file():
            episodes += _read_csv(folder / "episodes_gt.csv")[1]
        else:
            gt_missing.append(str(folder))
    if trial_header is None:
        raise ValueError("nothing to merge")
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "trials.csv", trial_header, trials)
    _write_csv(out / "schedule.csv", SCHEDULE_COLUMNS, schedule)
    written = {"trials.csv": len(trials), "schedule.csv": len(schedule)}
    if gt_missing:
        print("merge: no episodes_gt.csv in %s; episodes_gt.csv not written" % ", ".join(gt_missing), file=sys.stderr)
        (out / "episodes_gt.csv").unlink(missing_ok=True)
    else:
        _write_csv(out / "episodes_gt.csv", EPISODE_COLUMNS, episodes)
        written["episodes_gt.csv"] = len(episodes)
    return written


def cmd_merge(args: argparse.Namespace) -> int:
    try:
        written = merge_conditions(args.inputs, args.out)
    except (OSError, ValueError, KeyError) as exc:
        print("merge: %s" % exc, file=sys.stderr)
        return 2
    print("merge: %s -> %s" % (", ".join("%d %s" % (n, f) for f, n in written.items()), args.out), file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("episodes", help="per-episode ground-truth metrics from the bags")
    p.add_argument("--bags", type=Path, nargs="+", required=True, help="bags or directories of bags")
    p.add_argument("--poses", type=Path, help="start poses file (recorded for reference only)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_episodes)
    p = sub.add_parser("table", help="summary table over condition directories")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--vs", action="append", default=[], metavar="METHOD",
                   help="also pair METHOD with each method after it in the table (repeatable)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_table)
    p = sub.add_parser("stops", help="one method's stop-commanded trials over conditions, and where they ended")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", type=Path, help="also write the lines here")
    p.set_defaults(func=cmd_stops)
    p = sub.add_parser("merge", help="condition folders of separate sessions -> one folder")
    p.add_argument("inputs", type=Path, nargs="+", help="folders with trials.csv (+ schedule.csv, episodes_gt.csv)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_merge)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
