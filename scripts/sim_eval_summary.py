#!/usr/bin/env python3
"""Ground-truth episode metrics and the summary table of the harder simulation evaluation.

    python scripts/sim_eval_summary.py episodes --bags sim_work/eval_v2/jazzy/noise/bags \
        --out results/sim_eval_v2/noise/episodes_gt.csv
    python scripts/sim_eval_summary.py table --results results/sim_eval_v2 \
        --conditions clean_ref noise wide hfov62 --out results/sim_eval_v2/SUMMARY_table.md

episodes: one row per episode in the bags (every start ... end on /episode/event, aborts
included) with what only the simulator knows, from /fake_robot/ground_truth, plus the judge's
and the controller's view from /dock_judge/status and /cmd_vel_in:
    collided          the footprint touched the box or a wall (fake_robot's latched flag; the
                      first 0.3 s after the start are skipped, the teleport may still be pending)
    final_*           at the last ground-truth sample before the end: base_link distance to the
                      box face plane, lateral offset from the marker centre line (+ = left when
                      facing the box), heading off square (deg, + = turned left) and bumper gap
    zone_reached      the judge's in_band and aligned held together for >= 1.0 s at some point,
                      whatever "stopped" said (the success rule without the stop requirement);
                      time_to_zone_s = when that first happened, from the start
    end_in_zone       in_band and aligned on the last judge status before the end
    cmd_v_end         median requested /cmd_vel_in v over the last 2 s before the end (m/s)
    cmd_v_min_last1s  smallest |requested v| in the last 1 s before the end: 0 when the
                      controller commanded a stop (the servo does once inside its band)
The default box of config/sim.yaml is assumed (front face centre (1.5, 0), facing -x).

table: per condition directory (trials.csv, episodes_gt.csv), per method: success k/n with the
Wilson 95% CI, the exact McNemar test against servo on pose x round pairs (scoring exactly as
tools/analyze_trials.py), median time to dock, failure reasons, final errors and the
ground-truth columns above.
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
sys.path.insert(0, str(REPO / "tools"))
import analyze_trials  # noqa: E402

BOX_X, BOX_Y = 1.5, 0.0  # config/sim.yaml; the face looks along -x, so facing it squarely is yaw 0
TELEPORT_SETTLE_S = 0.3
ZONE_HOLD_S = 1.0
CMD_TAIL_S = 2.0
TARGET_RANGE_M = 0.25
STOP_COMMANDED_V = 0.005  # m/s: "commanded a stop" = min |requested v| in the last 1 s below this
TOPICS = ("/episode/event", "/fake_robot/ground_truth", "/dock_judge/status", "/cmd_vel_in")
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
    "zone_reached",
    "time_to_zone_s",
    "end_in_zone",
    "cmd_v_end",
    "cmd_v_min_last1s",
    "bag",
)


def wrap_deg(angle_rad: float) -> float:
    return math.degrees((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


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
    cmd: list[tuple[float, float]] = field(default_factory=list)

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
        tail = [v for t, v in self.cmd if t >= t1 - CMD_TAIL_S]
        last1 = [abs(v) for t, v in self.cmd if t >= t1 - 1.0]
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
            "zone_reached": int(time_to_zone is not None),
            "time_to_zone_s": "%.2f" % time_to_zone if time_to_zone is not None else "",
            "end_in_zone": int(bool(end_status and end_status.get("in_band") and end_status.get("aligned"))),
            "cmd_v_end": "%.4f" % statistics.median(tail) if tail else "",
            "cmd_v_min_last1s": "%.4f" % min(last1) if last1 else "",
            "bag": self.bag,
        }


def read_episodes(bag: Path) -> list[Episode]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    episodes: list[Episode] = []
    current: Episode | None = None
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as reader:
        connections = [c for c in reader.connections if c.topic in TOPICS]
        for connection, t_ns, raw in reader.messages(connections=connections):
            t = t_ns * 1e-9
            msg = reader.deserialize(raw, connection.msgtype)
            topic = connection.topic
            if topic == "/episode/event":
                words = msg.data.split()
                if not words:
                    continue
                if words[0] == "start" and len(words) == 4:
                    if current is not None:
                        current.reason = "no end before the next start"
                        current.t1 = t
                        episodes.append(current)
                    current = Episode(words[1], int(words[2]), int(words[3]), t, bag.name)
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
                current.cmd.append((t, float(msg.linear.x)))
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
    "stop commanded = the smallest requested |v| in the last 1 s was below 0.005 m/s (the judge's "
    "threshold for still is 0.02 m/s); collisions from fake_robot's ground truth."
)
CAPTION_ERRORS = (
    "**Final errors** at the end of each scored trial, median (IQR). Range error and bearing from "
    "the judge (trials.csv); GT = fake_robot ground truth (heading off square to the box face, "
    "sideways offset from the marker centre line, bumper-to-box gap)."
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
    errors = [
        "| Condition | Method | Range error, cm | \\|Bearing\\|, deg | GT \\|lateral\\|, cm | "
        "GT \\|heading off square\\|, deg | GT bumper gap, cm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
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
                fail_text += " (+%d aborted, re-run)" % s.categories["abort"]
            primary.append(
                "| %s | %s | %d/%d | %s | %s | %s | %s | %s |"
                % (cond, method, s.k, s.n, _ci(s.k, s.n), bc, p, _median_iqr(s.dock_times_s), fail_text)
            )
            rows = [r for (m, _, _), r in gt.items() if m == method]
            zone = zone_by_method[method]
            zbc, zp = ("—", "—") if method == "servo" else _paired(zone_by_method.get("servo", {}), zone)
            to_zone = [float(r["time_to_zone_s"]) for r in rows if r["time_to_zone_s"]]
            cmd_v = [float(r["cmd_v_end"]) for r in rows if r["cmd_v_end"]]
            cmd_min = [float(r["cmd_v_min_last1s"]) for r in rows if r.get("cmd_v_min_last1s")]
            collisions = sum(int(r["collided"]) for r in rows)
            stops = sum(1 for v in cmd_min if v < STOP_COMMANDED_V)
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
            errors.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    cond,
                    method,
                    _median_iqr(s.range_errors_cm),
                    _median_iqr(s.bearing_errors_deg),
                    _median_iqr(lateral),
                    _median_iqr(heading),
                    _median_iqr(bumper),
                )
            )
    text = "\n".join(
        ["<!-- generated by scripts/sim_eval_summary.py table; SIMULATION results -->", ""]
        + [CAPTION_PRIMARY, ""]
        + primary
        + ["", CAPTION_SECONDARY, ""]
        + secondary
        + ["", CAPTION_ERRORS, ""]
        + errors
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(text)
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
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
