#!/usr/bin/env python3
"""Counterbalanced trial schedule for the paired docking evaluation (docs/PROTOCOL.md).

    python tools/make_schedule.py --methods servo,bc,act --poses 10 --rounds 3 --seed 2026 \
        --out trials/schedule.csv --markdown trials/schedule.md
    python tools/make_schedule.py --pose-table ros2_ws/src/jbl_nodes/config/start_poses.yaml

Every method runs every pose once per round, so each (pose_id, round) slot yields one paired
observation per method. Within round r (1-based) the pose order is a shuffle seeded with
(seed, r), and the method order comes from a cyclic Latin square over the methods:

    interleaved (default)  poses are blocks; block j of round r runs the methods in Latin-square
                           row (j + r - 1) mod M, back to back. Paired trials are minutes apart,
                           and over M rounds (or any multiple) each method is in each position
                           of a block equally often, so battery, temperature and lighting drift
                           cannot favour one method. Needs every method ready at once (servo and
                           one policy_runner per model with method_filter).
    blocked                round r runs whole methods in Latin-square row (r - 1) mod M, each
                           over all poses in that round's order. Fewer model switches, but paired
                           trials are up to a third of a round apart; use it only if the robot
                           cannot hold all policies at once, and say so in the report.

The schedule is a pure function of the arguments: rerunning the command reproduces it.

CSV columns: trial, round, block, method, pose_id, event
    trial  1-based position in the whole session; block = 1-based position of the pose in the
    round; event = the /episode/event payload, i.e. `episode start <method> <pose_id> <round>`.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ORDERS = ("interleaved", "blocked")
CSV_COLUMNS = ("trial", "round", "block", "method", "pose_id", "event")
DEFAULT_METHODS = ("servo", "bc", "act")
_METHOD_RE = re.compile(r"^[A-Za-z0-9_.-]+$")  # the episode CLI accepts exactly these
HEADING_TIP_M = 0.30  # length of the heading line taped at each start pose


@dataclass(frozen=True)
class ScheduledTrial:
    trial: int
    round: int
    block: int
    method: str
    pose_id: int

    @property
    def event(self) -> str:
        return "start %s %d %d" % (self.method, self.pose_id, self.round)


def latin_square(size: int) -> list[list[int]]:
    """Cyclic Latin square: row i is (i, i+1, ..., i+size-1) mod size."""
    return [[(i + j) % size for j in range(size)] for i in range(size)]


def round_pose_order(pose_ids: Sequence[int], seed: int, round_number: int) -> list[int]:
    # String seeds hash deterministically across Python versions and never collide between
    # (seed, round) pairs the way seed + round would.
    rng = random.Random("jbl-schedule/%d/round-%d" % (seed, round_number))
    order = list(pose_ids)
    rng.shuffle(order)
    return order


def make_schedule(
    methods: Sequence[str],
    pose_ids: Sequence[int],
    rounds: int,
    seed: int,
    order: str = "interleaved",
) -> list[ScheduledTrial]:
    methods = list(methods)
    pose_ids = list(pose_ids)
    if not methods:
        raise ValueError("need at least one method")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique, got %s" % methods)
    bad = [m for m in methods if not _METHOD_RE.match(m)]
    if bad:
        raise ValueError("method names must be letters, digits, '_', '-' or '.': %s" % bad)
    if not pose_ids or len(set(pose_ids)) != len(pose_ids) or min(pose_ids) < 0:
        raise ValueError("pose ids must be unique non-negative integers")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if order not in ORDERS:
        raise ValueError("order must be one of %s" % ", ".join(ORDERS))

    square = latin_square(len(methods))
    trials: list[ScheduledTrial] = []
    for r in range(1, rounds + 1):
        poses = round_pose_order(pose_ids, seed, r)
        if order == "interleaved":
            slots = [
                (block, methods[m], pose)
                for block, pose in enumerate(poses, start=1)
                for m in square[(block - 1 + r - 1) % len(methods)]
            ]
        else:
            slots = [
                (block, methods[m], pose)
                for m in square[(r - 1) % len(methods)]
                for block, pose in enumerate(poses, start=1)
            ]
        for block, method, pose in slots:
            trials.append(ScheduledTrial(len(trials) + 1, r, block, method, pose))
    return trials


def write_csv(trials: Sequence[ScheduledTrial], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for t in trials:
            writer.writerow([t.trial, t.round, t.block, t.method, t.pose_id, t.event])


def render_markdown(trials: Sequence[ScheduledTrial], seed: int, order: str) -> str:
    """Printable run sheet: one table per round with blank columns to fill in by hand."""
    methods = list(dict.fromkeys(t.method for t in trials))
    rounds = sorted({t.round for t in trials})
    lines = [
        "# Trial schedule",
        "",
        "Methods %s; %d poses; %d rounds; order %s; seed %d; %d trials."
        % (", ".join(methods), len({t.pose_id for t in trials}), len(rounds), order, seed, len(trials)),
        "Run each row with `ros2 run jbl_nodes episode --timeout 5 <event>`; do not reorder rows.",
        "",
    ]
    for r in rounds:
        lines += [
            "## Round %d" % r,
            "",
            "Battery at start: ______ V  Time: ______  Lights: ______",
            "",
            "| # | Pose | Method | Event | Result | Video time | Notes |",
            "|---:|---:|---|---|---|---|---|",
        ]
        lines += [
            "| %d | %d | %s | `%s` | | | |" % (t.trial, t.pose_id, t.method, t.event)
            for t in trials
            if t.round == r
        ]
        lines.append("")
    return "\n".join(lines)


def pose_marking_table(yaml_path: Path, tip_m: float = HEADING_TIP_M) -> str:
    """Floor-marking table for config/start_poses.yaml (the YAML is the source of truth).

    A = robot centre (base_link, the midpoint between the drive wheels); B = the point tip_m
    ahead of A along the start heading. Both are given as (distance from the box face plane,
    lateral offset from the marker centre line, positive = left when looking at the box).
    """
    import yaml  # only this subcommand needs PyYAML

    with open(yaml_path) as f:
        poses = yaml.safe_load(f)["start_poses"]
    lines = [
        "| Pose | A: distance m | A: lateral m | Heading deg | B: distance m | B: lateral m |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(poses, key=lambda p: int(p["pose_id"])):
        d, lat, h = float(p["distance_m"]), float(p["lateral_m"]), float(p["heading_deg"])
        rad = math.radians(h)
        lines.append(
            "| %d | %.2f | %+.2f | %s | %.2f | %+.2f |"
            % (
                int(p["pose_id"]),
                d,
                lat,
                "%+.0f" % h if round(h) else "0",
                d - tip_m * math.cos(rad),
                lat + tip_m * math.sin(rad),
            )
        )
    return "\n".join(lines)


def _parse_methods(text: str) -> list[str]:
    return [m.strip() for m in text.split(",") if m.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated (default %(default)s)")
    parser.add_argument("--poses", type=int, default=10, help="pose ids 0..N-1 (default %(default)s)")
    parser.add_argument("--rounds", type=int, default=3, help="default %(default)s")
    parser.add_argument("--seed", type=int, default=2026, help="default %(default)s")
    parser.add_argument("--order", choices=ORDERS, default="interleaved")
    parser.add_argument("--out", type=Path, help="schedule CSV (default: print CSV rows to stdout)")
    parser.add_argument("--markdown", type=Path, help="also write the printable run sheet here")
    parser.add_argument(
        "--pose-table", type=Path, metavar="START_POSES_YAML", help="print the floor-marking table and exit"
    )
    args = parser.parse_args(argv)

    if args.pose_table is not None:
        print(pose_marking_table(args.pose_table))
        return 0
    try:
        trials = make_schedule(_parse_methods(args.methods), range(args.poses), args.rounds, args.seed, args.order)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out is None:
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows([t.trial, t.round, t.block, t.method, t.pose_id, t.event] for t in trials)
    else:
        write_csv(trials, args.out)
        print("wrote %d trials to %s" % (len(trials), args.out), file=sys.stderr)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(trials, args.seed, args.order) + "\n")
        print("wrote run sheet to %s" % args.markdown, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
