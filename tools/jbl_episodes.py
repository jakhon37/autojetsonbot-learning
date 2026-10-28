#!/usr/bin/env python3
"""Per-episode metadata of a converted dataset (meta/jbl_episodes.jsonl) and episode-level splits.

    python tools/jbl_episodes.py --root data/jbl_dock                    # table + outcome counts
    python tools/jbl_episodes.py --root data/jbl_dock --split --val-fraction 0.2 --split-by pose_id

One JSON object per LeRobot episode, written by bag_to_lerobot.py:

    episode_index  LeRobot episode index (the join key)
    bag            bag path relative to the converted input
    method, pose_id, round, outcome, reason    from /episode/event (null when the bag had none)
    duration_s     episode length in bag time (start event -> end event, or whole bag)
    num_frames     frames written at the dataset fps
    start_time_ns  bag log time of the episode start
    max_image_age_s  oldest camera frame used at any tick (camera stalls show up here)

Splits are by episode, never by frame: neighbouring frames of one episode are nearly identical,
so a frame-level split would leak training data into validation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

EPISODES_RELPATH = Path("meta") / "jbl_episodes.jsonl"
FIELDS = (
    "episode_index",
    "bag",
    "method",
    "pose_id",
    "round",
    "outcome",
    "reason",
    "duration_s",
    "num_frames",
    "start_time_ns",
    "max_image_age_s",
)


def episodes_path(root: str | Path) -> Path:
    return Path(root) / EPISODES_RELPATH


def write_episode_meta(root: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write records (sorted by episode_index) as JSON lines; returns the file path."""
    rows = sorted((dict(r) for r in records), key=lambda r: r["episode_index"])
    _check_unique(rows)
    path = episodes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def read_episode_meta(root: str | Path) -> list[dict[str, Any]]:
    """Records of meta/jbl_episodes.jsonl sorted by episode_index; FileNotFoundError if absent."""
    path = episodes_path(root)
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: %s" % (path, lineno, exc)) from None
            if not isinstance(row, dict) or not isinstance(row.get("episode_index"), int):
                raise ValueError("%s:%d: expected an object with an integer episode_index" % (path, lineno))
            rows.append(row)
    rows.sort(key=lambda r: r["episode_index"])
    _check_unique(rows)
    return rows


def load_episode_table(root: str | Path, total_episodes: int | None = None) -> list[dict[str, Any]]:
    """Metadata records, or bare {"episode_index": i} records for datasets without the file.

    With ``total_episodes`` the records are checked against the LeRobot episode count.
    """
    try:
        rows = read_episode_meta(root)
    except FileNotFoundError:
        if total_episodes is None:
            raise
        return [{"episode_index": i} for i in range(total_episodes)]
    if total_episodes is not None:
        indices = [r["episode_index"] for r in rows]
        if indices != list(range(total_episodes)):
            raise ValueError(
                "%s lists episodes %s but the dataset has %d episodes"
                % (episodes_path(root), _short(indices), total_episodes)
            )
    return rows


def filter_episodes(
    records: Sequence[Mapping[str, Any]],
    outcomes: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Records whose outcome / method is in the given lists ("none" matches a missing value)."""

    def keep(value: Any, allowed: Sequence[str] | None) -> bool:
        if allowed is None:
            return True
        return ("none" if value is None else str(value)) in allowed

    return [dict(r) for r in records if keep(r.get("outcome"), outcomes) and keep(r.get("method"), methods)]


def split_episodes(
    records: Sequence[Mapping[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 0,
    group_by: str | None = None,
) -> tuple[list[int], list[int]]:
    """(train, val) episode indices, deterministic for a given seed.

    ``group_by`` (e.g. "pose_id") keeps all episodes of a group on one side, so validation
    measures generalisation to start poses never seen in training. At least one episode stays
    in training; validation is empty only when there is a single episode (or group) or
    ``val_fraction`` is 0.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1), got %r" % val_fraction)
    by_group: dict[Any, list[int]] = {}
    for r in records:
        key = r.get(group_by) if group_by else r["episode_index"]
        by_group.setdefault(key, []).append(int(r["episode_index"]))
    groups = sorted(by_group, key=lambda k: (k is None, str(k)))
    random.Random(seed).shuffle(groups)
    n = sum(len(v) for v in by_group.values())
    target = int(round(n * val_fraction))
    if val_fraction > 0.0 and len(groups) >= 2:
        target = max(target, 1)
    val: list[int] = []
    for key in groups[:-1]:  # the last group always trains
        if len(val) >= target:
            break
        val.extend(by_group[key])
    val_set = set(val)
    train = sorted(i for v in by_group.values() for i in v if i not in val_set)
    return train, sorted(val)


def _check_unique(rows: Sequence[Mapping[str, Any]]) -> None:
    seen = Counter(r["episode_index"] for r in rows)
    dupes = sorted(i for i, c in seen.items() if c > 1)
    if dupes:
        raise ValueError("duplicate episode_index in episode metadata: %s" % dupes)


def _short(indices: Sequence[int]) -> str:
    return str(list(indices)) if len(indices) <= 12 else "[%s ... %s] (%d)" % (indices[0], indices[-1], len(indices))


def _csv(text: str | None) -> list[str] | None:
    return None if text is None else [x.strip() for x in text.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show episode metadata and episode-level splits.")
    parser.add_argument("--root", required=True, type=Path, help="local LeRobot dataset directory")
    parser.add_argument("--outcomes", help="comma list to keep, e.g. success,none")
    parser.add_argument("--methods", help="comma list to keep, e.g. teleop")
    parser.add_argument("--split", action="store_true", help="print a train/val split as JSON")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-by", default=None, help="group key kept on one side, e.g. pose_id")
    args = parser.parse_args(argv)

    rows = filter_episodes(read_episode_meta(args.root), _csv(args.outcomes), _csv(args.methods))
    if args.split:
        train, val = split_episodes(rows, args.val_fraction, args.seed, args.split_by)
        print(json.dumps({"train": train, "val": val}))
        return 0
    print("%4s  %-28s %-8s %4s %5s  %-8s %7s %6s"
          % ("ep", "bag", "method", "pose", "round", "outcome", "dur_s", "frames"))
    for r in rows:
        print(
            "%4d  %-28s %-8s %4s %5s  %-8s %7.2f %6d"
            % (r["episode_index"], str(r.get("bag"))[-28:], r.get("method"), r.get("pose_id"), r.get("round"),
               r.get("outcome"), r.get("duration_s") or 0.0, r.get("num_frames") or 0)
        )
    counts = Counter(str(r.get("outcome")) for r in rows)
    frames = sum(r.get("num_frames") or 0 for r in rows)
    outcomes = ", ".join("%s=%d" % kv for kv in sorted(counts.items()))
    print("%d episodes, %d frames; outcomes: %s" % (len(rows), frames, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
