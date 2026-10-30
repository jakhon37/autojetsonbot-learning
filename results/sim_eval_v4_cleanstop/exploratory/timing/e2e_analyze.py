"""Summarise the end-to-end check bags like the v3/v4 session bags (timing_bags.py) plus the
validity check's per-episode fallback zeros in the last 1 s (scripts/sim_eval_summary.py).

    python e2e_analyze.py REPO E2E_DIR NAME [NAME ...]

Per configuration, one summary line, then (added after run 1) one line per episode: pose, round,
outcome, fallback zeros over requests, fallback zeros in the last 1 s, the validity flag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_bags  # noqa: E402


def main(repo: str, e2e: str, names) -> int:
    sys.path.insert(0, str(Path(repo) / "scripts"))
    import sim_eval_summary as ses

    for name in names:
        bag = Path(e2e) / name / "bag"
        r = timing_bags.read_bag(bag, "bc_v4")
        eps = [e for e in ses.read_episodes(bag) if e.method == "bc_v4"]
        rows = [e.row() for e in eps]
        with_zero = sum(1 for row in rows if (row["fallback_zeros_last1s"] or 0) > 0)
        outcomes = {}
        for row in rows:
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        print(
            "%-14s n=%4d  pre %5.1f/%5.1f/%5.1f  inf %5.1f/%5.1f/%5.1f  tot %5.1f/%5.1f/%5.1f  "
            "age %5.1f/%5.1f/%5.1f  int p99 %5.1f (>150: %d of %d)  zeros %d/%d ticks (%.2f %%)  "
            "episodes %d %s, with a fallback zero in the last 1 s: %d"
            % (
                name, r["n"], *r["preprocess_ms"], *r["infer_ms"], *r["total_ms"], *r["image_age_ms"],
                r["interval_ms"][2], r["interval_gt_150"], r["n_intervals"], r["zeros"], r["ticks"],
                100.0 * r["zeros"] / max(r["ticks"], 1), len(rows), outcomes, with_zero,
            )
        )
        for e, row in zip(eps, rows):
            print(
                "  pose %d round %d: %-7s fallback zeros %d / %d, in the last 1 s %s, flagged %s"
                % (e.pose_id, e.round, row["outcome"], sum(1 for c in e.cmd if e.is_fallback(c)), len(e.cmd),
                   row["fallback_zeros_last1s"], row["stale_last1s"])
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
