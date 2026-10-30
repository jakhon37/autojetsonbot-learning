"""Fallback zeros per policy episode in the v3 and v4 session bags, and the end-to-end check
against v4's own session in the same condition.

    python episode_zeros.py REPO [--observed NAME=ZEROS/TICKS/WITH_LAST1S/EPISODES ...]

Per session bag (timing_bags.SESSIONS): the policy arm's episodes, its exact (0, 0) requests on
/cmd_vel_in over all its requests, their range per episode, the episodes with a fallback zero in
the last 1 s before the end event and the successes the validity check flags (both from
scripts/sim_eval_summary.py, as SUMMARY.md section 3 counts them).

Each --observed run (an end-to-end check configuration, numbers from e2e_summary.txt) is set
against the reference bag, v4's noise session (bc_v4, 30 episodes):
- the fallback-zero rate of N episodes drawn at random from the reference's, without
  replacement (20,000 draws, seed 0): the share of draws at or below the observed rate, and the
  draws' minimum and 5th percentile;
- the episodes with a last-1 s fallback zero, observed against the reference's, by Fisher's exact
  test, one-sided (fewer than the reference): P(X <= observed) under the hypergeometric law.
"""

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_bags import CONDITIONS, SESSIONS  # noqa: E402

REFERENCE = ("v4", "sim_work/v4", "bc_v4", "noise")


def episodes(repo: Path, work: str, method: str, condition: str):
    sys.path.insert(0, str(repo / "scripts"))
    import sim_eval_summary as ses

    out = []
    for bag in sorted((repo / work / "eval_v2/jazzy" / condition / "bags").glob("seg_*")):
        if not bag.is_dir():
            continue
        for e in ses.read_episodes(bag):
            if e.method != method:
                continue
            row = e.row()
            out.append(
                {
                    "bag": bag.name,
                    "pose_id": e.pose_id,
                    "round": e.round,
                    "zeros": sum(1 for c in e.cmd if e.is_fallback(c)),
                    "ticks": len(e.cmd),
                    "zeros_last1s": int(row["fallback_zeros_last1s"] or 0),
                    "outcome": row["outcome"],
                    "flagged": row["outcome"] == "success" and row["stale_last1s"] == 1,
                }
            )
    return out


def fisher_less(k_obs: int, n_obs: int, k_ref: int, n_ref: int) -> float:
    """One-sided Fisher exact p: P(X <= k_obs), X ~ hypergeometric(n_obs + n_ref, k_obs + k_ref, n_obs)."""
    total, marked = n_obs + n_ref, k_obs + k_ref
    return sum(comb(marked, x) * comb(total - marked, n_obs - x) for x in range(k_obs + 1)) / comb(total, n_obs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--observed", action="append", default=[], metavar="NAME=ZEROS/TICKS/WITH_LAST1S/EPISODES")
    args = ap.parse_args()

    reference = None
    print("Run | session | condition | episodes | fallback zeros / requests | per episode | "
          "episodes with one in the last 1 s | successes flagged")
    for run, label, work, method in SESSIONS:
        for cond in CONDITIONS:
            eps = episodes(args.repo, work, method, cond)
            if not eps:
                continue
            z = sum(e["zeros"] for e in eps)
            n = sum(e["ticks"] for e in eps)
            print(
                "%s | %s | %s | %d | %d / %d (%.2f %%) | %d-%d | %d | %d"
                % (run, label, cond, len(eps), z, n, 100.0 * z / n, min(e["zeros"] for e in eps),
                   max(e["zeros"] for e in eps), sum(1 for e in eps if e["zeros_last1s"] > 0),
                   sum(1 for e in eps if e["flagged"]))
            )
            if (run, work, method, cond) == REFERENCE:
                reference = eps
    if reference is None:
        print("no reference bag (%s %s)" % (REFERENCE[1], REFERENCE[3]))
        return 1

    zeros = np.array([e["zeros"] for e in reference], dtype=np.float64)
    ticks = np.array([e["ticks"] for e in reference], dtype=np.float64)
    k_ref = sum(1 for e in reference if e["zeros_last1s"] > 0)
    print()
    print("Reference: v4 noise, bc_v4: %d episodes, %d / %d fallback zeros (%.2f %%), %d with one in the last 1 s"
          % (len(reference), zeros.sum(), ticks.sum(), 100.0 * zeros.sum() / ticks.sum(), k_ref))
    print("  per episode (pose round: zeros / requests, in the last 1 s):")
    for e in reference:
        print("    %d %d: %d / %d, %d" % (e["pose_id"], e["round"], e["zeros"], e["ticks"], e["zeros_last1s"]))
    for spec in args.observed:
        name, numbers = spec.split("=", 1)
        z_obs, n_obs, k_obs, e_obs = (int(v) for v in numbers.split("/"))
        rng = np.random.default_rng(0)
        rates = np.empty(20000)
        for i in range(rates.size):
            pick = rng.choice(len(reference), e_obs, replace=False)
            rates[i] = zeros[pick].sum() / ticks[pick].sum()
        rate = z_obs / n_obs
        print(
            "%s: %d / %d (%.2f %%) over %d episodes; draws of %d reference episodes at or below that rate: "
            "%.4f (min %.2f %%, 5th percentile %.2f %%); with a last-1 s zero %d of %d against %d of %d, "
            "Fisher one-sided p = %.4f"
            % (name, z_obs, n_obs, 100.0 * rate, e_obs, e_obs, float(np.mean(rates <= rate)), 100.0 * rates.min(),
               100.0 * np.percentile(rates, 5), k_obs, e_obs, k_ref, len(reference),
               fisher_less(k_obs, e_obs, k_ref, len(reference)))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
