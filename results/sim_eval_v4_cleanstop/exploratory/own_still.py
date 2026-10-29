"""EXPLORATORY (not in the PREREG): for each policy-arm episode, were the policy's own requests in
the last 1.0 s before the end all still by the v4 judge's request rule (|v| <= 0.005, |w| <= 0.05),
i.e. would the episode's ending hold without policy_runner's fallback zeros? Also, per judge status
tick in the last 1.0 s, whether the latest recorded request before the tick was a fallback zero
while the policy's own latest request was not still (a tick a fallback zero may have carried)."""
import csv, sys
from pathlib import Path
REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "scripts"))
import sim_eval_summary as ses  # noqa: E402

V_STILL, W_STILL = 0.005, 0.05
sources = {  # condition -> session bag folders
    "clean_ref": [REPO / "sim_work/v4/eval_v2/jazzy/clean_ref/bags"],
    "noise": [REPO / "sim_work/v4/eval_v2/jazzy/noise/bags"],
    "wide": [REPO / "sim_work/v4/eval_v2/jazzy/wide/bags"],
    "hfov62": [REPO / "sim_work/v4/eval_v2/jazzy/hfov62/bags", REPO / "sim_work/v4_62/eval_v2/jazzy/hfov62/bags"],
}
out = csv.writer(sys.stdout)
out.writerow(["condition", "method", "pose_id", "round", "outcome", "fallback_zeros_last1s", "own_n_last1s",
              "own_max_abs_v_last1s", "own_max_abs_w_last1s", "own_all_still_last1s", "status_ticks_last1s",
              "ticks_carried_by_fallback"])
for cond, folders in sources.items():
    for folder in folders:
        for bag in sorted(ses.analyze_trials.find_bags(folder)):
            for ep in ses.read_episodes(bag):
                if not ep.policy or ep.outcome not in ("success", "fail") or ep.t1 is None:
                    continue
                t1 = ep.t1
                last = [c for c in ep.cmd if c[0] >= t1 - 1.0]
                own = [c for c in last if not ep.is_fallback(c)]
                fz = sum(1 for c in last if ep.is_fallback(c))
                vmax = max((abs(c[1]) for c in own), default=float("nan"))
                wmax = max((abs(c[2]) for c in own if len(c) > 2), default=float("nan"))
                still = bool(own) and all(abs(c[1]) <= V_STILL and (len(c) < 3 or abs(c[2]) <= W_STILL) for c in own)
                ticks = [t for t, _ in ep.status if t1 - 1.0 <= t <= t1]
                carried = 0
                for t in ticks:
                    before = [c for c in ep.cmd if c[0] <= t]
                    if not before or not ep.is_fallback(before[-1]):
                        continue
                    own_before = [c for c in before if not ep.is_fallback(c)]
                    if own_before and not (abs(own_before[-1][1]) <= V_STILL and abs(own_before[-1][2]) <= W_STILL):
                        carried += 1
                out.writerow([cond, ep.method, ep.pose_id, ep.round, ep.outcome, fz, len(own), "%.4f" % vmax,
                              "%.4f" % wmax, int(still), len(ticks), carried])
