import sys
from pathlib import Path
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "scripts"))
import sim_eval_summary as ses
for bag in sorted(ses.analyze_trials.find_bags(REPO / "sim_work/v4/eval_v2/jazzy/noise/bags")):
    for ep in ses.read_episodes(bag):
        if ep.method == "bc_v4" and ep.pose_id == 1 and ep.round == 3:
            last = [c for c in ep.cmd if c[0] >= ep.t1 - 1.0]
            print(ep.outcome, [(round(c[0] - ep.t1, 3), repr(c[1]), round(c[2], 4)) for c in last if abs(c[1]) > 0.0045])
