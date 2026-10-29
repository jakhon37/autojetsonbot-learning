import sys
from pathlib import Path
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "scripts"))
import sim_eval_summary as ses
for ep in ses.read_episodes(REPO / "sim_work/v4_62/bags/jazzy/demos_r2"):
    if ep.pose_id != 17:
        continue
    i, in_band, was_stop, rng = 0, False, False, None
    for t, v, *w in ep.cmd:
        while i < len(ep.status) and ep.status[i][0] <= t:
            in_band = bool(ep.status[i][1].get("in_band")); rng = ep.status[i][1].get("front_range_m"); i += 1
        stop = abs(v) < 0.005
        if was_stop and not stop and in_band and t >= ep.t0 + 0.3:
            print("restart in band at %.2f s after the start (end at %.2f s): v %.4f w %.3f, judge range %s" % (t - ep.t0, ep.t1 - ep.t0, v, w[0] if w else float('nan'), rng))
            prev = [c for c in ep.cmd if t - 1.0 <= c[0] < t]
            print("  preceding stop requests:", [("%.2f" % (c[0] - ep.t0), round(c[1], 4), round(c[2], 3)) for c in prev if abs(c[1]) < 0.005][-5:])
        was_stop = stop
