"""scripts/sim_eval_summary.py: per-episode ground-truth metrics and the summary table."""

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sim_eval_summary", REPO / "scripts" / "sim_eval_summary.py")
ses = importlib.util.module_from_spec(_spec)
sys.modules["sim_eval_summary"] = ses
_spec.loader.exec_module(ses)


def _status(in_band, aligned):
    return {"in_band": in_band, "aligned": aligned}


def _gt(x, y, yaw, collided=False, bumper=0.2):
    return {"x": x, "y": y, "yaw": yaw, "collided": collided, "bumper_to_box_m": bumper}


def test_episode_row_zone_final_pose_and_commands():
    ep = ses.Episode("bc", 3, 2, t0=100.0, bag="b", outcome="fail", reason="timeout", t1=130.0)
    # A collided flag left over from the previous episode inside the teleport window is ignored.
    ep.gt = [(100.1, _gt(0.0, 0.0, 0.0, collided=True)), (129.9, _gt(1.2, 0.05, math.radians(-10)))]
    # In zone 110.0-110.9 (too short), then from 120.0 on: reached at 121.0 -> 21.0 s.
    ep.status = [(110.0 + 0.1 * i, _status(True, True)) for i in range(10)]
    ep.status += [(111.0, _status(True, False))]
    ep.status += [(120.0 + 0.1 * i, _status(True, True)) for i in range(100)]
    ep.cmd = [(127.0, 0.05), (128.5, 0.03), (129.2, 0.02), (129.8, 0.025)]
    row = ep.row()
    assert row["collided"] == 0
    assert row["zone_reached"] == 1
    assert float(row["time_to_zone_s"]) == pytest.approx(21.0, abs=0.01)
    assert row["end_in_zone"] == 1
    assert float(row["final_dist_m"]) == pytest.approx(0.3)
    assert float(row["final_lateral_m"]) == pytest.approx(0.05)
    assert float(row["final_heading_deg"]) == pytest.approx(-10.0)
    assert float(row["cmd_v_end"]) == pytest.approx(0.025)  # median of 0.03, 0.02, 0.025
    assert float(row["cmd_v_min_last1s"]) == pytest.approx(0.02)
    assert float(row["duration_s"]) == pytest.approx(30.0)


def test_episode_row_never_in_zone_and_collision():
    ep = ses.Episode("servo", 0, 1, t0=0.0, bag="b", outcome="fail", reason="timeout", t1=30.0)
    ep.gt = [(5.0, _gt(1.0, 0.0, 0.0, collided=True))]
    ep.status = [(0.1 * i, _status(True, i % 5 != 0)) for i in range(300)]  # never 1 s in a row
    row = ep.row()
    assert row["collided"] == 1 and row["zone_reached"] == 0 and row["time_to_zone_s"] == ""
    assert row["cmd_v_end"] == "" and row["cmd_v_min_last1s"] == ""


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_table_scores_like_analyze_trials(tmp_path):
    cond = tmp_path / "c"
    cond.mkdir()
    trials = []
    episodes = []
    for pose in range(3):
        trials.append(["servo", pose, 1, "success", 10.0, 0.27, 3.0, 0, ""])
        trials.append(["bc", pose, 1, "fail", 30.0, 0.23, 1.0, 0, "timeout"])
        for m, outcome, zone in (("servo", "success", 1), ("bc", "fail", int(pose != 2))):
            row = {c: "" for c in ses.EPISODE_COLUMNS}
            row.update(
                method=m,
                pose_id=pose,
                round=1,
                outcome=outcome,
                collided=0,
                zone_reached=zone,
                final_lateral_m="0.1",
                final_heading_deg="-5",
                final_bumper_m="0.2",
            )
            episodes.append([row[c] for c in ses.EPISODE_COLUMNS])
    # An aborted bc attempt first: never scored, and its ground-truth row is not used.
    trials.insert(0, ["bc", 0, 1, "abort", 3.0, "", "", 0, "start not delivered"])
    episodes.insert(0, ["bc", 0, 1, "abort", "x"] + [""] * (len(ses.EPISODE_COLUMNS) - 5))
    _write(cond / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, trials)
    _write(cond / "episodes_gt.csv", ses.EPISODE_COLUMNS, episodes)
    out = tmp_path / "t.md"
    assert (
        ses.main(["table", "--results", str(tmp_path), "--conditions", "c", "missing", "--out", str(out)])
        == 0
    )
    text = out.read_text()
    assert "| c | servo | 3/3 | 44–100% | — | — | 10.0 (10.0–10.0) | none |" in text
    assert "| c | bc | 0/3 | 0–56% | 3 / 0 | 0.25 | — | timeout 3 (+1 aborted, re-run) |" in text
    assert "| c | bc | 2/3 | 21–94% | 1 / 0 | 1 |" in text  # zone reached, paired with servo's 3/3
    assert "missing" not in text
