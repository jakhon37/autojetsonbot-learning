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


def _condition(folder, arms, schedule=True):
    """A condition folder with trials.csv, episodes_gt.csv and (optionally) schedule.csv.

    arms: {method: [success per pose 0..n-1]}, all in round 1.
    """
    folder.mkdir(parents=True)
    trials, episodes, slots = [], [], []
    for method, outcomes in arms.items():
        for pose, ok in enumerate(outcomes):
            trials.append([method, pose, 1, "success" if ok else "fail", 10.0, 0.27, 2.0, 0, ""])
            row = {c: "" for c in ses.EPISODE_COLUMNS}
            row.update(method=method, pose_id=pose, round=1, outcome="success" if ok else "fail", collided=0,
                       zone_reached=1)
            episodes.append([row[c] for c in ses.EPISODE_COLUMNS])
            slots.append([len(slots) + 1, 1, pose + 1, method, pose, "start %s %d 1" % (method, pose)])
    _write(folder / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, trials)
    _write(folder / "episodes_gt.csv", ses.EPISODE_COLUMNS, episodes)
    if schedule:
        _write(folder / "schedule.csv", ses.SCHEDULE_COLUMNS, slots)


def test_merge_pairs_a_new_arm_with_an_earlier_session(tmp_path):
    _condition(tmp_path / "v2" / "c", {"servo": [1, 1, 1, 1], "bc": [0, 0, 0, 1]})
    _condition(tmp_path / "v3" / "c", {"bc_v2": [1, 1, 0, 1]}, schedule=False)
    out = tmp_path / "res" / "c"
    assert ses.main(["merge", "--out", str(out), str(tmp_path / "v2" / "c"), str(tmp_path / "v3" / "c")]) == 0
    with open(out / "trials.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["method"] for r in rows] == ["servo"] * 4 + ["bc"] * 4 + ["bc_v2"] * 4
    with open(out / "schedule.csv", newline="") as f:
        schedule = list(csv.DictReader(f))
    # the earlier schedule as written, then the slots the session without a schedule ran
    assert len(schedule) == 12 and schedule[0]["trial"] == "1"
    assert schedule[-1] == {"trial": "", "round": "1", "block": "", "method": "bc_v2", "pose_id": "3",
                            "event": "start bc_v2 3 1"}
    assert len(ses.analyze_trials.read_schedule(out / "schedule.csv")) == 12
    with open(out / "episodes_gt.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 12
    table = tmp_path / "t.md"
    assert ses.main(["table", "--results", str(tmp_path / "res"), "--conditions", "c", "--vs", "bc",
                     "--out", str(table)]) == 0
    text = table.read_text()
    assert "| c | bc_v2 | 3/4 | 30–95% | 1 / 0 | 1 |" in text  # vs servo
    assert "**Paired with `bc`**" in text
    assert "| c | bc_v2 vs bc | 4 | 0 / 2 | 0.5 | +50.0 pp" in text
    assert "bc vs bc" not in text and "servo vs bc" not in text
    # each pair once: bc_v2 is the last method, so nothing is paired with it
    assert ses.main(["table", "--results", str(tmp_path / "res"), "--conditions", "c", "--vs", "bc", "--vs",
                     "bc_v2", "--out", str(table)]) == 0
    assert "Paired with `bc_v2`" not in table.read_text() and "bc vs bc_v2" not in table.read_text()


def test_merge_refuses_a_method_in_two_sessions(tmp_path, capsys):
    _condition(tmp_path / "a", {"servo": [1], "bc": [0]})
    _condition(tmp_path / "b", {"bc": [1]})
    assert ses.main(["merge", "--out", str(tmp_path / "o"), str(tmp_path / "a"), str(tmp_path / "b")]) == 2
    assert "method 'bc' has trials in both" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_table_without_vs_pairs_is_unchanged(tmp_path):
    _condition(tmp_path / "c", {"servo": [1, 1], "bc": [0, 1]})
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    assert ses.main(["table", "--results", str(tmp_path), "--conditions", "c", "--out", str(a)]) == 0
    assert ses.main(["table", "--results", str(tmp_path), "--conditions", "c", "--vs", "bc", "--out", str(b)]) == 0
    assert a.read_text() == b.read_text()  # only servo and bc: nothing else to pair with bc


# ---------------------------------------------------------------------------- scripts/sim_eval.sh


def _sim_eval(tmp_path, *args):
    import subprocess

    return subprocess.run(
        ["bash", str(REPO / "scripts" / "sim_eval.sh"), "--work", str(tmp_path / "work"), "--results",
         str(tmp_path / "res"), "--python", sys.executable, *args],
        capture_output=True, text=True, timeout=120,
    )


def _schedule(tmp_path, cond):
    with open(tmp_path / "work" / "eval_v2" / "jazzy" / cond / "schedule.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_sim_eval_methods_and_clean_ref_schedules(tmp_path):
    (tmp_path / "work" / "runs" / "bc").mkdir(parents=True)
    (tmp_path / "work" / "runs" / "bc" / "policy.onnx").write_bytes(b"stand-in")
    proc = _sim_eval(tmp_path, "--methods", "bc_v2", "--report-only", "clean_ref", "noise")
    assert proc.returncode == 0, proc.stderr
    clean, noise = _schedule(tmp_path, "clean_ref"), _schedule(tmp_path, "noise")
    assert {r["method"] for r in clean + noise} == {"bc_v2"}
    assert len(clean) == 10 and {r["round"] for r in clean} == {"1"}  # randomisation off: 1 round
    assert len(noise) == 30
    # the default arms reproduce the v2 schedule
    proc = _sim_eval(tmp_path, "--report-only", "noise")
    assert proc.returncode == 0, proc.stderr
    with open(REPO / "results" / "sim_eval_v2" / "noise" / "schedule.csv", newline="") as f:
        assert _schedule(tmp_path, "noise") == list(csv.DictReader(f))


def test_sim_eval_rejects_two_policy_arms_and_needs_a_model_only_for_one(tmp_path):
    proc = _sim_eval(tmp_path, "--methods", "servo,bc,bc_v2", "--report-only", "noise")
    assert proc.returncode == 1 and "one policy arm per session" in proc.stderr
    proc = _sim_eval(tmp_path, "--methods", "bc_v2", "--report-only", "noise")
    assert proc.returncode == 1 and "policy.onnx" in proc.stderr
    proc = _sim_eval(tmp_path, "--methods", "servo", "--report-only", "noise")
    assert proc.returncode == 0, proc.stderr
    proc = _sim_eval(tmp_path, "--pair-with", str(tmp_path / "missing"), "--methods", "servo", "--report-only")
    assert proc.returncode == 1 and "is not a directory" in proc.stderr
