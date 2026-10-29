"""scripts/sim_eval_summary.py: per-episode ground-truth metrics and the summary table."""

import csv
import importlib.util
import math
import re
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
    # Ground truth at (1.2, 0.05, -10 deg): the lidar 0.064 m ahead of base_link, the median beam
    # of the judge's sector along the heading to the face plane x = 1.5; the camera 0.10 m ahead.
    yaw = math.radians(-10)
    lidar_x = 1.2 + 0.064 * math.cos(yaw)
    assert float(row["final_range_gt_m"]) == pytest.approx((1.5 - lidar_x) / math.cos(yaw), abs=1e-3)
    cam_x, cam_y = 1.2 + 0.10 * math.cos(yaw), 0.05 + 0.10 * math.sin(yaw)
    bearing = math.degrees(math.atan2(0.0 - cam_y, 1.5 - cam_x) - yaw)
    assert float(row["final_bearing_gt_deg"]) == pytest.approx(bearing, abs=0.01) and bearing > 0.0


def test_episode_row_never_in_zone_and_collision():
    ep = ses.Episode("servo", 0, 1, t0=0.0, bag="b", outcome="fail", reason="timeout", t1=30.0)
    ep.gt = [(5.0, _gt(1.0, 0.0, 0.0, collided=True))]
    ep.status = [(0.1 * i, _status(True, i % 5 != 0)) for i in range(300)]  # never 1 s in a row
    row = ep.row()
    assert row["collided"] == 1 and row["zone_reached"] == 0 and row["time_to_zone_s"] == ""
    assert row["cmd_v_end"] == "" and row["cmd_v_min_last1s"] == ""
    # Square to the face, the median of the sector's 19 beams (-9..9 deg) is a beam 5 deg off axis.
    assert float(row["final_range_gt_m"]) == pytest.approx((1.5 - 1.0 - 0.064) / math.cos(math.radians(5)), abs=1e-4)
    assert float(row["final_bearing_gt_deg"]) == pytest.approx(0.0, abs=1e-6)
    ep.gt = []  # no ground truth: nothing to report
    assert ep.row()["final_range_gt_m"] == "" and ep.row()["final_bearing_gt_deg"] == ""


def test_ground_truth_range_is_what_the_judge_reads_from_a_noise_free_scan():
    sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "jbl_nodes"))
    from jbl_nodes.judge_logic import front_range
    from jbl_nodes.sim_world import Pose2D, SimWorld

    for pose in (Pose2D(1.15, 0.0, 0.0), Pose2D(1.17, -0.08, math.radians(20)), Pose2D(0.4, 0.3, math.radians(-30))):
        sim = SimWorld()
        sim.reset(pose)
        lidar = sim.lidar
        expected = front_range(sim.scan(), lidar.angle_min, lidar.angle_increment, lidar.range_min, lidar.range_max)
        r, b = ses.gt_range_and_bearing(*pose)
        assert r == pytest.approx(expected, abs=1e-6) and b == pytest.approx(sim.marker_bearing_deg(), abs=1e-9)


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------- validity check: stale inputs at the end (v4)


def test_stale_flag_thresholds():
    # fewer than 5 camera frames, or (policy arms) fewer than 5 fresh inferences or any fallback zero,
    # in the last 1 s
    assert ses.stale_flag("bc_v4", 15, 10, 0) == 0
    assert ses.stale_flag("bc_v4", 5, 5, 0) == 0 and ses.stale_flag("bc_v4", 4, 10, 0) == 1
    assert ses.stale_flag("bc_v4", 15, 4, 0) == 1
    assert ses.stale_flag("bc_v4", 15, 10, 1) == 1  # one fallback zero is enough
    for servo in ("servo", "servo_demo"):  # servo_baseline's arms: only the frames count
        assert ses.stale_flag(servo, 15, None, None) == 0 and ses.stale_flag(servo, 15, 0, 3) == 0
        assert ses.stale_flag(servo, 4, None, None) == 1
    # a count that was not recorded: unknown, unless what was recorded already flags it
    assert ses.stale_flag("bc_v4", None, 10, 0) == "" and ses.stale_flag("bc_v4", 15, None, 0) == ""
    assert ses.stale_flag("bc_v4", 15, 10, None) == ""
    assert ses.stale_flag("bc_v4", None, 3, 0) == 1 and ses.stale_flag("bc_v4", None, None, 2) == 1
    assert ses.stale_flag("servo", None, None, None) == ""


def test_episode_row_counts_the_last_second_from_its_first_instant_and_leaves_out_fallback_zeros():
    ep = ses.Episode("bc_v4", 1, 1, t0=0.0, bag="b", outcome="fail", reason="timeout", t1=10.0)
    # the window [t1 - 1 s, t1] includes its first instant, as P1's max_v_last1s does
    ep.frames = [8.99, 9.0, 9.3, 9.6, 9.9]
    ep.inferences = [8.99, 9.0, 9.4, 9.7, 9.95]
    # creep of 0.012 m/s, and policy_runner's fallback zeros: exactly (0, 0), one at the window's start
    ep.cmd = [(8.99, 0.0, 0.0), (9.0, 0.0, 0.0), (9.1, 0.012, 0.01), (9.45, 0.0, 0.0), (9.5, 0.013, -0.01),
              (9.6, 0.0, 0.2), (9.9, 0.012)]  # (0, 0.2) turns in place: v = 0 alone is the policy's own
    row = ep.row()
    assert row["camera_frames_last1s"] == 4 and row["fresh_inferences_last1s"] == 4 and row["stale_last1s"] == 1
    assert row["fallback_zeros_last1s"] == 2
    assert row["cmd_v_min_last1s"] == "0.0000"  # every request, fallback zeros included
    assert row["own_v_min_last1s"] == "0.0000"  # the turn in place
    ep.cmd = [c for c in ep.cmd if c[2:] != (0.2,)]
    row = ep.row()
    assert row["cmd_v_min_last1s"] == "0.0000" and row["own_v_min_last1s"] == "0.0120"
    assert not ses.stop_commanded(row) and ses.stop_commanded(row, "cmd_v_min_last1s")
    ep.cmd = [(9.5, 0.0, 0.0)]  # nothing of its own in the last second
    row = ep.row()
    assert row["own_v_min_last1s"] == "" and row["cmd_v_min_last1s"] == "0.0000" and not ses.stop_commanded(row)
    ep.frames, ep.inferences = [9.0 + k / 15 for k in range(15)], [9.0 + k / 10 for k in range(10)]
    ep.cmd = [(9.0 + k / 10, 0.002, 0.0) for k in range(10)]
    assert ep.row()["stale_last1s"] == 0 and ep.row()["fallback_zeros_last1s"] == 0

    # servo_baseline's arms: no inferences, and its exact zeros are its own (the latched stop)
    servo = ses.Episode("servo_demo", 2, 1, t0=0.0, bag="b", outcome="success", t1=10.0)
    servo.frames = [9.0 + k / 15 for k in range(15)]
    servo.inferences = [9.0 + k / 10 for k in range(10)]  # another arm's runner: not the servo's
    servo.cmd = [(9.0 + 0.05 * k, 0.0, 0.0) for k in range(20)]
    row = servo.row()
    assert row["fresh_inferences_last1s"] == "" and row["fallback_zeros_last1s"] == "" and row["stale_last1s"] == 0
    assert row["own_v_min_last1s"] == row["cmd_v_min_last1s"] == "0.0000" and ses.stop_commanded(row)


def _stale_bag(tmp_path):
    """Six episodes. The camera stalls in the last second of the second; the runner's fresh
    inferences stop in the last second of the fourth (its frames still arrive, but too old); the
    fifth, a fail, creeps with one fallback zero (0, 0) in its last second, without frames there;
    the sixth succeeds with a fallback zero in its last second."""
    import make_synthetic_bags as msb

    spec = msb.BagSpec(
        "stale_end", "jazzy", duration_s=17.5, camera_hz=15.0, cmd_hz=10.0, latency_hz=10.0,
        camera_stalls=[(5.15, 6.5), (12.95, 14.3)], latency_gaps=[(11.15, 12.4)], zero_cmds=[13.55, 16.85],
        events=[
            (0.5, "start bc_v4 1 1"), (3.0, "end success"),
            (3.5, "start bc_v4 2 1"), (6.0, "end success"),
            (6.5, "start servo 3 1"), (9.0, "end fail timeout"),
            (9.5, "start bc_v4 4 1"), (12.0, "end success"),
            (12.5, "start bc_v4 5 1"), (14.0, "end fail timeout"),
            (14.5, "start bc_v4 6 1"), (17.0, "end success"),
        ],
        seed=7,
    )
    return msb.write_bag(spec, tmp_path / "bags")


def _expected_counts(bag, start_s, end_s):
    """Camera frames, /policy/latency messages and requests of exactly (0, 0) the episode holds in
    its last 1 s, and the smallest |v| of its other requests there (the oracle)."""
    t0, t1 = bag.origin_ns + round(start_s * 1e9), bag.origin_ns + round(end_s * 1e9)
    lo = max(t0, t1 - 10**9)
    cmd = [(v, w) for t, v, w in bag.cmd if lo <= t < t1]
    return (sum(1 for t, _ in bag.images if lo <= t < t1), sum(1 for t in bag.latency if lo <= t < t1),
            sum(1 for c in cmd if c == (0.0, 0.0)), min(abs(v) for v, w in cmd if (v, w) != (0.0, 0.0)))


def test_episodes_count_the_inputs_of_the_last_second_and_flag_stale_ones(tmp_path):
    bag = _stale_bag(tmp_path)
    out = tmp_path / "episodes_gt.csv"
    assert ses.main(["episodes", "--bags", str(bag.path), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rows = {r["pose_id"]: r for r in csv.DictReader(f)}

    def counts(pose):
        r = rows[pose]
        return tuple(int(r[c]) for c in ("camera_frames_last1s", "fresh_inferences_last1s", "fallback_zeros_last1s"))

    nominal = _expected_counts(bag, 0.5, 3.0)
    assert 14 <= nominal[0] <= 16 and 9 <= nominal[1] <= 11 and nominal[2] == 0  # 15 Hz camera, 10 Hz runner
    assert counts("1") == nominal[:3] and rows["1"]["stale_last1s"] == "0"
    assert rows["1"]["own_v_min_last1s"] == rows["1"]["cmd_v_min_last1s"] == "%.4f" % nominal[3]
    # the camera stalls 0.85 s before the end: the frames of 0.15 s are left, so it is flagged
    stalled = _expected_counts(bag, 3.5, 6.0)
    assert stalled[0] < 5 <= stalled[1]
    assert counts("2") == stalled[:3] and rows["2"]["stale_last1s"] == "1"
    # the servo's episode: frames only (the runner's messages in that window are not the servo's)
    assert int(rows["3"]["camera_frames_last1s"]) == _expected_counts(bag, 6.5, 9.0)[0]
    assert rows["3"]["fresh_inferences_last1s"] == rows["3"]["fallback_zeros_last1s"] == ""
    assert rows["3"]["stale_last1s"] == "0"
    # frames keep coming, but the runner made 1-2 fresh inferences in the last second
    starved = _expected_counts(bag, 9.5, 12.0)
    assert starved[0] >= 14 and starved[1] < 5
    assert counts("4") == starved[:3] and rows["4"]["stale_last1s"] == "1"
    # a creeping fail whose one fallback zero would pass for a commanded stop over every request
    creep = _expected_counts(bag, 12.5, 14.0)
    assert creep[0] == 0 and creep[1] >= 9 and creep[2] == 1 and creep[3] > 0.005
    assert counts("5") == creep[:3] and rows["5"]["stale_last1s"] == "1"
    assert rows["5"]["cmd_v_min_last1s"] == "0.0000" and rows["5"]["own_v_min_last1s"] == "%.4f" % creep[3]
    # nominal inputs and one fallback zero: flagged
    late = _expected_counts(bag, 14.5, 17.0)
    assert late[0] >= 14 and late[1] >= 9 and late[2] == 1
    assert counts("6") == late[:3] and rows["6"]["stale_last1s"] == "1"


def test_episodes_without_the_counted_topics_leave_the_check_empty(synthetic_bags, tmp_path):
    # jazzy_trials records the camera but not /policy/latency, so a method other than the servo's
    # (here teleop and bc) has enough frames but cannot be checked
    out = tmp_path / "episodes_gt.csv"
    assert ses.main(["episodes", "--bags", str(synthetic_bags["jazzy_trials"].path), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rows = {r["pose_id"]: r for r in csv.DictReader(f)}
    assert int(rows["3"]["camera_frames_last1s"]) >= 14 and rows["3"]["fresh_inferences_last1s"] == ""
    assert rows["3"]["fallback_zeros_last1s"] == "0"  # /cmd_vel_in is always read
    assert rows["3"]["stale_last1s"] == "" and rows["5"]["method"] == "bc" and rows["5"]["stale_last1s"] == ""
    # raw frames only: no /camera/image_raw/compressed, so not even the servo can be checked
    import make_synthetic_bags as msb

    bag = msb.write_bag(msb.BagSpec("raw_only", "jazzy", duration_s=3.0, image="bgr8", camera_hz=2.0, seed=9,
                                    events=[(0.5, "start servo 1 1"), (2.5, "end success")]), tmp_path / "raw")
    assert ses.main(["episodes", "--bags", str(bag.path), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["camera_frames_last1s"] == "" and row["stale_last1s"] == ""


def test_table_reports_the_validity_check_as_a_sensitivity_line(tmp_path):
    bag = _stale_bag(tmp_path)
    cond = tmp_path / "res" / "c"
    cond.mkdir(parents=True)
    assert ses.main(["episodes", "--bags", str(bag.path), "--out", str(cond / "episodes_gt.csv")]) == 0
    trials = [
        ["bc_v4", 1, 1, "success", 2.5, 0.27, 1.0, 0, ""],
        ["bc_v4", 2, 1, "success", 2.5, 0.27, 1.0, 0, ""],
        ["servo", 3, 1, "fail", 2.5, 0.40, 1.0, 0, "timeout"],
        ["bc_v4", 4, 1, "success", 2.5, 0.27, 1.0, 0, ""],
        ["bc_v4", 5, 1, "fail", 1.5, 0.30, 1.0, 0, "timeout"],
        ["bc_v4", 6, 1, "success", 2.5, 0.27, 1.0, 0, ""],
    ]
    _write(cond / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, trials)
    out = tmp_path / "t.md"
    assert ses.main(["table", "--results", str(tmp_path / "res"), "--conditions", "c", "--out", str(out)]) == 0
    text = out.read_text()
    assert "**Validity check: stale inputs at the end**" in text
    lines = {line.split(" | ")[1]: line for line in text.splitlines() if line.startswith("| c | ")
             and line.count("|") == 8}
    # primary results stay as scored (4/5); the check counts the three flagged successes as failures
    assert "| c | bc_v4 | 4/5 |" in text
    assert lines["bc_v4"].endswith("| 2/5 | 3 | 1/5 |")  # trials with a fallback zero: the 5th and the 6th
    assert lines["servo"].endswith("| — | — | 0 | 0/1 |")  # no inferences or fallback zeros for the servo
    # the medians and minimums are over every scored trial, failures included (the 5th had no frames)
    frames = [_expected_counts(bag, a, b)[0] for a, b in ((0.5, 3.0), (3.5, 6.0), (9.5, 12.0), (12.5, 14.0),
                                                          (14.5, 17.0))]
    assert min(frames) == 0
    assert "| %g (min 0) |" % float(sorted(frames)[2]) in lines["bc_v4"]


def test_table_validity_check_without_the_columns_is_not_checked(tmp_path):
    # an episodes_gt.csv written before the check existed (v2/v3): the v3 header, without the new
    # columns. The successes are listed as not checked, and so is one without a ground-truth row.
    v4_only = ("final_range_gt_m", "final_bearing_gt_deg", "own_v_min_last1s", "camera_frames_last1s",
               "fresh_inferences_last1s", "fallback_zeros_last1s", "stale_last1s")
    header = [c for c in ses.EPISODE_COLUMNS if c not in v4_only]
    folder = tmp_path / "c"
    folder.mkdir()
    trials, episodes = [], []
    for method, pose, ok in (("servo", 0, 1), ("servo", 1, 1), ("bc", 0, 0), ("bc", 1, 1), ("bc", 2, 1)):
        trials.append([method, pose, 1, "success" if ok else "fail", 10.0, 0.27, 2.0, 0, ""])
        if (method, pose) != ("bc", 2):  # this success has no ground-truth row
            row = dict.fromkeys(header, "")
            row.update(method=method, pose_id=pose, round=1, outcome="success" if ok else "fail", collided=0,
                       zone_reached=1, cmd_v_min_last1s="0.0000" if method == "servo" else "0.0170")
            episodes.append([row[c] for c in header])
    _write(folder / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, trials)
    _write(folder / "episodes_gt.csv", header, episodes)
    out = tmp_path / "t.md"
    assert ses.main(["table", "--results", str(tmp_path), "--conditions", "c", "--out", str(out)]) == 0
    text = out.read_text()
    assert "| c | servo | — | — | — | 0 (2 not checked) | 2/2 |" in text
    assert "| c | bc | — | — | — | 0 (2 not checked) | 2/3 |" in text
    # stop commanded falls back to every request (cmd_v_min_last1s), the only column v2/v3 have
    assert "| 0.000 (0.000–0.000) | 2/2 | 0/2 |" in text and "| 0.017 (0.017–0.017) | 0/2 | 0/2 |" in text


# ------------------------------------------------------- stops: P3 of the v4 pre-registration


def _gt_row(method, pose, outcome, own_min, gt_range, every_min=None):
    row = {c: "" for c in ses.EPISODE_COLUMNS}
    row.update(method=method, pose_id=pose, round=1, outcome=outcome, collided=0, zone_reached=1,
               own_v_min_last1s=own_min, cmd_v_min_last1s=own_min if every_min is None else every_min,
               final_range_gt_m=gt_range)
    return [row[c] for c in ses.EPISODE_COLUMNS]


def test_stops_pools_conditions_like_the_tables_stop_commanded(tmp_path, capsys):
    arms = {
        # the third creeps at 0.012 m/s with one fallback zero: a stop only over every request
        "clean_ref": [("0.0000", "0.2700", "success"), ("0.0000", "0.2800", "fail"),
                      ("0.0120", "0.3100", "fail", "0.0000"), ("0.0049", "0.2600", "success")],
        # 0.005 is not below 0.005; no request at all is no stop; a stop without ground truth counts
        "noise": [("0.0050", "0.2500", "success"), ("", "0.2400", "fail"), ("0.0000", "", "success")],
    }
    for cond, slots in arms.items():
        folder = tmp_path / cond
        folder.mkdir()
        trials, episodes = [["bc_v4", 0, 1, "abort", 3.0, "", "", 0, "start not delivered"]], []
        episodes.append(_gt_row("bc_v4", 0, "abort", "0.0000", "0.1000"))  # never scored
        for pose, (own_min, gt_range, outcome, *every) in enumerate(slots):
            trials.append(["bc_v4", pose, 1, outcome, 10.0, 0.27, 1.0, 0, ""])
            trials.append(["servo", pose, 1, "success", 10.0, 0.27, 1.0, 0, ""])
            episodes.append(_gt_row("bc_v4", pose, outcome, own_min, gt_range, *every))
            episodes.append(_gt_row("servo", pose, "success", "0.0000", "0.3500"))
        _write(folder / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, trials)
        _write(folder / "episodes_gt.csv", ses.EPISODE_COLUMNS, episodes)
    out = tmp_path / "p3.txt"
    stops = ["stops", "--results", str(tmp_path), "--conditions", "clean_ref", "noise", "--method", "bc_v4",
             "--out", str(out)]
    assert ses.main(stops) == 0
    assert out.read_text().splitlines() == [
        "bc_v4, clean_ref + noise: stop commanded in 4 of 7 scored trials (57.1 %)",
        "ground-truth final lidar range of those 4 trials: median 0.2700 m (IQR 0.2650-0.2750, range "
        "0.2600-0.2800; 3 with ground truth)",
        "sensitivity, every request counted (cmd_v_min_last1s, policy_runner's fallback zeros included): stop "
        "commanded in 5 of 7 scored trials (71.4 %); ground-truth final lidar range of those: median 0.2750 m "
        "(IQR 0.2675-0.2875, range 0.2600-0.3100; 4 with ground truth)",
    ]
    # the same rows as the secondary table's "Stop commanded": 3/4 and 1/3; the final-errors table
    # gives the ground-truth range to 4 decimals (P5)
    table = tmp_path / "t.md"
    assert ses.main(["table", "--results", str(tmp_path), "--conditions", "clean_ref", "noise", "--out", str(table)]) == 0
    text = table.read_text()
    assert "| 3/4 | 0/4 |" in text and "| 1/3 | 0/3 |" in text  # stop commanded, collisions
    # the smallest request of the last 1 s, by the same measure: own requests (0, 0, 0.0049, 0.012)
    assert "| 0.002 (0.000–0.007) | 3/4 | 0/4 |" in text
    assert "| 0.2750 (0.2675–0.2875) |" in text  # clean_ref, bc_v4: 0.26, 0.27, 0.28 and 0.31
    assert ses.main(["stops", "--results", str(tmp_path), "--conditions", "noise", "--method", "nobody"]) == 0
    assert "stop commanded in 0 of 0 scored trials (n/a)" in capsys.readouterr().out
    assert ses.main(["stops", "--results", str(tmp_path), "--conditions", "wide", "--method", "bc_v4"]) == 2

    # n counts every scored trial; one without a ground-truth row makes the file incomplete (exit 2)
    with open(tmp_path / "noise" / "trials.csv", "a", newline="") as f:
        csv.writer(f).writerow(["bc_v4", 7, 1, "success", 10.0, 0.27, 1.0, 0, ""])
    assert ses.main(stops) == 2
    lines = out.read_text().splitlines()
    assert lines[0] == "bc_v4, clean_ref + noise: stop commanded in 4 of 8 scored trials (50.0 %)"
    assert lines[-1] == ("INCOMPLETE: 1 of the 8 scored trials have no ground-truth row in episodes_gt.csv and are "
                         "not counted as stops above: noise pose 7 round 1")
    (tmp_path / "noise" / "episodes_gt.csv").unlink()  # a condition without ground truth at all
    assert ses.main(stops) == 2
    assert "INCOMPLETE: 4 of the 8 scored trials" in out.read_text()


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
    assert "| c | bc | 0/3 | 0–56% | 3 / 0 | 0.25 | — | timeout 3 (+1 aborted, not scored) |" in text
    assert "| c | bc | 2/3 | 21–94% | 1 / 0 | 1 |" in text  # zone reached, paired with servo's 3/3
    # final errors: judge range error and bearing, GT lateral, heading, bumper; no GT range or bearing here
    errors = "| 2.0 (2.0–2.0) | 3.0 (3.0–3.0) | 10.0 (10.0–10.0) | 5.0 (5.0–5.0) | 20.0 (20.0–20.0) | — | — |"
    assert "| c | servo " + errors in text
    assert "| missing |" not in text  # a condition without trials.csv is left out


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


# ------------------------------------------- scripts/sim_e2e.sh and sim_eval.sh with a stand-in docker


def _fake_docker(tmp_path, fail_inner_run=False):
    """A docker stand-in on PATH that logs each call; returns (env, log path)."""
    import os

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$FAKE_DOCKER_LOG"\n'
        'case $1 in run) echo fake_container ;; image) exit 1 ;; esac\n'
        + ('case $* in *"--inner run") exit 3 ;; esac\n' if fail_inner_run else "")
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]), FAKE_DOCKER_LOG=str(log))
    env.pop("SIM_CPUSET", None)
    return env, log


def _run(script, env, *args):
    import subprocess

    return subprocess.run(["bash", str(REPO / "scripts" / script), *args], env=env, capture_output=True, text=True,
                          timeout=120)


def test_sim_e2e_records_through_the_given_lens_on_the_given_cores(tmp_path):
    import yaml

    env, log = _fake_docker(tmp_path)
    env["SIM_CPUSET"] = "0-2"
    work = tmp_path / "work"
    (work / "runs" / "bc").mkdir(parents=True)
    (work / "runs" / "bc" / "policy.onnx").write_bytes(b"")  # the bc stage only checks it exists
    stages = ("servo", "demos", "bc")
    proc = _run("sim_e2e.sh", env, "--work", str(work), "--hfov", "1.0856", "--demo-rounds", "1", *stages)
    assert proc.returncode == 0, proc.stderr
    calls = log.read_text().splitlines()
    runs = [c for c in calls if c.startswith("run ")]
    assert len(runs) == 1 and "--cpuset-cpus 0-2" in runs[0]
    servo, demos, bc = [c for c in calls if c.endswith("--inner trials")]
    assert "LAUNCH_ARGS=hfov_rad:=1.0856 extra_params_file:=/work/params/lens.yaml " in servo
    assert "LAUNCH_ARGS=hfov_rad:=1.0856 start_poses_file:=" in demos
    assert "seed:=1 extra_params_file:=/work/params/demo_r1.yaml " in demos
    assert "LAUNCH_ARGS=hfov_rad:=1.0856 model_path:=" in bc and " extra_params_file:=/work/params/lens.yaml " in bc
    lens = yaml.safe_load((work / "params" / "lens.yaml").read_text())
    assert lens == {"fake_robot": {"ros__parameters": {"camera_hfov_rad": 1.0856}}}
    demo = yaml.safe_load((work / "params" / "demo_r1.yaml").read_text())
    assert demo["fake_robot"]["ros__parameters"] == {"initial_pose_id": 10, "camera_hfov_rad": 1.0856}
    assert demo["servo_baseline"]["ros__parameters"]["k_v"] == 1.0
    prov = (work / "logs" / "jazzy" / "demos.provenance").read_text()
    assert "camera hfov_rad: 1.0856" in prov and "SIM_CPUSET: 0-2" in prov
    assert "loadavg at start: " in prov and "launch demos_r1" in prov and "loadavg at end: " in prov
    assert re.search(r"^dirty: \d+ paths outside results/, \d+ tracked under results/$", prov, re.M)
    # the train stage runs on the host: no lens and no pinning, whatever SIM_CPUSET says (here it
    # stops at once, for want of demonstration bags, after writing its provenance)
    proc = _run("sim_e2e.sh", env, "--work", str(work), "train")
    assert proc.returncode == 1 and "no servo/demo bags" in proc.stderr
    prov = (work / "logs" / "jazzy" / "train.provenance").read_text()
    assert "SIM_CPUSET: n/a (a host stage, not pinned)" in prov and "camera hfov_rad" not in prov

    # the default: the simulator's own lens, an unpinned container
    env.pop("SIM_CPUSET")
    log.unlink()
    assert _run("sim_e2e.sh", env, "--work", str(work), "servo").returncode == 0
    calls = log.read_text().splitlines()
    assert "--cpuset-cpus" not in calls[0]
    assert "LAUNCH_ARGS=hfov_rad:=1.396 extra_params_file:=/work/params/lens.yaml " in calls[-2]
    # without a decimal point the parameter files would carry an int, which Humble+ reject
    proc = _run("sim_e2e.sh", env, "--work", str(work), "--hfov", "1", "servo")
    assert proc.returncode == 1 and "--hfov" in proc.stderr


def test_sim_eval_pins_the_container_and_logs_each_segment(tmp_path):
    env, log = _fake_docker(tmp_path)
    env["SIM_CPUSET"] = "0-2"
    work = tmp_path / "work"
    args = ("--work", str(work), "--results", str(tmp_path / "res"), "--python", sys.executable)
    proc = _run("sim_eval.sh", env, *args, "--methods", "servo", "noise")
    assert proc.returncode == 0, proc.stderr
    calls = log.read_text().splitlines()
    assert "--cpuset-cpus 0-2" in [c for c in calls if c.startswith("run ")][0]
    (inner,) = [c for c in calls if c.endswith("--inner run")]
    assert "PARAMS_LOG=/work/eval_v2/jazzy/noise/logs/seg_" in inner
    segments = (work / "eval_v2" / "jazzy" / "noise" / "segments.txt").read_text()
    assert "SIM_CPUSET: 0-2" in segments and "loadavg at start: " in segments and "loadavg at end: " in segments
    assert re.search(r"^dirty: \d+ paths outside results/, \d+ tracked under results/$", segments, re.M)

    # a failed run is recorded in the segment, and stops the session
    env, log = _fake_docker(tmp_path / "again", fail_inner_run=True)
    proc = _run("sim_eval.sh", env, *args, "--methods", "servo", "wide")
    assert proc.returncode == 1 and "failed with exit status 3" in proc.stderr
    assert "--cpuset-cpus" not in log.read_text()
    segments = (work / "eval_v2" / "jazzy" / "wide" / "segments.txt").read_text()
    assert "SIM_CPUSET: (not set" in segments and "segment failed: exit status 3" in segments


def test_sim_eval_exits_nonzero_when_a_report_fails(tmp_path):
    import make_synthetic_bags as msb

    env, _ = _fake_docker(tmp_path)
    work, res = tmp_path / "work", tmp_path / "res"
    cond = work / "eval_v2" / "jazzy" / "noise"
    msb.write_bag(msb.BagSpec("seg_1", "jazzy", duration_s=3.0, seed=3,
                              events=[(0.5, "start servo 0 1"), (2.5, "end success")]), cond / "bags")
    _write(cond / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, [["servo", 0, 1, "success", 2.0, 0.27, 1.0, 0, ""]])
    args = ("--work", str(work), "--results", str(res), "--methods", "servo", "--report-only")
    proc = _run("sim_eval.sh", env, *args, "--python", sys.executable, "noise")
    assert proc.returncode == 0, proc.stderr
    assert all((res / "noise" / f).is_file() for f in ("trials.csv", "episodes_gt.csv", "report.md", "provenance.txt"))

    def fails(*extra, python=sys.executable):
        proc = _run("sim_eval.sh", env, *args, "--python", str(python), *extra, "noise")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "noise: report failed" in proc.stdout
        assert "report failed: noise; the trials are recorded, rerun the same command with --report-only" in proc.stderr
        return proc

    # an earlier session whose trials.csv has other columns: merge refuses to pair it
    earlier = tmp_path / "earlier" / "noise"
    earlier.mkdir(parents=True)
    _write(earlier / "trials.csv", list(ses.analyze_trials.TRIAL_COLUMNS) + ["extra"],
           [["bc", 0, 1, "fail", 30.0, "", "", 0, "", "x"]])
    assert "has columns" in fails("--pair-with", str(tmp_path / "earlier")).stderr
    # an earlier session without episodes_gt.csv: the merged folder has no ground truth
    _write(earlier / "trials.csv", ses.analyze_trials.TRIAL_COLUMNS, [["bc", 0, 1, "fail", 30.0, "", "", 0, ""]])
    assert "no episodes_gt.csv in" in fails("--pair-with", str(tmp_path / "earlier")).stdout
    # only the ground truth from the bags fails (the earlier files are still there): still a failure
    python = tmp_path / "python"
    python.write_text('#!/usr/bin/env bash\ncase "$2" in episodes) echo "episodes: simulated failure" >&2; '
                      'exit 2 ;; esac\nexec "%s" "$@"\n' % sys.executable)
    python.chmod(0o755)
    assert "episodes: simulated failure" in fails(python=python).stderr
