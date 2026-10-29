"""scripts/stop_labels.py: stop examples in demonstrations and datasets (v3 BC fix analysis)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stop_labels", REPO / "scripts" / "stop_labels.py")
sl = importlib.util.module_from_spec(_spec)
sys.modules["stop_labels"] = sl
_spec.loader.exec_module(sl)

ARMS = ["--methods", "teleop", "bc", "none", "--outcomes", "success", "fail", "none"]


def test_frame_classes():
    v = [0.0, 0.001, 0.1, 0.0, 0.1] + [0.05] * 20 + [0.03, 0.0] * 10
    cls = sl.frame_classes(v)
    assert cls[:2] == ["stop_start"] * 2
    assert cls[2] == "moving" and cls[3] == "stop_middle" and cls[4] == "moving"
    assert cls[-20:] == ["moving_end", "stop_end"] * 10  # the last 2 s, interleaved labels kept apart
    assert sl.frame_classes([0.0] * 25) == ["stop_start"] * 5 + ["stop_end"] * 20
    assert sl.frame_classes([0.1, 0.0]) == ["moving_end", "stop_end"]
    assert sl.frame_classes([]) == []


def test_ending_measures_the_final_stop_request():
    cmd = [(10.0, 0.05), (10.5, 0.0), (11.0, 0.03), (11.2, 0.0), (11.8, 0.002), (12.5, 0.0)]
    e = sl.ending(cmd, t1=12.6)
    assert e["max_v_last1s"] == pytest.approx(0.002) and e["min_v_last1s"] == 0.0
    assert e["stop_share_last1s"] == 1.0
    assert e["stop_for_s"] == pytest.approx(1.4)
    assert e["zero_for_s"] == pytest.approx(0.1), "0.002 m/s is a stop, but not exactly zero"
    e = sl.ending([(1.0, 0.01), (1.9, 0.015)], t1=2.0)  # creep the v2/v3 judge called still
    assert e["max_v_last1s"] == pytest.approx(0.015) and e["stop_for_s"] == 0.0 and e["zero_for_s"] == 0.0
    e = sl.ending([(1.0, 0.03), (1.2, 0.0), (2.0, -0.0), (2.4, 0.0)], t1=2.5)  # the latched servo
    assert e["zero_for_s"] == pytest.approx(1.3) and e["stop_for_s"] == pytest.approx(1.3)
    assert e["max_v_last1s"] == 0.0  # P1's exact zero: -0.0 counts, |v| is taken


def test_max_v_last1s_is_p1s_exact_zero():
    # Every request of the last 1.00 s exactly 0, though the run of zeros began 0.95 s before the
    # end: P1's measure (max_v_last1s) passes it, zero_for_s >= 1 s would not.
    zeros = [(9.05 + 0.05 * i, 0.0) for i in range(19)]  # 9.05 ... 9.95, the servo's 20 Hz
    e = sl.ending([(8.95, 0.02)] + zeros, t1=10.0)
    assert e["max_v_last1s"] == 0.0 and e["zero_for_s"] == pytest.approx(0.95)
    # the window includes its start: a request at exactly t1 - 1.00 s counts
    assert sl.ending([(9.0, 0.02)] + zeros, t1=10.0)["max_v_last1s"] == 0.02
    # a request that a 4-decimal CSV would print as 0.0000 is not zero
    e = sl.ending([(8.5, 0.0), (9.5, 4e-5), (9.9, 0.0)], t1=10.0)
    assert e["max_v_last1s"] == 4e-5 and "%.4f" % e["max_v_last1s"] == "0.0000"
    # no request in the window: nan, which is not 0
    e = sl.ending([(8.0, 0.0)], t1=10.0)
    assert e["max_v_last1s"] != e["max_v_last1s"] and e["zero_for_s"] == pytest.approx(2.0)


def test_bags_counts_p1_exact_zero_from_unrounded_requests(monkeypatch, tmp_path, capsys):
    import sim_eval_summary as ses  # what cmd_bags reads the bags with

    def episode(pose, cmd, outcome="success", status=()):
        ep = ses.Episode("servo_demo", pose, 1, t0=0.0, bag="b", outcome=outcome, reason="", t1=10.0)
        ep.cmd, ep.status = cmd, list(status)
        return ep

    zeros = [(8.0 + 0.05 * i, 0.0) for i in range(40)]  # 8.00 ... 9.95
    band = {"in_band": True, "front_range_m": 0.25}
    episodes = [
        episode(10, [(7.9, 0.02)] + zeros),  # exactly 0 for 2 s
        episode(11, [(8.95, 0.02)] + zeros[21:]),  # zeros from 9.05 s: the last 1.00 s are all 0
        episode(12, [(7.9, 0.02)] + zeros[:-2] + [(9.9, 4e-5), (9.95, 0.0)]),  # rounds to 0.0000
        episode(13, [(7.9, 0.02), (8.5, 0.0)]),  # silent for the last 1.5 s: no request to read
        episode(14, zeros, outcome="fail"),
        # Stop toggling in band: once 0.2 s after the start (skipped: the judge may still report the
        # last pose), once at 0.6 s, when a status of that very instant says in band. Then the
        # judge's last range, not its first.
        episode(15, [(0.1, 0.0), (0.2, 0.05), (0.5, 0.0), (0.6, 0.05), (7.9, 0.02)] + zeros,
                status=[(0.0, band), (0.25, {"in_band": False, "front_range_m": 0.5}), (0.6, band),
                        (9.9, {"in_band": True, "front_range_m": 0.2655})]),
        # Exactly v = 0 for exactly 1.000 s (zeros from 9.00 s), and a last status without a range.
        episode(16, [(8.95, 0.02)] + [(9.0 + 0.05 * i, 0.0) for i in range(20)],
                status=[(9.9, {"in_band": True, "front_range_m": None})]),
    ]
    monkeypatch.setattr(ses, "read_episodes", lambda path: episodes)
    out = tmp_path / "endings.csv"
    assert sl.main(["bags", str(tmp_path / "bag"), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rows = {r["pose_id"]: r for r in csv.DictReader(f)}
    assert [rows[p]["max_v_last1s"] for p in ("10", "11", "13", "14", "15", "16")] == ["0.0", "0.0", "nan", "0.0",
                                                                                      "0.0", "0.0"]
    assert float(rows["12"]["max_v_last1s"]) == 4e-5  # written in full, not as 0.0000
    # zero_for_s to 1 ms; the restarts and the judge's last range
    assert [rows[p]["zero_for_s"] for p in ("10", "11", "13", "16")] == ["2.000", "0.950", "1.500", "1.000"]
    assert rows["15"]["restarts_in_band"] == "1" and rows["10"]["restarts_in_band"] == "0"
    assert rows["15"]["final_range_m"] == "0.2655" and rows["16"]["final_range_m"] == "" == rows["10"]["final_range_m"]
    text = capsys.readouterr().out
    assert "successes whose largest requested |v| in the last 1.00 s was exactly 0 (max_v_last1s, unrounded): 4/6" \
        in text
    # zero_for_s stays descriptive: it passes 10, 13, 15 and 16 (exactly 1.000 s), not 11
    assert "successes that ended with a request of exactly v = 0 for >= 1 s: 4/6" in text
    assert "episodes whose request went from a stop back to moving while in band: 1/7 15/1:1" in text
    assert "final judge range 0.266-0.266 m (median 0.266)" in text


def test_restarts_in_band_counts_stop_toggling_near_the_box():
    status = [(0.0, {"in_band": False}), (5.0, {"in_band": True})]
    # a turn in place, then driving, far from the box (not counted); then the v2/v3 servo's
    # toggling across its stop range inside the band (counted twice)
    cmd = [(1.0, 0.0), (1.5, 0.1), (5.5, 0.019), (6.0, 0.0), (6.5, 0.019), (7.0, 0.0), (7.5, 0.0185), (8.0, 0.0)]
    assert sl.restarts_in_band(cmd, status) == 2
    assert sl.restarts_in_band(cmd, status, t_from=7.0) == 1
    assert sl.restarts_in_band([(5.5, 0.02), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0)], status) == 0  # stops once
    assert sl.restarts_in_band(cmd, []) == 0  # no judge status: never in band
    # a status of the very instant of the request counts; requests may carry w
    assert sl.restarts_in_band([(4.0, 0.0, 0.3), (5.0, 0.05, 0.0)], status) == 1


def test_bags_reports_every_episode(synthetic_bags, tmp_path, capsys):
    bag = synthetic_bags["jazzy_trials"]
    out = tmp_path / "endings.csv"
    assert sl.main(["bags", str(bag.path), "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["pose_id"], r["outcome"]) for r in rows] == [("3", "success"), ("4", "abort"), ("5", "fail")]
    for r in rows:
        assert float(r["zero_for_s"]) <= float(r["stop_for_s"]) + 0.006  # stop_for_s is rounded to 0.01 s
        assert r["restarts_in_band"] == "0" and r["final_range_m"] == ""  # this bag has no judge status
    # max_v_last1s is the bag's own value, unrounded: the largest |v| of the episode's last 1.00 s
    (start, _), (end, _) = bag.events[0], bag.events[1]  # "start teleop 3 1", "end success"
    expected = max(abs(v) for t, v, _ in bag.cmd if start <= t < end and t >= end - 10**9)
    assert float(rows[0]["max_v_last1s"]) == expected and len(rows[0]["max_v_last1s"]) > 6
    assert "exactly 0 (max_v_last1s, unrounded): 0/1" in capsys.readouterr().out


def test_dataset_counts_stop_frames_by_position(dataset_root, tmp_path):
    out = tmp_path / "stats.json"
    assert sl.main(["dataset", "--root", str(dataset_root), *ARMS, "--out", str(out)]) == 0
    stats = json.loads(out.read_text())["all"]
    assert stats["frames"] > 0 and stats["episodes"] > 0
    stop_classes = [stats[c] for c in ("stop_end", "stop_start", "stop_middle")]
    assert stats["stop_frames"] == sum(c["frames"] for c in stop_classes)
    assert stats["full_stop_frames"] == sum(c["full_stops"] for c in stop_classes)
    assert all(c["frames"] == c["full_stops"] + c["turns"] for c in stop_classes)
    assert stats["frames"] == stats["stop_frames"] + stats["moving_end"] + stats["moving"]
    assert stats["stop_fraction"] == pytest.approx(stats["stop_frames"] / stats["frames"], abs=1e-4)
