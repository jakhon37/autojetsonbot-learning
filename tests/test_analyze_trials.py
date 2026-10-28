"""tools/analyze_trials.py and tools/make_schedule.py on synthetic trials with hand-checked numbers.

The synthetic session (write_trials) is 3 methods x 10 poses x 3 rounds with these rules:

    servo  succeeds unless pose 5 (fail timeout) or pose 6 (fail stop_obstacle)      -> 24/30
    bc     succeeds only at poses 0, 1, 2 (other fails: timeout)                      ->  9/30
    act    succeeds unless pose 5 (fail with 1 intervention); exceptions:
             pose 0 round 1: fail "wheel slipped" first, then a duplicate success row (ignored)
             pose 8 round 3: logged success but 1 intervention (scored as a fail)
             pose 3 round 2: never run (unpaired)
             pose 4 round 1: an abort row before its scored success                   -> 24/29
    plus one malformed row (pose_id "x").

Successful durations are 10 + pose (servo), 12 + pose (bc), 8 + pose (act) seconds.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import analyze_trials as at
import make_schedule as ms
import stats

REPO = Path(__file__).resolve().parents[1]
START_POSES = REPO / "ros2_ws" / "src" / "jbl_nodes" / "config" / "start_poses.yaml"
COLUMNS = list(at.TRIAL_COLUMNS)


def _row(method, pose, rnd, outcome, duration, rng=None, bearing=None, interventions=0, notes=""):
    return {
        "method": method,
        "pose_id": pose,
        "round": rnd,
        "outcome": outcome,
        "duration_s": "" if duration is None else "%.3f" % duration,
        "final_range_m": "" if rng is None else "%.4f" % rng,
        "final_bearing_deg": "" if bearing is None else "%.2f" % bearing,
        "interventions": interventions,
        "notes": notes,
    }


def _ok(method, pose, rnd, base):
    return _row(method, pose, rnd, "success", base + pose, 0.25 + 0.01 * (pose % 3), 1.0)


def _fail(method, pose, rnd, notes, interventions=0):
    return _row(method, pose, rnd, "fail", 30.0, 0.60, -20.0, interventions, notes)


def write_trials(path: Path) -> Path:
    rows = []
    for rnd in (1, 2, 3):
        for pose in range(10):
            if pose == 5:
                rows.append(_fail("servo", pose, rnd, "timeout"))
            elif pose == 6:
                rows.append(_fail("servo", pose, rnd, "stop_obstacle; no dock_judge status"))
            else:
                rows.append(_ok("servo", pose, rnd, 10))
            rows.append(_ok("bc", pose, rnd, 12) if pose in (0, 1, 2) else _fail("bc", pose, rnd, "timeout"))
            if (pose, rnd) == (3, 2):
                continue
            if (pose, rnd) == (4, 1):
                rows.append(_row("act", pose, rnd, "abort", 4.0, notes="no end before next start"))
            if pose == 5:
                rows.append(_fail("act", pose, rnd, "intervention about to hit the wall", interventions=1))
            elif (pose, rnd) == (0, 1):
                rows.append(_fail("act", pose, rnd, "wheel slipped"))
            elif (pose, rnd) == (8, 3):
                row = _ok("act", pose, rnd, 8)
                row["interventions"] = 1
                rows.append(row)
            else:
                rows.append(_ok("act", pose, rnd, 8))
    rows.append(_ok("act", 0, 1, 8))  # re-run of a finished trial: must be ignored
    rows.append(dict(_ok("bc", 1, 1, 12), pose_id="x"))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def trials_csv(tmp_path):
    return write_trials(tmp_path / "trials.csv")


@pytest.fixture()
def analysis(trials_csv):
    return at.analyze(trials_csv, n_boot=2000, seed=0)


def _summary(analysis, method):
    return next(s for s in analysis.summaries if s.method == method)


def _comparison(analysis, method):
    return next(c for c in analysis.comparisons if c.method == method)


# --------------------------------------------------------------------------------------------
# Scoring and statistics


def test_success_counts_and_wilson(analysis):
    assert analysis.order == ["servo", "bc", "act"]
    expected = {"servo": (24, 30), "bc": (9, 30), "act": (24, 29)}
    for method, (k, n) in expected.items():
        s = _summary(analysis, method)
        assert (s.k, s.n) == (k, n)
        assert s.ci == stats.wilson_interval(k, n)


def test_outcome_breakdown(analysis):
    servo, bc, act = (_summary(analysis, m) for m in ("servo", "bc", "act"))
    assert dict(servo.categories) == {"success": 24, "timeout": 3, "stop_obstacle": 3, "abort": 0}
    assert dict(bc.categories) == {"success": 9, "timeout": 21, "abort": 0}
    assert dict(act.categories) == {"success": 24, "intervention": 4, "other": 1, "abort": 1}
    assert act.interventions == 4


def test_mcnemar_pairs_and_missing_slots(analysis):
    bc = _comparison(analysis, "bc")
    t = bc.table
    assert (t.n_pairs, t.both_success, t.b, t.c, t.both_fail) == (30, 9, 15, 0, 6)
    assert bc.test.p_exact == pytest.approx(2 / 2**15, rel=1e-12)
    assert bc.test.p_mid == pytest.approx(1 / 2**15, rel=1e-12)
    assert bc.diff.estimate == pytest.approx(-15 / 30)
    assert bc.diff.low <= -15 / 30 <= bc.diff.high < 0
    assert bc.only_baseline == [] and bc.only_method == []

    act = _comparison(analysis, "act")
    t = act.table
    assert (t.n_pairs, t.both_success, t.b, t.c, t.both_fail) == (29, 21, 2, 3, 3)
    assert act.test.p_exact == pytest.approx(1.0)
    assert act.test.p_mid == pytest.approx(1 - 10 / 32)
    assert act.diff.estimate == pytest.approx(1 / 29)
    assert act.only_baseline == [(3, 2)] and act.only_method == []


def test_medians_and_final_errors(analysis):
    servo = _summary(analysis, "servo")
    assert sorted(servo.dock_times_s) == sorted([10.0 + p for p in (0, 1, 2, 3, 4, 7, 8, 9) for _ in range(3)])
    q1, med, q3 = np.percentile(servo.dock_times_s, [25, 50, 75])
    assert (q1, med, q3) == (11.75, 13.5, 17.25)
    assert len(servo.range_errors_cm) == 30
    assert max(servo.range_errors_cm) == pytest.approx(35.0)  # the 0.60 m fails
    assert sorted(servo.bearing_errors_deg)[:24] == [1.0] * 24


def test_duplicates_interventions_and_bad_rows_are_reported(analysis):
    issues = "\n".join(analysis.issues)
    assert "skipped: pose_id is not an integer: 'x'" in issues
    assert "duplicate act pose 0 round 1 (success) ignored" in issues
    assert "act pose 8 round 3 logged success with 1 intervention(s); scored as a fail" in issues
    aborts = analysis.scoring.aborts
    assert [(t.method, t.pose_id, t.round) for t in aborts] == [("act", 4, 1)]


def test_report_tables(analysis):
    report = at.render_report(analysis)
    assert "| servo | 24/30 | 80.0% | 62.7–90.5% |" in report  # Wilson 24/30: 0.626943, 0.904949
    assert "| bc | 9/30 | 30.0% |" in report
    assert "| act | 24/29 | 82.8% |" in report
    assert "| bc | 30 | 9 | 6 | 15 | 0 | 6.1e-05 | 3.05e-05 | -50.0 pp |" in report
    assert "| act | 29 | 21 | 3 | 2 | 3 | 1 | 0.688 | +3.4 pp |" in report
    assert "- `act` missing where `servo` has a trial: pose 3 round 2" in report
    assert "| servo | 13.5 (11.8–17.2), n=24 |" in report
    assert "| servo | 24 | 3 | 3 | 0 | 0 | 0 | 0 |" in report
    assert "| act | 24 | 0 | 0 | 4 | 1 | 1 | 4 |" in report
    assert "| 5 | 0/3 | 0/3 | 0/3 |" in report and "| 3 | 3/3 | 0/3 | 2/2 |" in report
    assert "act pose 4 round 1 — no end before next start" in report
    assert "Setting: not stated" in report
    # tables must not be broken by stray pipes: every table row has the header's column count
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---") or line.startswith("|:--"):
            ncol = lines[i - 1].count("|")
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == ncol, lines[j]
                j += 1


def test_missing_baseline_and_custom_order(trials_csv):
    a = at.analyze(trials_csv, baseline="dagger", methods=["act", "bc"], n_boot=100)
    assert a.order == ["act", "bc", "servo"]
    report = at.render_report(a)
    assert "No scored `dagger` trials" in report


def test_missing_columns_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("method,pose_id,round\nservo,1,1\n")
    assert at.main([str(bad), "--out-dir", str(tmp_path / "out")]) == 2
    assert "missing columns" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# CLI outputs: report and chart


def test_cli_writes_report_and_valid_png(trials_csv, tmp_path, capsys):
    from matplotlib import image as mpimg

    out = tmp_path / "report"
    assert at.main([str(trials_csv), "--out-dir", str(out), "--n-boot", "500", "--setting", "simulation"]) == 0
    assert "servo    24/30  80.0%" in capsys.readouterr().out
    report = (out / "report.md").read_text()
    assert "Setting: simulation." in report and "](success.png)" in report

    png = out / "success.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    img = mpimg.imread(str(png))
    assert img.shape[:2] == (630, 960)
    rgb = np.round(img[..., :3] * 255).astype(int)

    def hex_rgb(h):
        return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])

    assert (rgb[2, 2] == hex_rgb(at.BACKGROUND)).all()
    centroids = []
    for colour in at.SERIES_COLORS[:3]:
        mask = (np.abs(rgb - hex_rgb(colour)).sum(axis=2) <= 3)
        assert mask.sum() > 2000, colour  # a real bar, not an antialiasing fringe
        centroids.append(np.nonzero(mask)[1].mean())
    assert centroids == sorted(centroids)  # servo, bc, act from left to right, colours in order
    # bar heights follow the success rates: act (82.8 %) > servo (80 %) > bc (30 %)
    heights = [(np.abs(rgb - hex_rgb(c)).sum(axis=2) <= 3).any(axis=1).sum() for c in at.SERIES_COLORS[:3]]
    assert heights[2] > heights[0] > heights[1]


def test_chart_handles_method_without_trials(tmp_path):
    summaries = [
        at.MethodSummary("servo", 3, 4, stats.wilson_interval(3, 4), at.Counter(), 0, [], [], []),
        at.MethodSummary("bc", 0, 0, None, at.Counter(), 0, [], [], []),
    ]
    at.plot_success(summaries, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000


# --------------------------------------------------------------------------------------------
# Latency and gate logs


def _latency_record(backend, infer, total, age):
    return {"preprocess_ms": 2.0, "infer_ms": infer, "total_ms": total, "image_age_ms": age, "backend": backend}


def _gate_record(ticks, missed, p99, state="ok"):
    return {
        "state": state, "front_min_m": 0.8, "scale": 1.0,
        "period_ms": {"p50": 20.0, "p99": p99, "max": p99 + 3.0},
        "missed_deadlines": missed, "ticks": ticks, "cb_latency_us": {"p50": 40.0, "p99": 50.0 + ticks % 7},
    }


GATE_SEGMENT_A = [(100, 1, 20.5), (125, 1, 20.9), (150, 2, 21.4), (20, 0, 20.2), (45, 1, 20.6)]  # restart
GATE_SEGMENT_B = [(5000, 10, 20.4), (5025, 12, 22.0)]


def _write_jsonl_logs(tmp_path):
    lat = tmp_path / "latency.jsonl"
    with open(lat, "w") as f:
        for i in range(200):
            f.write(json.dumps(_latency_record("tensorrt", 1.0 + i, 5.0 + i, 30.0)) + "\n")
            if i % 50 == 0:
                f.write("---\n\n")
        for i in range(10):  # ros2 topic echo style wrapper with the payload as JSON text
            payload = json.dumps(_latency_record("onnxruntime", 40.0 + i, 45.0, 60.0))
            f.write(json.dumps({"t": i, "data": payload}) + "\n")
        f.write("not json\n")
        f.write('["a list"]\n')
    gate_a, gate_b = tmp_path / "gate_a.jsonl", tmp_path / "gate_b.jsonl"
    gate_a.write_text("".join(json.dumps(_gate_record(*r)) + "\n" for r in GATE_SEGMENT_A))
    gate_b.write_text("".join(json.dumps({"data": _gate_record(*r)}) + "\n" for r in GATE_SEGMENT_B))
    return lat, gate_a, gate_b


def test_latency_and_gate_from_jsonl(trials_csv, tmp_path):
    lat, gate_a, gate_b = _write_jsonl_logs(tmp_path)
    a = at.analyze(trials_csv, latency_sources=["jetson=%s" % lat], gate_sources=[str(gate_a), "b=%s" % gate_b],
                   n_boot=100)
    trt, ort = sorted(a.latency, key=lambda s: s.backend, reverse=True)
    assert (trt.source, trt.backend, trt.n) == ("jetson", "tensorrt", 200)
    infer = np.arange(1.0, 201.0)
    assert trt.fields["infer_ms"] == tuple(np.percentile(infer, [50, 99]))
    assert trt.fields["total_ms"] == tuple(np.percentile(infer + 4.0, [50, 99]))
    assert trt.fields["image_age_ms"] == (30.0, 30.0)
    assert (ort.backend, ort.n, ort.fields["infer_ms"][0]) == ("onnxruntime", 10, 44.5)
    assert any("2 line(s) are not JSON objects" in i and "lines 219, 220" in i for i in a.issues), a.issues

    g_a, g_b = a.gate
    assert (g_a.source, g_a.n_status, g_a.restarts) == ("gate_a", 5, 1)
    # increments 25 + 25, restart counts the new node's 20 ticks from zero, then 25
    assert (g_a.ticks, g_a.missed) == (95, 0 + 1 + 0 + 1)
    assert g_a.period_p99_ms == pytest.approx(20.6) and g_a.period_p99_worst_ms == pytest.approx(21.4)
    assert g_a.period_max_ms == pytest.approx(24.4)
    assert (g_b.source, g_b.ticks, g_b.missed, g_b.restarts) == ("b", 25, 2, 0)  # first record = baseline

    report = at.render_report(a)
    assert "| jetson | tensorrt | 200 | 100.5 | 198.0 |" in report
    assert "| jetson | onnxruntime | 10 | 44.5 | 48.9* |" in report
    assert "* p99 from fewer than 100 samples." in report
    assert "| gate_a | 5 | 95 | 2 (2.105%) | 20.00 | 20.60 | 21.40 | 1.40 | 24.40 |" in report
    assert "| b | 2 | 25 | 2 (8.000%) |" in report


def _write_bag(path: Path, layout: str):
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_FOXY if layout == "foxy" else Stores.ROS2_JAZZY)
    string = store.types["std_msgs/msg/String"]
    plugin = StoragePlugin.SQLITE3 if layout == "foxy" else StoragePlugin.MCAP
    with Writer(path, version=8 if layout == "foxy" else 9, storage_plugin=plugin) as writer:
        lat = writer.add_connection(at.LATENCY_TOPIC, "std_msgs/msg/String", typestore=store)
        gate = writer.add_connection(at.GATE_TOPIC, "std_msgs/msg/String", typestore=store)
        other = writer.add_connection("/episode/event", "std_msgs/msg/String", typestore=store)
        t = 1_000_000_000
        writer.write(other, t, store.serialize_cdr(string(data="start bc 1 1"), "std_msgs/msg/String"))
        for i in range(120):
            payload = json.dumps(_latency_record("tensorrt", 10.0 + i % 12, 15.0, 50.0))
            writer.write(lat, t + i * 100_000_000, store.serialize_cdr(string(data=payload), "std_msgs/msg/String"))
        for i, rec in enumerate(GATE_SEGMENT_A):
            payload = json.dumps(_gate_record(*rec))
            writer.write(gate, t + i * 500_000_000, store.serialize_cdr(string(data=payload), "std_msgs/msg/String"))
        writer.write(gate, t + 10**10, store.serialize_cdr(string(data="garbage"), "std_msgs/msg/String"))


@pytest.mark.parametrize("layout", ["foxy", "jazzy"])
def test_logs_from_rosbag(trials_csv, tmp_path, layout):
    bag = tmp_path / "trial_bags" / ("run_" + layout)
    bag.parent.mkdir()
    _write_bag(bag, layout)
    for source in (str(bag), str(bag.parent)):  # the bag itself, or a directory holding bags
        a = at.analyze(trials_csv, latency_sources=["onboard=" + source], gate_sources=[source], n_boot=100)
        (lat,) = a.latency
        assert (lat.source, lat.backend, lat.n) == ("onboard", "tensorrt", 120)
        assert lat.fields["infer_ms"] == tuple(np.percentile([10.0 + i % 12 for i in range(120)], [50, 99]))
        (gate,) = a.gate
        assert (gate.n_status, gate.ticks, gate.missed, gate.restarts) == (5, 95, 2, 1)
        assert any("1 /safety_gate/status message(s) are not JSON objects" in i for i in a.issues)


def test_missing_log_path_is_an_input_error(trials_csv, tmp_path, capsys):
    rc = at.main([str(trials_csv), "--out-dir", str(tmp_path), "--gate", str(tmp_path / "nope.jsonl")])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_parse_source():
    assert at.parse_source("jetson=bags/run1") == ("jetson", Path("bags/run1"))
    assert at.parse_source("bags/run=1") == ("run=1", Path("bags/run=1"))
    assert at.parse_source("logs/lat.jsonl") == ("lat", Path("logs/lat.jsonl"))
    assert at.parse_source("laptop=~/lat.jsonl") == ("laptop", Path.home() / "lat.jsonl")


# --------------------------------------------------------------------------------------------
# make_schedule


def test_schedule_is_complete_balanced_and_reproducible():
    methods = ["servo", "bc", "act"]
    trials = ms.make_schedule(methods, range(10), rounds=3, seed=2026)
    assert len(trials) == 90 and [t.trial for t in trials] == list(range(1, 91))
    slots = [(t.method, t.pose_id, t.round) for t in trials]
    assert sorted(slots) == sorted((m, p, r) for m in methods for p in range(10) for r in (1, 2, 3))
    for r in (1, 2, 3):
        blocks = [[t for t in trials if t.round == r and t.block == b] for b in range(1, 11)]
        for block in blocks:  # one pose per block, every method once, back to back
            assert len({t.pose_id for t in block}) == 1 and sorted(t.method for t in block) == sorted(methods)
            assert [t.trial for t in block] == list(range(block[0].trial, block[0].trial + 3))
    positions = {m: [0, 0, 0] for m in methods}
    for t in trials:
        block = [u for u in trials if u.round == t.round and u.block == t.block]
        positions[t.method][block.index(t)] += 1
    assert positions == {m: [10, 10, 10] for m in methods}  # each method 10x first, second, third
    assert trials == ms.make_schedule(methods, range(10), rounds=3, seed=2026)
    orders = [[t.pose_id for t in trials if t.round == r and t.method == "servo"] for r in (1, 2, 3)]
    assert len({tuple(o) for o in orders}) == 3  # rounds are shuffled independently
    other_seed = ms.make_schedule(methods, range(10), rounds=3, seed=7)
    assert [t.pose_id for t in other_seed] != [t.pose_id for t in trials]


def test_blocked_schedule_rotates_methods_per_round():
    trials = ms.make_schedule(["servo", "bc", "act"], range(4), rounds=3, seed=1, order="blocked")
    firsts = [[t.method for t in trials if t.round == r][::4] for r in (1, 2, 3)]
    assert firsts == [["servo", "bc", "act"], ["bc", "act", "servo"], ["act", "servo", "bc"]]
    for r in (1, 2, 3):
        rows = [t for t in trials if t.round == r]
        per_method = [[t.pose_id for t in rows if t.method == m] for m in ("servo", "bc", "act")]
        assert per_method[0] == per_method[1] == per_method[2]  # same pose order for every method


@pytest.mark.parametrize(
    "methods,poses,rounds,order",
    [([], range(3), 1, "interleaved"), (["a", "a"], range(3), 1, "interleaved"), (["a b"], range(3), 1, "interleaved"),
     (["a"], [], 1, "interleaved"), (["a"], [1, 1], 1, "interleaved"), (["a"], range(3), 0, "interleaved"),
     (["a"], range(3), 1, "random")],
)
def test_schedule_rejects_bad_arguments(methods, poses, rounds, order):
    with pytest.raises(ValueError):
        ms.make_schedule(methods, poses, rounds, 0, order)


def test_schedule_cli_files_and_completeness_check(trials_csv, tmp_path):
    csv_path, md_path = tmp_path / "s" / "schedule.csv", tmp_path / "s" / "schedule.md"
    assert ms.main(["--out", str(csv_path), "--markdown", str(md_path)]) == 0
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(ms.CSV_COLUMNS) and len(rows) == 90
    assert all(r["event"] == "start %s %s %s" % (r["method"], r["pose_id"], r["round"]) for r in rows)
    sheet = md_path.read_text()
    assert sheet.count("| `start ") == 90 and "## Round 3" in sheet

    a = at.analyze(trials_csv, schedule_path=csv_path, n_boot=100)
    assert a.schedule_missing == [("act", 3, 2)] and a.schedule_extra == []
    assert "- Scheduled but not scored (1): act pose 3 round 2" in at.render_report(a)


def test_pose_table_matches_start_poses_yaml():
    yaml = pytest.importorskip("yaml")
    poses = {p["pose_id"]: p for p in yaml.safe_load(START_POSES.read_text())["start_poses"]}
    table = ms.pose_marking_table(START_POSES).splitlines()
    assert len(table) == 2 + len(poses)
    row5 = next(line for line in table if line.startswith("| 5 |"))
    # pose 5: 0.85 m out, 0.50 m left, turned 40 deg left -> tip 0.30 m ahead
    assert row5 == "| 5 | 0.85 | +0.50 | +40 | 0.62 | +0.69 |"
    assert next(line for line in table if line.startswith("| 0 |")) == "| 0 | 1.00 | +0.00 | 0 | 0.70 | +0.00 |"
