"""scripts/stop_labels.py: stop examples in demonstrations and datasets (v3 BC fix analysis)."""

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
    e = sl.ending([(1.0, 0.01), (1.9, 0.015)], t1=2.0)  # creep the judge calls still
    assert e["max_v_last1s"] == pytest.approx(0.015) and e["stop_for_s"] == 0.0


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
