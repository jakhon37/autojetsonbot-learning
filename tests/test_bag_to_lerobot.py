"""bag_to_lerobot.py and jbl_episodes.py: resampling rules, episode splitting, metadata and the
written LeRobot dataset, checked against an independent oracle built from what the synthetic
bags contain."""

import json
import subprocess
import sys

import numpy as np
import pytest

import bag_to_lerobot as b2l
import jbl_episodes
import make_synthetic_bags as msb
from conftest import FPS, REPO_ID, TOOLS

S = 1_000_000_000  # ns per second


# --------------------------------------------------------------------------------------------
# Oracle


def episode_windows(bag):
    """[(start_ns, end_ns, start event or None, end event or None)] the converter should find."""
    if not bag.events:
        return [(bag.start_ns, bag.end_ns, None, None)]
    windows, open_ = [], None
    for t, text in bag.events:
        words = text.split()
        if words[0] == "start":
            open_ = (t, words)
        elif words[0] == "end" and open_ is not None:
            windows.append((open_[0], t, open_[1], words))
            open_ = None
    if open_ is not None:
        windows.append((open_[0], bag.end_ns, open_[1], None))
    return windows


def expected_ticks(bag, start_ns, end_ns, fps=FPS, timeout_s=0.3, keep_idle_start=False):
    """[(T, image index, state, action)] by direct lookup in the recorded message lists."""
    img_t = np.array([t for t, _ in bag.images], dtype=np.int64)
    odom_t = np.array([t for t, _, _ in bag.odom], dtype=np.int64)
    cmd_t = np.array([t for t, _, _ in bag.cmd], dtype=np.int64)
    out = []
    for k in range(10**6):
        T = start_ns + k * S // fps
        if T >= end_ns:
            break
        i = int(np.searchsorted(img_t, T, side="right")) - 1
        if i < 0 or img_t[i] < start_ns:
            continue
        o = int(np.searchsorted(odom_t, T, side="right")) - 1
        c = int(np.searchsorted(cmd_t, T, side="right")) - 1
        state = bag.odom[o][1:] if o >= 0 else (0.0, 0.0)
        fresh = c >= 0 and T - cmd_t[c] <= int(timeout_s * S)
        action = bag.cmd[c][1:] if fresh else (0.0, 0.0)
        if not keep_idle_start and not out and not any(action):
            continue
        out.append((T, bag.images[i][1], state, action))
    return out


def episode_frames(root, episode_index):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root, episodes=[episode_index], return_uint8=True)
    return [ds[i] for i in range(len(ds))]


# --------------------------------------------------------------------------------------------
# Pure logic


def test_parse_event():
    assert b2l.parse_event("start teleop 3 1") == b2l.EpisodeEvent("start", "teleop", 3, 1)
    assert b2l.parse_event("end fail pushed into  wall") == b2l.EpisodeEvent(
        "end", outcome="fail", reason="pushed into wall")
    assert b2l.parse_event(" end success ").outcome == "success"
    assert b2l.parse_event("intervention lifted it") == b2l.EpisodeEvent("intervention", reason="lifted it")
    for bad in ("", "hello", "start teleop 3", "start teleop x 1", "start teleop -1 0", "end", "end maybe"):
        with pytest.raises(ValueError):
            b2l.parse_event(bad)


def _run(messages, **kwargs):
    warnings = []
    kwargs.setdefault("keep_idle_start", True)
    items = list(b2l.resample(messages, warn=warnings.append, **kwargs))
    ticks = [x for x in items if isinstance(x, b2l.Tick)]
    ended = [x.episode for x in items if isinstance(x, b2l.EpisodeEnded)]
    return ticks, ended, warnings


def test_resample_tick_rules():
    M = b2l.Msg
    ms = 1_000_000
    messages = [
        M("odom", 0, (0.1, 0.0)),
        M("cmd", 50 * ms, (0.2, 0.5)),
        M("image", 150 * ms, "img-a"),  # ticks at 0 and 100 ms come before the first frame
        M("odom", 200 * ms, (0.3, -0.1)),  # exactly at a tick: "at or before" includes it
        M("image", 260 * ms, "img-b"),
        M("cmd", 300 * ms, (0.25, 1.0)),
        M("odom", 650 * ms, (0.0, 0.0)),  # last message
    ]
    ticks, ended, warnings = _run(messages, fps=10, whole_bag=(0, 650 * ms + 1))
    assert [t.t_ns for t in ticks] == [200 * ms, 300 * ms, 400 * ms, 500 * ms, 600 * ms]
    assert [t.image for t in ticks] == ["img-a", "img-b", "img-b", "img-b", "img-b"]
    assert ticks[0].state == (0.3, -0.1)
    # cmd at 50 ms: 150 ms old at 200 ms (fresh); cmd at 300 ms: exactly 0.3 s old at 600 ms
    # (still fresh, the timeout is "older than 0.3 s").
    assert [t.action for t in ticks] == [(0.2, 0.5), (0.25, 1.0), (0.25, 1.0), (0.25, 1.0), (0.25, 1.0)]
    assert len(ended) == 1 and ended[0].num_frames == 5 and ended[0].outcome is None
    assert ended[0].max_image_age_ns == 600 * ms - 260 * ms
    assert warnings == []

    later = messages[:-1] + [M("odom", 700 * ms + 1, (0.0, 0.0))]
    ticks, _, _ = _run(later, fps=10, whole_bag=(0, 800 * ms))
    assert ticks[-1].t_ns == 700 * ms and ticks[-1].action == (0.0, 0.0)  # command 0.4 s old


def test_resample_drops_idle_start():
    M = b2l.Msg
    ms = 1_000_000
    messages = [M("event", 0, "start teleop 1 0")]
    messages += [M("image", t * ms, t) for t in range(0, 3000, 50)]
    messages += [M("cmd", t * ms, (0.1, 0.0)) for t in range(1520, 2500, 40)]  # operator reacts at 1.52 s
    messages += [M("event", 3000 * ms, "end success")]
    messages.sort(key=lambda m: (m.t_ns, m.kind != "event"))
    ticks, ended, _ = _run(messages, fps=10, keep_idle_start=False)
    assert ticks[0].t_ns == 1600 * ms and ticks[0].action == (0.1, 0.0)
    assert ticks[-1].t_ns == 2900 * ms and ticks[-1].action == (0.0, 0.0)  # the stop at the end stays
    assert ended[0].idle_start_ticks == 16 and ended[0].num_frames == 14
    kept, ended, _ = _run(messages, fps=10)
    assert len(kept) == 30 and ended[0].idle_start_ticks == 0


def test_resample_episode_events():
    M = b2l.Msg
    ms = 1_000_000
    frames = [M("image", t * ms, t) for t in range(0, 2000, 50)]
    events = [
        M("event", 120 * ms, "end success"),  # no episode open: warned, ignored
        M("event", 200 * ms, "start teleop 1 0"),
        M("event", 500 * ms, "end success"),
        M("event", 600 * ms, "start bc 2 1"),
        M("event", 900 * ms, "start bc 3 1"),  # supersedes the open episode
        M("event", 950 * ms, "intervention lifted it"),  # not a boundary, not a warning
        M("event", 1000 * ms, "what"),
        M("event", 1300 * ms, "end fail timeout"),
        M("event", 1400 * ms, "start servo 4 2"),  # never ended
    ]
    messages = sorted(frames + events, key=lambda m: (m.t_ns, m.kind != "event"))
    ticks, ended, warnings = _run(messages, fps=10)
    summary = [(e.method, e.pose_id, e.round, e.outcome, e.start_ns // ms, e.end_ns // ms, e.num_frames) for e in ended]
    assert ended[-1].end_ns == 1950 * ms + 1
    assert summary == [
        ("teleop", 1, 0, "success", 200, 500, 3),
        ("bc", 2, 1, "abort", 600, 900, 3),
        ("bc", 3, 1, "fail", 900, 1300, 4),
        ("servo", 4, 2, "abort", 1400, 1950, 6),  # ends 1 ns after the last message
    ]
    assert ended[1].reason == "no end event before the next start"
    assert ended[2].reason == "timeout"
    assert ended[3].reason == "recording ended before the end event"
    assert len(warnings) == 2 and "no episode was started" in warnings[0] and "malformed" in warnings[1]
    assert all(t.t_ns % (100 * ms) == 0 for t in ticks)


def test_resample_needs_log_time_order():
    M = b2l.Msg
    with pytest.raises(ValueError, match="log-time order"):
        _run([M("image", 10, "a"), M("image", 5, "b")], whole_bag=(5, 11))


def test_find_bags(synthetic_bags, bags_dir, tmp_path):
    found = b2l.find_bags([bags_dir])
    assert [name for _, name in found] == sorted(synthetic_bags)
    one = synthetic_bags["foxy_teleop"].path
    db3 = next(one.glob("*.db3"))
    assert b2l.find_bags([one, db3, one]) == [(one, "foxy_teleop"), (db3, db3.name)]
    with pytest.raises(FileNotFoundError):
        b2l.find_bags([tmp_path])
    with pytest.raises(FileNotFoundError):
        b2l.find_bags([tmp_path / "missing"])


def test_split_episodes():
    records = [{"episode_index": i, "pose_id": i % 5} for i in range(20)]
    train, val = jbl_episodes.split_episodes(records, 0.2, seed=3)
    assert len(val) == 4 and sorted(train + val) == list(range(20)) and not set(train) & set(val)
    assert jbl_episodes.split_episodes(records, 0.2, seed=3) == (train, val)
    assert jbl_episodes.split_episodes(records, 0.2, seed=4) != (train, val)
    train, val = jbl_episodes.split_episodes(records, 0.2, seed=0, group_by="pose_id")
    val_poses = {records[i]["pose_id"] for i in val}
    assert val_poses and not val_poses & {records[i]["pose_id"] for i in train}
    assert jbl_episodes.split_episodes(records[:1], 0.5) == ([0], [])
    assert jbl_episodes.split_episodes(records[:2], 0.1)[1] != []  # at least one val episode
    assert jbl_episodes.split_episodes(records[:5], 0.0) == ([0, 1, 2, 3, 4], [])
    with pytest.raises(ValueError):
        jbl_episodes.split_episodes(records, 1.0)


def test_episode_meta_roundtrip(tmp_path):
    rows = [{"episode_index": 1, "outcome": "fail"}, {"episode_index": 0, "outcome": None}]
    jbl_episodes.write_episode_meta(tmp_path, rows)
    assert [r["episode_index"] for r in jbl_episodes.read_episode_meta(tmp_path)] == [0, 1]
    assert jbl_episodes.filter_episodes(rows, outcomes=["none"]) == [{"episode_index": 0, "outcome": None}]
    assert jbl_episodes.load_episode_table(tmp_path / "other", total_episodes=2) == [
        {"episode_index": 0}, {"episode_index": 1}]
    with pytest.raises(ValueError, match="dataset has 3 episodes"):
        jbl_episodes.load_episode_table(tmp_path, total_episodes=3)
    with pytest.raises(ValueError, match="duplicate"):
        jbl_episodes.write_episode_meta(tmp_path, rows + rows)


# --------------------------------------------------------------------------------------------
# Synthetic bags -> dataset


def test_dataset_loads_with_lerobot(converted, synthetic_bags):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, report = converted
    ds = LeRobotDataset(REPO_ID, root=root)
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == FPS and info["robot_type"] == "autojetsonbot"
    assert info["features"][b2l.IMAGE_KEY]["dtype"] == "video"
    assert info["features"][b2l.IMAGE_KEY]["shape"] == [240, 320, 3]
    assert ds.num_episodes == len(report.episodes) == 5
    assert len(ds) == report.num_frames
    item = ds[0]
    assert tuple(item[b2l.IMAGE_KEY].shape) == (3, 240, 320)
    assert tuple(item[b2l.STATE_KEY].shape) == (2,) and tuple(item[b2l.ACTION_KEY].shape) == (2,)
    assert item["task"] == b2l.TASK
    assert report.storage.startswith("video")


def test_episode_metadata(converted, synthetic_bags):
    root, report = converted
    rows = jbl_episodes.read_episode_meta(root)
    assert rows == report.episodes
    assert all(set(r) == set(jbl_episodes.FIELDS) for r in rows)
    by_bag = {}
    for r in rows:
        by_bag.setdefault(r["bag"], []).append(r)
    assert sorted(by_bag) == ["foxy_raw_bgr", "foxy_teleop", "jazzy_raw_rgb", "jazzy_trials"]
    trials = [(r["method"], r["pose_id"], r["round"], r["outcome"], r["reason"], r["duration_s"])
              for r in by_bag["jazzy_trials"]]
    assert trials == [("teleop", 3, 1, "success", "", 3.05), ("bc", 5, 2, "fail", "timeout", 2.85)]
    whole = by_bag["foxy_teleop"][0]
    bag = synthetic_bags["foxy_teleop"]
    assert (whole["method"], whole["pose_id"], whole["round"], whole["outcome"]) == (None, None, None, None)
    assert whole["duration_s"] == round((bag.end_ns - bag.start_ns) / 1e9, 3)
    dropped = sorted((d["bag"], d["outcome"], d["reason"]) for d in report.dropped)
    assert dropped == [
        ("foxy_unterminated", "abort", "recording ended before the end event"),
        ("jazzy_trials", "abort", "operator stopped"),
    ]
    # The camera stall in jazzy_trials (2.0-2.45 s) is visible in the metadata.
    assert by_bag["jazzy_trials"][0]["max_image_age_s"] > 0.4
    assert by_bag["jazzy_trials"][1]["max_image_age_s"] < 0.1


def test_frames_follow_the_tick_rules(converted, synthetic_bags):
    """Every frame's image, state and action equal the oracle's at that tick."""
    root, report = converted
    for record in report.episodes:
        bag = synthetic_bags[record["bag"]]
        window = next(w for w in episode_windows(bag) if w[0] == record["start_time_ns"])
        expected = expected_ticks(bag, window[0], window[1])
        frames = episode_frames(root, record["episode_index"])
        duration_s = (window[1] - window[0]) / 1e9
        first_image_delay = min(t for t, _ in bag.images if t >= window[0]) - window[0]
        assert len(frames) == len(expected) == record["num_frames"]
        assert abs(len(frames) - (duration_s - first_image_delay / 1e9 - record["idle_start_s"]) * FPS) <= 1.0
        for frame, (T, image_index, state, action) in zip(frames, expected):
            rgb = frame[b2l.IMAGE_KEY].permute(1, 2, 0).numpy()
            assert msb.decode_index(rgb) == image_index, (record["bag"], T)
            np.testing.assert_allclose(frame[b2l.STATE_KEY].numpy(), np.float32(state), atol=1e-6)
            np.testing.assert_allclose(frame[b2l.ACTION_KEY].numpy(), np.float32(action), atol=1e-6)
            # The commands really are the known function of their log time.
            if any(action):
                rel = [bag.rel_s(t) for t, v, w in bag.cmd if (v, w) == tuple(action)][0]
                np.testing.assert_allclose(action, msb.command_at(rel), atol=1e-12)


def test_command_gaps_and_late_start_give_zero_action(converted, synthetic_bags):
    root, report = converted
    rows = {(r["bag"], r["pose_id"]): r for r in report.episodes}

    def actions_by_time(record):
        """[(tick time in s since the bag's time zero, action)]; tick times from the oracle
        because frame_index restarts at the first kept tick."""
        bag = synthetic_bags[record["bag"]]
        window = next(w for w in episode_windows(bag) if w[0] == record["start_time_ns"])
        ticks = expected_ticks(bag, window[0], window[1])
        frames = episode_frames(root, record["episode_index"])
        assert len(ticks) == len(frames)
        return [(bag.rel_s(T), f[b2l.ACTION_KEY].numpy()) for (T, _, _, _), f in zip(ticks, frames)]

    # Command gap 6.0-7.0 s in jazzy_trials (pose 5 episode): zero once the last command is
    # more than 0.3 s old, i.e. from about 6.25 s until commands resume.
    checked = 0
    for t, a in actions_by_time(rows[("jazzy_trials", 5)]):
        if 6.31 < t < 6.99:
            assert not a.any(), t
            checked += 1
        elif t < 5.95 or t > 7.05:
            assert a.any(), t
    assert checked == 6  # ticks 6.4 ... 6.9
    # foxy_teleop: the operator starts commanding at 1.0 s; the idle start is dropped.
    teleop = actions_by_time(rows[("foxy_teleop", None)])
    assert 1.0 <= teleop[0][0] < 1.1 and teleop[0][1].any()
    assert rows[("foxy_teleop", None)]["idle_start_s"] > 0


def test_colour_order_is_rgb(converted):
    root, report = converted
    for record in report.episodes:
        frame = episode_frames(root, record["episode_index"])[0][b2l.IMAGE_KEY].permute(1, 2, 0).numpy()
        left, right = msb.patch_colours(frame)
        assert left[0] > 180 and left[2] < 80, (record["bag"], left)  # red stays red
        assert right[2] > 180 and right[0] < 80, (record["bag"], right)


def test_keep_abort_and_image_storage(synthetic_bags, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    bags = [(synthetic_bags[n].path, n) for n in ("jazzy_trials", "foxy_unterminated")]
    root = tmp_path / "with_aborts"
    report = b2l.convert(bags, root, REPO_ID, keep_abort=True, video="off")
    assert report.storage.startswith("images")
    assert [(r["outcome"], r["pose_id"]) for r in report.episodes] == [
        ("success", 3), ("abort", 4), ("fail", 5), ("abort", 1)]
    assert report.dropped == []
    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.meta.features[b2l.IMAGE_KEY]["dtype"] == "image"
    assert len(ds) == report.num_frames
    first = ds[0][b2l.IMAGE_KEY]
    assert tuple(first.shape) == (3, 240, 320)
    expected = expected_ticks(synthetic_bags["jazzy_trials"], *episode_windows(synthetic_bags["jazzy_trials"])[0][:2])
    rgb = (first.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    assert msb.decode_index(rgb) == expected[0][1]


def test_storage_choice():
    use_video, encoder, text = b2l.choose_storage("auto")
    assert use_video and encoder.vcodec == "libsvtav1" and text == "video (libsvtav1)"
    use_video, encoder, text = b2l.choose_storage("auto", vcodec="no-such-codec")
    assert not use_video and encoder is None and text.startswith("images (video encoding unavailable")
    assert b2l.choose_storage("off")[0] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        b2l.choose_storage("on", vcodec="no-such-codec")


def test_slow_camera_frames_are_reused(converted, synthetic_bags):
    root, report = converted
    record = next(r for r in report.episodes if r["bag"] == "foxy_raw_bgr")
    indices = [msb.decode_index(f[b2l.IMAGE_KEY].permute(1, 2, 0).numpy())
               for f in episode_frames(root, record["episode_index"])]
    assert len(indices) > len(set(indices)) and indices == sorted(indices)
    assert record["max_image_age_s"] >= 0.1


def test_output_root_safety(synthetic_bags, tmp_path):
    bags = [(synthetic_bags["jazzy_raw_rgb"].path, "jazzy_raw_rgb")]
    root = tmp_path / "ds"
    b2l.convert(bags, root, REPO_ID, video="off")
    with pytest.raises(FileExistsError, match="--overwrite"):
        b2l.convert(bags, root, REPO_ID, video="off")
    b2l.convert(bags, root, REPO_ID, video="off", overwrite=True)
    other = tmp_path / "not_a_dataset"
    other.mkdir()
    (other / "keep.txt").write_text("precious")
    with pytest.raises(FileExistsError, match="not a LeRobot dataset"):
        b2l.convert(bags, other, REPO_ID, video="off", overwrite=True)
    assert (other / "keep.txt").read_text() == "precious"
    # A failed conversion leaves neither the target nor the staging directory behind.
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="no camera messages"):
        b2l.convert(bags, failed, REPO_ID, video="off", image_topic="/front_camera/image")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "not_a_dataset"]


def test_cli(synthetic_bags, tmp_path):
    root = tmp_path / "cli_ds"
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "bag_to_lerobot.py"), str(synthetic_bags["jazzy_raw_rgb"].path),
         "--root", str(root), "--repo-id", REPO_ID, "--video", "off"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 episodes, 14 frames at 10 fps" in proc.stdout
    proc = subprocess.run([sys.executable, str(TOOLS / "jbl_episodes.py"), "--root", str(root), "--split"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"train": [0], "val": []}
