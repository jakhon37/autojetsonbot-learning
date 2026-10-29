"""camera_node, policy_runner and the episode CLI on a live ROS graph (runs under colcon test).

Everything runs in-process on a private context and a random ROS_DOMAIN_ID, so parallel
test runs on the same host (docker --network host) do not see each other's topics.
"""

import json
import os
import random
import subprocess
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

import cv2  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from jbl_nodes import imgconv  # noqa: E402
from jbl_nodes.camera_node import CameraNode, RateLimiter, parse_device, render_test_pattern  # noqa: E402
from jbl_nodes.episode_cli import (  # noqa: E402
    EVENT_TOPIC,
    EpisodeEvent,
    event_qos,
    format_episode_event,
    parse_episode_event,
)
from jbl_nodes.judge_logic import is_intervention  # noqa: E402
from jbl_nodes.policy_runner import PolicyRunner  # noqa: E402

CONSTANT = [0.4, -0.2]
CONSTANT_TWIST = (0.4 * 0.25, -0.2 * 1.5)
LATENCY_KEYS = {"preprocess_ms", "infer_ms", "total_ms", "image_age_ms", "backend"}


class Graph(object):
    """A private ROS context with a probe node and one executor for everything."""

    def __init__(self, context):
        self.context = context
        self.executor = SingleThreadedExecutor(context=context)
        self.nodes = []
        self.probe = self.add(Node("probe", context=context))
        self.received = {}

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def record(self, msg_type, topic, qos=10):
        log = self.received.setdefault(topic, [])
        self.probe.create_subscription(msg_type, topic, lambda m: log.append((m, time.monotonic())), qos)
        return log

    def spin_for(self, seconds, every=None, period=0.05):
        """Spin for ``seconds``, calling ``every()`` each ``period`` if given."""
        end = time.monotonic() + seconds
        next_call = time.monotonic()
        while time.monotonic() < end:
            if every is not None and time.monotonic() >= next_call:
                every()
                next_call += period
            self.executor.spin_once(timeout_sec=0.005)

    def spin_until(self, predicate, timeout, every=None, period=0.05):
        end = time.monotonic() + timeout
        next_call = time.monotonic()
        while time.monotonic() < end:
            if predicate():
                return True
            if every is not None and time.monotonic() >= next_call:
                every()
                next_call += period
            self.executor.spin_once(timeout_sec=0.005)
        return predicate()

    def close(self):
        for node in reversed(self.nodes):
            self.executor.remove_node(node)
            node.destroy_node()
        self.executor.shutdown()


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(random.SystemRandom().randint(30, 99)))
    context = Context()
    rclpy.init(context=context)
    g = Graph(context)
    try:
        yield g
    finally:
        g.close()
        context.try_shutdown()


def vw(entry):
    msg = entry[0]
    return (msg.linear.x, msg.angular.z)


def is_constant(entry):
    return np.allclose(vw(entry), CONSTANT_TWIST, atol=1e-6)


def is_zero(entry):
    return vw(entry) == (0.0, 0.0)


def runner(graph, **params):
    overrides = [Parameter("backend", value="constant"), Parameter("constant_action", value=CONSTANT)]
    overrides += [Parameter(k, value=v) for k, v in params.items()]
    return graph.add(PolicyRunner(context=graph.context, parameter_overrides=overrides))


def camera(graph, **params):
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    return graph.add(CameraNode(context=graph.context, parameter_overrides=overrides))


def stamped_image(graph, encoding, offset_s=0.0, zero_stamp=False):
    frame = render_test_pattern(3, 64, 48)
    if encoding == "rgb8":
        frame = frame[:, :, ::-1]
    elif encoding == "mono8":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    msg = imgconv.numpy_to_image(np.ascontiguousarray(frame), encoding)
    if not zero_stamp:
        ns = graph.probe.get_clock().now().nanoseconds + int(offset_s * 1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 10**9, ns % 10**9
    return msg


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


# ---------------------------------------------------------------- pure helpers


def test_episode_event_text_round_trip():
    start = parse_episode_event("start bc 3 2")
    assert start == EpisodeEvent("start", method="bc", pose_id=3, round=2)
    assert format_episode_event(start) == "start bc 3 2"
    end = parse_episode_event("end fail  timeout after\t30 s")
    assert (end.kind, end.outcome, end.reason) == ("end", "fail", "timeout after 30 s")
    assert format_episode_event(end) == "end fail timeout after 30 s"
    assert format_episode_event(parse_episode_event("end success")) == "end success"
    note = parse_episode_event("intervention  lifted it\tbefore the wall")
    assert note == EpisodeEvent("intervention", reason="lifted it before the wall")
    assert format_episode_event(note) == "intervention lifted it before the wall"
    assert is_intervention(format_episode_event(note))  # what dock_judge and trial_logger match
    assert format_episode_event(parse_episode_event("intervention")) == "intervention"
    for bad in ("", "begin bc 1 1", "start bc 1", "start bc x 1", "start bc 1 -1", "end maybe", "end"):
        with pytest.raises(ValueError):
            parse_episode_event(bad)


def test_rate_limiter_decimates_30_to_15_hz_despite_jitter():
    limiter = RateLimiter(15.0, 0.5 / 30.0)
    rng = random.Random(0)
    times = [i / 30.0 + rng.uniform(-0.008, 0.008) for i in range(300)]
    passed = [i for i, t in enumerate(times) if limiter.ready(t)]
    assert passed == list(range(0, 300, 2))
    assert not RateLimiter(0.0, 0.01).ready(1.0)


def test_parse_device():
    assert parse_device(0) == 0 and parse_device(" 2 ") == 2
    assert parse_device("/dev/video1") == "/dev/video1"
    assert parse_device("videotestsrc ! appsink") == "videotestsrc ! appsink"
    with pytest.raises(ValueError):
        parse_device("")


def test_test_pattern_moves():
    a, b = render_test_pattern(0, 640, 480), render_test_pattern(5, 640, 480)
    assert a.shape == (480, 640, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 0


# ---------------------------------------------------------------- camera_node


def test_camera_publishes_stamped_raw_and_compressed(graph):
    raw = graph.record(Image, "/camera/image_raw")
    jpeg = graph.record(CompressedImage, "/camera/image_raw/compressed")
    camera(graph, source="test_pattern")
    assert graph.spin_until(lambda: len(raw) >= 45 and len(jpeg) >= 22, timeout=6.0), (len(raw), len(jpeg))

    msg = raw[-1][0]
    assert (msg.encoding, msg.width, msg.height, msg.step) == ("bgr8", 640, 480, 1920)
    assert msg.header.frame_id == "camera_link"
    ages = [graph.probe.get_clock().now().nanoseconds * 1e-9 - stamp_s(m) for m, _ in raw[-5:]]
    assert all(0.0 <= age < 1.0 for age in ages), ages
    assert not np.array_equal(imgconv.image_to_numpy(raw[-1][0]), imgconv.image_to_numpy(raw[-2][0]))

    raw_stamps = [stamp_s(m) for m, _ in raw]
    jpeg_stamps = [stamp_s(m) for m, _ in jpeg]
    assert all(b > a for a, b in zip(raw_stamps, raw_stamps[1:]))
    assert np.median(np.diff(raw_stamps)) == pytest.approx(1 / 30.0, rel=0.15)
    assert np.median(np.diff(jpeg_stamps)) == pytest.approx(1 / 15.0, rel=0.15)
    in_window = [s for s in jpeg_stamps if raw_stamps[0] <= s <= raw_stamps[-1]]
    shared = set(in_window) & set(raw_stamps)
    assert len(shared) >= 0.8 * len(in_window), "compressed frames must carry their raw frame's stamp"

    cmsg = jpeg[-1][0]
    assert cmsg.format == "jpeg" and cmsg.header.frame_id == "camera_link"
    assert imgconv.compressed_to_bgr(cmsg).shape == (480, 640, 3)


def test_camera_survives_a_missing_device(graph):
    raw = graph.record(Image, "/camera/image_raw")
    node = camera(graph, source="camera", device="/nonexistent/video0")
    graph.spin_for(1.0)
    assert raw == [] and node._thread.is_alive()
    t0 = time.monotonic()
    graph.executor.remove_node(node)
    graph.nodes.remove(node)
    node.destroy_node()
    assert time.monotonic() - t0 < 2.0 and not node._thread.is_alive()


def test_camera_reopens_a_source_after_read_failures(graph, tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    assert writer.isOpened()
    for i in range(5):
        writer.write(render_test_pattern(i, 320, 240))
    writer.release()

    raw = graph.record(Image, "/camera/image_raw")
    camera(graph, source="camera", device=path, width=160, height=120)
    # The clip ends after 5 frames; every later read fails until the node reopens the file.
    assert graph.spin_until(lambda: len(raw) >= 2, timeout=5.0), len(raw)
    assert (raw[0][0].width, raw[0][0].height) == (160, 120)  # resized to the requested size
    assert stamp_s(raw[1][0]) - stamp_s(raw[0][0]) > 0.3


# ---------------------------------------------------------------- policy_runner


def test_policy_runner_follows_episode_events(graph):
    jpeg = graph.record(CompressedImage, "/camera/image_raw/compressed")
    cmds = graph.record(Twist, "/cmd_vel_in")
    latency = graph.record(String, "/policy/latency")
    events = graph.probe.create_publisher(String, EVENT_TOPIC, event_qos())
    camera(graph, source="test_pattern")
    runner(graph)

    assert graph.spin_until(lambda: len(jpeg) >= 5, timeout=5.0)
    graph.spin_for(0.5)
    assert cmds == [], "a runner waiting for an episode must stay silent"

    events.publish(String(data="start bc 1 1"))
    assert graph.spin_until(lambda: any(is_constant(c) for c in cmds), timeout=3.0), [vw(c) for c in cmds]
    n0 = len(cmds)
    graph.spin_for(1.0)
    window = cmds[n0:]
    assert 6 <= len(window) <= 14, "expected ~10 Hz, got %d in 1 s" % len(window)
    assert sum(is_constant(c) for c in window) >= len(window) - 1

    assert latency, "no /policy/latency while enabled"
    report = json.loads(latency[-1][0].data)
    assert set(report) == LATENCY_KEYS and report["backend"] == "constant"
    for key in LATENCY_KEYS - {"backend"}:
        assert isinstance(report[key], float)
    assert 0.0 <= report["preprocess_ms"] <= report["total_ms"]
    assert -1.0 <= report["image_age_ms"] <= 200.0

    events.publish(String(data="end success"))
    assert graph.spin_until(lambda: is_zero(cmds[-1]), timeout=2.0)
    n_end = len(cmds)
    graph.spin_for(0.5)
    assert len(cmds) == n_end, "a disabled runner must not publish"

    # An operator intervention stops it too, before the judge's "end fail intervention".
    events.publish(String(data="start bc 2 1"))
    assert graph.spin_until(lambda: is_constant(cmds[-1]), timeout=3.0)
    events.publish(String(data="intervention lifted it"))
    assert graph.spin_until(lambda: is_zero(cmds[-1]), timeout=2.0)
    n_end = len(cmds)
    graph.spin_for(0.5)
    assert len(cmds) == n_end, "a runner stopped by an intervention must not publish"


def test_policy_runner_zeroes_stale_future_and_missing_images(graph):
    cmds = graph.record(Twist, "/cmd_vel_in")
    latency = graph.record(String, "/policy/latency")
    images = graph.probe.create_publisher(Image, "/test/image_raw", 10)
    runner(graph, image_topic="/test/image_raw", enabled_on_start=True, rate_hz=20.0)

    assert graph.spin_until(lambda: len(cmds) >= 3, timeout=3.0)
    assert all(is_zero(c) for c in cmds), "no image yet -> zero"

    def phase(seconds, **image_args):
        start = len(cmds)
        graph.spin_for(seconds, every=lambda: images.publish(stamped_image(graph, "rgb8", **image_args)))
        return cmds[start:]

    stale = phase(0.6, offset_s=-1.0)
    assert len(stale) >= 5 and all(is_zero(c) for c in stale)
    future = phase(0.4, offset_s=+1.0)
    assert len(future) >= 3 and all(is_zero(c) for c in future)
    assert latency == []

    fresh = phase(0.6)
    assert sum(is_constant(c) for c in fresh) >= 5
    assert latency and json.loads(latency[-1][0].data)["image_age_ms"] < 200.0

    # The camera stops: the last frame ages past max_image_age and the runner falls back to zero.
    assert graph.spin_until(lambda: is_zero(cmds[-1]), timeout=1.0)

    unstamped = phase(0.5, zero_stamp=True)
    assert sum(is_constant(c) for c in unstamped) >= 4, "unstamped frames are aged from arrival"


def test_policy_runner_rejects_bad_parameters(graph, tmp_path):
    with pytest.raises(ValueError, match="image_reliability"):
        runner(graph, image_reliability="sometimes")
    with pytest.raises(ValueError, match="unknown backend"):
        runner(graph, backend="tflite")
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"")
    (tmp_path / "policy_meta.json").write_text(json.dumps({"input_shape": [1, 3, 96, 96]}))
    with pytest.raises(ValueError, match="input_shape"):
        runner(graph, backend="onnxruntime", model_path=str(model))

    from ament_index_python.packages import get_package_prefix

    exe = os.path.join(get_package_prefix("jbl_nodes"), "lib", "jbl_nodes", "policy_runner")
    proc = subprocess.run(
        [exe, "--ros-args", "-p", "model_path:=/nonexistent/policy.onnx"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    assert proc.returncode == 1
    assert b"cannot start: model file not found" in proc.stdout and b"Traceback" not in proc.stdout


def test_late_runner_gets_the_start_and_honours_method_filter(graph):
    cmds = graph.record(Twist, "/cmd_vel_in")
    events = graph.probe.create_publisher(String, EVENT_TOPIC, event_qos())
    images = graph.probe.create_publisher(Image, "/test/image_mono", 10)
    publish_image = lambda: images.publish(stamped_image(graph, "mono8"))  # noqa: E731

    events.publish(String(data="start bc 2 1"))  # before the runner exists: transient_local replays it
    graph.spin_for(0.2)
    runner(graph, image_topic="/test/image_mono", rate_hz=20.0, method_filter="bc, act")
    assert graph.spin_until(lambda: cmds and is_constant(cmds[-1]), timeout=4.0, every=publish_image)

    events.publish(String(data="end abort operator stop"))
    assert graph.spin_until(lambda: is_zero(cmds[-1]), timeout=2.0, every=publish_image)
    n = len(cmds)
    events.publish(String(data="start servo 3 1"))
    graph.spin_for(0.5, every=publish_image)
    assert len(cmds) == n, "a start for a filtered-out method must not enable the runner"

    events.publish(String(data="start act 3 1"))
    assert graph.spin_until(lambda: is_constant(cmds[-1]), timeout=2.0, every=publish_image)
    events.publish(String(data="start servo 4 1"))  # another method took over without an end
    assert graph.spin_until(lambda: is_zero(cmds[-1]), timeout=2.0, every=publish_image)
    n = len(cmds)
    graph.spin_for(0.3, every=publish_image)
    assert len(cmds) == n


# ---------------------------------------------------------------- episode CLI


def _episode_executable():
    from ament_index_python.packages import get_package_prefix

    path = os.path.join(get_package_prefix("jbl_nodes"), "lib", "jbl_nodes", "episode")
    assert os.access(path, os.X_OK), "episode executable not installed at %s" % path
    return path


def _run_cli(graph, *args):
    proc = subprocess.Popen(
        [_episode_executable()] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert graph.spin_until(lambda: proc.poll() is not None, timeout=10.0), "episode CLI did not exit"
    out, err = proc.communicate()
    return proc.returncode, out.decode(), err.decode()


def test_episode_cli_publishes_once_and_exits(graph):
    events = graph.record(String, EVENT_TOPIC, event_qos())

    # A generous --timeout: Fast DDS occasionally needs 2-3 s to discover a fresh process,
    # and this test is about delivery, not discovery speed.
    code, out, err = _run_cli(graph, "--timeout", "8", "start", "bc", "3", "2")
    assert code == 0, err
    assert "start bc 3 2" in out
    assert graph.spin_until(lambda: len(events) == 1, timeout=1.0)
    assert events[0][0].data == "start bc 3 2"

    code, _, err = _run_cli(graph, "--timeout", "8", "end", "fail", "timeout", "after", "30", "s")
    assert code == 0, err
    assert graph.spin_until(lambda: len(events) == 2, timeout=1.0)
    assert events[1][0].data == "end fail timeout after 30 s"

    code, out, err = _run_cli(graph, "--timeout", "8", "intervention", "lifted", "it  off")
    assert code == 0, err
    assert graph.spin_until(lambda: len(events) == 3, timeout=1.0)
    assert events[2][0].data == "intervention lifted it off"

    code, _, err = _run_cli(graph, "start", "bc", "three", "1")
    assert code == 2 and "pose_id" in err
    code, _, err = _run_cli(graph, "end", "abort", "--timeout", "3")
    assert code == 2 and "options go before" in err
    graph.spin_for(0.2)
    assert len(events) == 3


def test_episode_cli_reports_when_nobody_listens(graph):
    t0 = time.monotonic()
    code, _, err = _run_cli(graph, "end", "abort")
    assert code == 1 and "no subscriber" in err and "not delivered" in err
    assert time.monotonic() - t0 < 6.0
