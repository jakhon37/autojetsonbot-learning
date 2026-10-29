"""fake_robot on a live ROS graph (runs under colcon test): teleport, driving, sensor streams.

The node runs in-process on a private context and a random ROS_DOMAIN_ID, so parallel test
runs on the same host (docker --network host) do not see each other's topics.
"""

import json
import math
import os
import random
import subprocess
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from jbl_nodes import imgconv  # noqa: E402
from jbl_nodes.aruco_utils import MarkerObserver  # noqa: E402
from jbl_nodes.episode_cli import EVENT_TOPIC, event_qos  # noqa: E402
from jbl_nodes.fake_robot import GROUND_TRUTH_TOPIC, FakeRobot, default_start_poses_file  # noqa: E402
from jbl_nodes.judge_logic import front_range  # noqa: E402
from jbl_nodes.sim_world import load_start_poses, start_pose_to_world  # noqa: E402


class Graph(object):
    """A private ROS context with a probe node and one executor for everything."""

    def __init__(self, context):
        self.context = context
        self.executor = SingleThreadedExecutor(context=context)
        self.nodes = []
        self.probe = self.add(Node("probe", context=context))

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def record(self, msg_type, topic, qos=10):
        log = []
        self.probe.create_subscription(msg_type, topic, lambda m: log.append((m, time.monotonic())), qos)
        return log

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

    def spin_for(self, seconds, every=None, period=0.05):
        self.spin_until(lambda: False, seconds, every, period)

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


def truth(entry):
    return json.loads(entry[0].data)


def front(scan_msg):
    return front_range(
        scan_msg.ranges,
        scan_msg.angle_min,
        scan_msg.angle_increment,
        scan_msg.range_min,
        scan_msg.range_max,
    )


def stamp_s(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def forward(v):
    msg = Twist()
    msg.linear.x = v
    return msg


def test_start_teleports_then_cmd_vel_drives_towards_the_box(graph):
    odom = graph.record(Odometry, "/odom", qos_profile_sensor_data)
    scans = graph.record(LaserScan, "/scan", qos_profile_sensor_data)
    jpegs = graph.record(CompressedImage, "/camera/image_raw/compressed", qos_profile_sensor_data)
    raws = graph.record(Image, "/camera/image_raw")
    gt = graph.record(String, GROUND_TRUTH_TOPIC)
    events = graph.probe.create_publisher(String, EVENT_TOPIC, event_qos())
    cmd = graph.probe.create_publisher(Twist, "/cmd_vel", 10)
    graph.add(FakeRobot(context=graph.context, parameter_overrides=[Parameter("initial_pose_id", value=3)]))

    poses = load_start_poses(default_start_poses_file())
    start3, start0 = start_pose_to_world(poses[3]), start_pose_to_world(poses[0])
    assert graph.spin_until(lambda: gt and odom and scans, timeout=5.0)
    assert (truth(gt[-1])["x"], truth(gt[-1])["y"]) == pytest.approx((start3.x, start3.y), abs=1e-4)

    events.publish(String(data="start servo 0 2"))
    at_start0 = lambda: gt and truth(gt[-1])["x"] == pytest.approx(start0.x, abs=1e-4)  # noqa: E731
    assert graph.spin_until(at_start0, timeout=3.0)
    report = truth(gt[-1])
    assert set(report) == {"x", "y", "yaw", "collided", "bumper_to_box_m"}
    assert (report["y"], report["yaw"], report["collided"]) == (pytest.approx(0.0, abs=1e-4), 0.0, False)
    assert report["bumper_to_box_m"] == pytest.approx(1.0 - 0.097, abs=1e-3)

    # Fresh sensor data from the start pose: the lidar sees the box face 1.0 - 0.064 m ahead.
    n_scan, n_jpeg = len(scans), len(jpegs)
    assert graph.spin_until(lambda: len(scans) > n_scan + 1 and len(jpegs) > n_jpeg + 1, timeout=3.0)
    range_before = front(scans[-1][0])
    assert range_before == pytest.approx(0.936, abs=0.01)
    frame = imgconv.compressed_to_bgr(jpegs[-1][0])
    assert frame.shape == (480, 640, 3)
    seen = MarkerObserver().observe(frame)
    assert seen is not None and abs(seen.bearing_deg) < 1.0

    # Drive forward at 0.2 m/s for 1 s.
    x_before = odom[-1][0].pose.pose.position.x
    n_odom, n_scan, n_jpeg, n_gt = len(odom), len(scans), len(jpegs), len(gt)
    t0 = time.monotonic()
    graph.spin_for(1.0, every=lambda: cmd.publish(forward(0.2)), period=0.05)
    elapsed = time.monotonic() - t0
    moving = odom[-1][0]
    assert moving.twist.twist.linear.x == pytest.approx(0.2, abs=1e-6)
    assert moving.pose.pose.position.x - x_before == pytest.approx(0.2 * elapsed - 0.02, abs=0.05)
    assert front(scans[-1][0]) < range_before - 0.1
    assert truth(gt[-1])["bumper_to_box_m"] < report["bumper_to_box_m"] - 0.1

    # Rates: scan / camera / ground truth at 10 Hz of ROS time; odom once per 50 Hz timer tick,
    # which a loaded machine may deliver late (the physics then catches up in one tick).
    for log, n, hz in ((scans, n_scan, 10.0), (jpegs, n_jpeg, 10.0), (gt, n_gt, 10.0)):
        assert len(log) - n == pytest.approx(hz * elapsed, rel=0.3), (hz, len(log) - n)
    assert 0.6 * 50.0 * elapsed <= len(odom) - n_odom <= 1.3 * 50.0 * elapsed

    # Stamps are ROS time at publication; camera frames carry their frame id.
    now = graph.probe.get_clock().now().nanoseconds * 1e-9
    for log in (odom, scans, jpegs):
        stamps = [stamp_s(m) for m, _ in log[-5:]]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        assert 0.0 <= now - stamps[-1] < 0.5
    assert scans[-1][0].header.frame_id == "laser_frame" and len(scans[-1][0].ranges) == 360
    assert jpegs[-1][0].header.frame_id == "camera_link" and jpegs[-1][0].format == "jpeg"
    assert odom[-1][0].header.frame_id == "odom" and odom[-1][0].child_frame_id == "base_footprint"
    assert raws and (raws[-1][0].encoding, raws[-1][0].width, raws[-1][0].height) == ("bgr8", 640, 480)
    shared = {stamp_s(m) for m, _ in raws} & {stamp_s(m) for m, _ in jpegs}
    assert len(shared) >= 0.8 * min(len(raws), len(jpegs)), "raw and jpeg frames must share stamps"

    # No more commands: after cmd_timeout (0.5 s) the robot brakes to a stop.
    assert graph.spin_until(lambda: odom[-1][0].twist.twist.linear.x == 0.0, timeout=1.5)


def test_camera_hfov_parameter_sets_the_simulated_lens(graph):
    jpegs = graph.record(CompressedImage, "/camera/image_raw/compressed", qos_profile_sensor_data)
    overrides = [Parameter("initial_pose_id", value=1), Parameter("camera_hfov_rad", value=1.0856)]
    graph.add(FakeRobot(context=graph.context, parameter_overrides=overrides))
    assert graph.spin_until(lambda: len(jpegs) >= 2, timeout=5.0)
    frame = imgconv.compressed_to_bgr(jpegs[-1][0])
    # Pose 1: the marker is 15.3 deg to the right. An observer that assumes the simulated lens
    # reads that; one that assumes the default 80 deg lens reads it too far out.
    matched = MarkerObserver(hfov_rad=1.0856).observe(frame)
    assumed_wide = MarkerObserver(hfov_rad=1.396).observe(frame)
    assert matched is not None and matched.bearing_deg == pytest.approx(-15.3, abs=1.0)
    assert assumed_wide is not None and assumed_wide.bearing_deg == pytest.approx(-20.8, abs=1.0)


def test_ignores_unknown_poses_and_other_events(graph):
    gt = graph.record(String, GROUND_TRUTH_TOPIC)
    events = graph.probe.create_publisher(String, EVENT_TOPIC, event_qos())
    graph.add(FakeRobot(context=graph.context))
    assert graph.spin_until(lambda: len(gt) >= 2, timeout=5.0)
    home = truth(gt[-1])
    for text in ("start bc 42 1", "start bc", "end success", "intervention pushed it"):
        events.publish(String(data=text))
    n = len(gt)
    graph.spin_for(0.5)
    assert len(gt) > n and all(truth(entry) == home for entry in gt), "only a valid start may move it"

    events.publish(String(data="start act 5 3"))
    pose5 = start_pose_to_world(load_start_poses(default_start_poses_file())[5])
    assert graph.spin_until(lambda: truth(gt[-1])["x"] == pytest.approx(pose5.x, abs=1e-4), timeout=3.0)
    assert truth(gt[-1])["yaw"] == pytest.approx(pose5.yaw, abs=1e-4)


def test_late_fake_robot_gets_the_start_from_transient_local(graph):
    gt = graph.record(String, GROUND_TRUTH_TOPIC)
    events = graph.probe.create_publisher(String, EVENT_TOPIC, event_qos())
    events.publish(String(data="start bc 7 1"))
    graph.spin_until(lambda: False, timeout=0.2)
    graph.add(FakeRobot(context=graph.context, parameter_overrides=[Parameter("randomize", value=True)]))
    pose7 = start_pose_to_world(load_start_poses(default_start_poses_file())[7])
    near7 = lambda: gt and math.hypot(truth(gt[-1])["x"] - pose7.x, truth(gt[-1])["y"] - pose7.y) < 0.05  # noqa: E731
    assert graph.spin_until(near7, timeout=4.0)


def test_executable_rejects_bad_parameters():
    from ament_index_python.packages import get_package_prefix

    exe = os.path.join(get_package_prefix("jbl_nodes"), "lib", "jbl_nodes", "fake_robot")
    env = dict(os.environ, ROS_DOMAIN_ID=str(random.SystemRandom().randint(30, 99)))
    for args, message in (
        (["-p", "initial_pose_id:=10"], b"initial_pose_id 10 is not in"),
        (["-p", "start_poses_file:=/nonexistent/poses.yaml"], b"No such file"),
        (["-p", "box_x:=-1.0"], b"start poses [0"),
        (["-p", "camera_hfov_rad:=0.0"], b"camera_hfov_rad must be in"),
    ):
        proc = subprocess.run(
            [exe, "--ros-args"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, env=env
        )
        assert proc.returncode == 1, proc.stdout
        assert b"cannot start" in proc.stdout and message in proc.stdout and b"Traceback" not in proc.stdout
