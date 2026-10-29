"""dock_judge, servo_baseline and trial_logger on a live ROS graph (runs under colcon test).

A small kinematic world stands in for the robot: the box face is the plane x = BOX_X, the
camera frame is rendered with the pinhole model, the lidar is ray-cast against the face, and
odometry reports the commanded twist. Everything runs in-process on a private context and a
random ROS_DOMAIN_ID, so parallel test runs on one host do not see each other's topics.
"""

import array
import csv
import json
import math
import random
import time

import numpy as np
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
from jbl_nodes.aruco_utils import render_marker_view  # noqa: E402
from jbl_nodes.dock_judge import DockJudgeNode  # noqa: E402
from jbl_nodes.episode_cli import EVENT_TOPIC, event_qos  # noqa: E402
from jbl_nodes.judge_logic import STATUS_KEYS, STATUS_TOPIC, ServoParams  # noqa: E402
from jbl_nodes.servo_baseline import ServoBaseline  # noqa: E402
from jbl_nodes.trial_logger import CSV_COLUMNS, TrialLogger  # noqa: E402

COMPRESSED_TOPIC = "/camera/image_raw/compressed"
RAW_TEST_TOPIC = "/test/image_raw"
N_BEAMS = 360
BEAM_INC = 2.0 * math.pi / N_BEAMS
SERVO = ServoParams()


class World(object):
    """Unicycle robot facing a box; heights are relative to the camera (marker at camera height)."""

    BOX_X = 1.0
    FACE_HALF_WIDTH = 0.15
    MARKER_SIDE = 0.10
    LIDAR_X = 0.064  # ahead of base_link
    CAMERA_X = 0.097
    LIDAR_MIN = 0.15

    def __init__(self, x, y, th):
        self.x, self.y, self.th = x, y, th

    @classmethod
    def facing_box(cls, lidar_range, yaw_deg=0.0):
        """On the box's centre line, lidar `lidar_range` from the face, turned by yaw_deg (left +)."""
        th = math.radians(yaw_deg)
        return cls(cls.BOX_X - lidar_range - cls.LIDAR_X * math.cos(th), -cls.LIDAR_X * math.sin(th), th)

    def step(self, v, w, dt):
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th += w * dt

    def _camera_frame(self, points):
        c, s = math.cos(self.th), math.sin(self.th)
        dx = points[:, 0] - (self.x + self.CAMERA_X * c)
        dy = points[:, 1] - (self.y + self.CAMERA_X * s)
        return np.stack([dx * s - dy * c, -points[:, 2], dx * c + dy * s], axis=1)

    def marker_corners_camera(self):
        h = 0.5 * self.MARKER_SIDE
        # top-left, top-right, bottom-right, bottom-left as seen from in front (+y is left)
        world = np.array([[self.BOX_X, h, h], [self.BOX_X, -h, h], [self.BOX_X, -h, -h], [self.BOX_X, h, -h]])
        return self._camera_frame(world)

    def bearing_deg(self):
        x, _, z = self._camera_frame(np.array([[self.BOX_X, 0.0, 0.0]]))[0]
        return math.degrees(math.atan2(-x, z))

    def camera_view(self, size):
        return render_marker_view(7, self.marker_corners_camera(), size)

    def ranges(self):
        c, s = math.cos(self.th), math.sin(self.th)
        lx, ly = self.x + self.LIDAR_X * c, self.y + self.LIDAR_X * s
        a = self.th + (-math.pi + BEAM_INC * np.arange(N_BEAMS))
        ca, sa = np.cos(a), np.sin(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.BOX_X - lx) / ca
            hit = (ca > 1e-6) & (np.abs(ly + t * sa) <= self.FACE_HALF_WIDTH)
        r = np.where(hit, t, np.inf)
        r[hit & (r < self.LIDAR_MIN)] = 0.0  # the RPLidar reports its dead zone as 0
        return r


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

    def spin_for(self, seconds, every=None, period=0.05):
        self.spin_until(lambda: False, seconds, every, period)

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


class Rig(object):
    """Publishes what the sensors see in a World; optionally moves it by the latest /cmd_vel_in."""

    def __init__(self, graph, world, image_topic=COMPRESSED_TOPIC, image_size=(640, 480)):
        node = graph.probe
        self.world = world
        self.image_size = image_size
        self.marker_visible = True
        self.camera_on = True
        self.lidar_on = True
        self.lidar_dead_zone = False  # every forward reading invalid, as when too close
        self.gate_state = "ok"
        self.move = False
        self._compressed = image_topic.endswith("/compressed")
        self._image_pub = node.create_publisher(
            CompressedImage if self._compressed else Image, image_topic, 5
        )
        self._scan_pub = node.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self._odom_pub = node.create_publisher(Odometry, "/odom", 10)
        self._gate_pub = node.create_publisher(String, "/safety_gate/status", 10)
        self.events = node.create_publisher(String, EVENT_TOPIC, event_qos())
        self.cmds = graph.record(Twist, "/cmd_vel_in")
        self._clock = node.get_clock()
        self._last = time.monotonic()

    def event(self, text):
        self.events.publish(String(data=text))

    def command(self):
        if not self.move or not self.cmds:
            return 0.0, 0.0
        msg = self.cmds[-1][0]
        return msg.linear.x, msg.angular.z

    def feed(self):
        now = time.monotonic()
        dt, self._last = now - self._last, now
        v, w = self.command()
        self.world.step(v, w, dt)
        stamp = self._clock.now().to_msg()

        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = stamp, "laser_frame"
        scan.angle_min, scan.angle_increment = -math.pi, BEAM_INC
        scan.angle_max = -math.pi + BEAM_INC * (N_BEAMS - 1)
        scan.range_min, scan.range_max = 0.15, 12.0
        ranges = self.world.ranges()
        if self.lidar_dead_zone:
            ranges[np.isfinite(ranges)] = 0.0
        scan.ranges = array.array("f", ranges.astype(np.float32).tobytes())
        if self.lidar_on:
            self._scan_pub.publish(scan)

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "odom"
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = float(v), float(w)
        self._odom_pub.publish(odom)
        self._gate_pub.publish(
            String(data=json.dumps({"state": self.gate_state, "front_min_m": None, "scale": 1.0}))
        )

        if not self.camera_on:
            return
        if self.marker_visible:
            frame = self.world.camera_view(self.image_size)
        else:
            frame = np.full((self.image_size[1], self.image_size[0], 3), 90, dtype=np.uint8)
        msg = (
            imgconv.numpy_to_compressed(frame, "jpeg", 85)
            if self._compressed
            else imgconv.numpy_to_image(frame)
        )
        msg.header.stamp, msg.header.frame_id = stamp, "camera_link"
        self._image_pub.publish(msg)


def statuses(log):
    return [json.loads(m.data) for m, _ in log]


def ends(log):
    return [(m.data, t) for m, t in log if m.data.startswith("end")]


def read_rows(path):
    with open(str(path), newline="") as f:
        header = f.readline().strip()
        return header, list(csv.DictReader(f, fieldnames=list(CSV_COLUMNS)))


def judge(graph, **params):
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    return graph.add(DockJudgeNode(context=graph.context, parameter_overrides=overrides))


def logger(graph, path):
    return graph.add(
        TrialLogger(context=graph.context, parameter_overrides=[Parameter("path", value=str(path))])
    )


def servo(graph, **params):
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    return graph.add(ServoBaseline(context=graph.context, parameter_overrides=overrides))


def vw(entry):
    return entry[0].linear.x, entry[0].angular.z


# ---------------------------------------------------------------- dock_judge + trial_logger


def test_judge_declares_success_after_the_hold_and_logger_writes_the_row(graph, tmp_path):
    csv_path = tmp_path / "trials" / "trials.csv"  # the logger creates the directory
    events = graph.record(String, EVENT_TOPIC, event_qos())
    status = graph.record(String, STATUS_TOPIC)
    rig = Rig(graph, World.facing_box(0.25, yaw_deg=2.0))
    judge(graph)
    logger(graph, csv_path)

    def docked_and_seen():
        s = statuses(status[-1:])
        return bool(s) and s[0]["marker_visible"] and s[0]["in_band"] and s[0]["stopped"]

    assert graph.spin_until(docked_and_seen, timeout=8.0, every=rig.feed), statuses(status[-1:])
    last = statuses(status[-1:])[0]
    assert tuple(last) == STATUS_KEYS
    assert last["bearing_deg"] == pytest.approx(rig.world.bearing_deg(), abs=1.5)
    assert last["front_range_m"] == pytest.approx(0.25, abs=0.005)
    assert last["marker_id"] == 7 and last["aligned"]
    assert ends(events) == [], "no episode, no verdict"

    graph.spin_for(0.3, every=rig.feed)  # the logger's subscriptions are matched by now too
    t_start = time.monotonic()
    rig.event("start servo 3 2")
    assert graph.spin_until(lambda: ends(events), timeout=5.0, every=rig.feed)
    ((text, t_end),) = ends(events)
    assert text == "end success"
    assert 0.95 <= t_end - t_start <= 2.0, "hold_s = 1.0 at 10 Hz"

    graph.spin_for(1.5, every=rig.feed)
    # The end can overtake the status of the deciding tick (different writers), so look later.
    decided = [s for s in statuses([e for e in status if e[1] > t_start]) if s["success"]]
    assert decided and decided[0]["success_hold_s"] == pytest.approx(1.0, abs=0.11)
    assert len(ends(events)) == 1, "exactly one end per episode"
    rate = len([1 for _, t in status if t > t_end])
    assert 11 <= rate <= 19, "status should be ~10 Hz, got %d in 1.5 s" % rate

    assert graph.spin_until(csv_path.exists, timeout=2.0)
    header, rows = read_rows(csv_path)
    assert header == ",".join(CSV_COLUMNS)
    assert len(rows) == 1
    row = rows[0]
    assert (row["method"], row["pose_id"], row["round"], row["outcome"]) == ("servo", "3", "2", "success")
    assert 0.95 <= float(row["duration_s"]) <= 2.0
    assert float(row["final_range_m"]) == pytest.approx(0.25, abs=0.005)
    assert float(row["final_bearing_deg"]) == pytest.approx(rig.world.bearing_deg(), abs=1.5)
    assert (row["interventions"], row["notes"]) == ("0", "")


def test_judge_fail_reasons_and_logged_interventions(graph, tmp_path):
    csv_path = tmp_path / "trials.csv"
    events = graph.record(String, EVENT_TOPIC, event_qos())
    status = graph.record(String, STATUS_TOPIC)
    rig = Rig(graph, World.facing_box(0.6))
    judge(graph, timeout_s=1.5, obstacle_fail_s=0.5)
    logger(graph, csv_path)
    assert graph.spin_until(lambda: status and statuses(status[-1:])[0]["marker_visible"], 8.0, rig.feed)
    graph.spin_for(0.3, every=rig.feed)

    def run_episode(start_text, *extra_events):
        n = len(ends(events))
        t0 = time.monotonic()
        rig.event(start_text)
        for text in extra_events:
            graph.spin_for(0.2, every=rig.feed)
            rig.event(text)
        assert graph.spin_until(lambda: len(ends(events)) > n, timeout=5.0, every=rig.feed)
        text, t = ends(events)[n]
        return text, t - t0

    text, elapsed = run_episode("start bc 1 1")  # 0.6 m away: never in band
    assert text == "end fail timeout" and 1.45 <= elapsed <= 2.2

    text, elapsed = run_episode("start act 2 1", "intervention", "intervention pushed by hand")
    assert text == "end fail intervention" and elapsed < 0.6

    rig.marker_visible = False
    rig.gate_state = "stop_obstacle"
    text, elapsed = run_episode("start bc 3 1")
    assert text == "end fail stop_obstacle" and 0.5 <= elapsed <= 1.2

    # An operator end overrides nothing once the judge decided, and an operator abort of an
    # undecided episode is logged as such without a judge verdict.
    rig.event("end abort")
    graph.spin_for(0.3, every=rig.feed)
    rig.gate_state = "ok"
    rig.event("start servo 4 1")
    graph.spin_for(0.4, every=rig.feed)
    rig.event("end abort operator stop")
    graph.spin_for(2.0, every=rig.feed)
    assert [text for text, _ in ends(events)][-2:] == ["end abort", "end abort operator stop"]
    assert len(ends(events)) == 5, "no judge verdict after an external end"

    _, rows = read_rows(csv_path)
    summary = [(r["method"], r["pose_id"], r["outcome"], r["interventions"], r["notes"]) for r in rows]
    assert summary == [
        ("bc", "1", "fail", "0", "timeout"),
        ("act", "2", "fail", "1", "intervention"),
        ("bc", "3", "fail", "0", "stop_obstacle"),
        ("servo", "4", "abort", "0", "operator stop"),
    ]
    assert float(rows[0]["final_range_m"]) == pytest.approx(0.6, abs=0.01)
    assert rows[2]["final_bearing_deg"] == "", "marker not visible at the end"


def test_logger_marks_an_episode_without_an_end_and_missing_judge(graph, tmp_path):
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text(",".join(CSV_COLUMNS) + "\n")  # existing file: no second header
    rig = Rig(graph, World.facing_box(0.6))
    logger(graph, csv_path)
    graph.spin_for(0.5)
    for text in ("end success", "start bc 5 3", "intervention", "start bc 6 3", "end success"):
        rig.event(text)
        graph.spin_for(0.3)
    assert graph.spin_until(lambda: len(read_rows(csv_path)[1]) == 2, timeout=3.0)
    header, rows = read_rows(csv_path)
    assert header == ",".join(CSV_COLUMNS)
    assert [(r["pose_id"], r["outcome"], r["interventions"]) for r in rows] == [
        ("5", "abort", "1"),
        ("6", "success", "0"),
    ]
    assert rows[0]["notes"] == "no end before next start; no dock_judge status"
    assert rows[1]["final_range_m"] == "" and rows[1]["notes"] == "no dock_judge status"


def test_logger_counts_cli_interventions_and_keeps_their_notes(graph, tmp_path):
    csv_path = tmp_path / "trials.csv"
    rig = Rig(graph, World.facing_box(0.6))
    logger(graph, csv_path)
    graph.spin_for(0.5)
    # "episode intervention <note>" during the trial, then the judge's (or the operator's) end.
    for text in (
        "start bc 7 2",
        "intervention lifted it before the wall",
        "intervention",
        "end fail intervention",
        "start servo 8 2",
        "end fail intervention cable snagged",
    ):
        rig.event(text)
        graph.spin_for(0.3)
    assert graph.spin_until(lambda: csv_path.exists() and len(read_rows(csv_path)[1]) == 2, timeout=3.0)
    _, rows = read_rows(csv_path)
    summary = [(r["pose_id"], r["outcome"], r["interventions"]) for r in rows]
    assert summary == [("7", "fail", "2"), ("8", "fail", "1")]
    assert rows[0]["notes"] == "intervention; lifted it before the wall; no dock_judge status"
    assert rows[1]["notes"] == "intervention cable snagged; no dock_judge status"


@pytest.mark.parametrize("params, still", [({}, False), ({"request_still_v": 0.02}, True)])
def test_judge_counts_a_creeping_request_as_still_only_up_to_request_still_v(graph, params, still):
    """The robot is still (odometry 0), but the controller keeps asking for 0.01 m/s: not still
    for the default judge (request_still_v 0.005); request_still_v:=0.02 is the v2/v3 rule."""
    status = graph.record(String, STATUS_TOPIC)
    rig = Rig(graph, World.facing_box(0.25))
    judge(graph, **params)
    creep = graph.probe.create_publisher(Twist, "/cmd_vel_in", 10)

    def feed():
        rig.feed()
        msg = Twist()
        msg.linear.x = 0.01
        creep.publish(msg)

    def docked():
        s = statuses(status[-1:])
        return bool(s) and s[0]["marker_visible"] and s[0]["in_band"] and s[0]["aligned"]

    assert graph.spin_until(docked, timeout=8.0, every=feed), statuses(status[-1:])
    graph.spin_for(1.5, every=feed)
    last = statuses(status[-5:])
    assert all(s["stopped"] == still and s["speed_v"] == 0.0 for s in last), last
    assert last[-1]["success"] == still


# ---------------------------------------------------------------- servo_baseline


def test_servo_runs_only_in_its_episodes_and_follows_the_law(graph):
    # Raw images at a small size: the law depends on bearing, not on resolution.
    rig = Rig(graph, World.facing_box(0.8, yaw_deg=-20.0), image_topic=RAW_TEST_TOPIC, image_size=(320, 240))
    servo(graph, image_topic=RAW_TEST_TOPIC)
    graph.spin_for(1.0, every=rig.feed)
    assert rig.cmds == [], "silent until a servo episode starts"
    rig.event("start bc 1 1")
    graph.spin_for(0.5, every=rig.feed)
    assert rig.cmds == [], "another method's episode"

    rig.event("start servo 1 1")
    assert graph.spin_until(lambda: len(rig.cmds) >= 5, timeout=3.0, every=rig.feed)
    n0 = len(rig.cmds)
    graph.spin_for(1.0, every=rig.feed)
    assert 15 <= len(rig.cmds) - n0 <= 25, "~20 Hz"
    v, w = vw(rig.cmds[-1])
    assert rig.world.bearing_deg() > 15.0  # marker left of centre
    assert w > 0.0 and 0.0 < v < SERVO.v_max, "turn left towards it, slowly forward"

    rig.world = World.facing_box(0.8, yaw_deg=25.0)  # marker right of centre, beyond slow_bearing_deg
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1])[1] < 0.0 and vw(rig.cmds[-1])[0] == 0.0

    rig.marker_visible = False
    graph.spin_for(0.15, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, 0.0), "a brief loss holds still"
    graph.spin_for(0.5, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, -SERVO.search_w), "search towards where it was last seen"

    rig.marker_visible = True
    rig.world = World.facing_box(0.8)
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1]) == pytest.approx((SERVO.v_max, 0.0))

    rig.lidar_dead_zone = True  # no valid lidar reading: fall back on the marker size
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1]) == pytest.approx((SERVO.v_max, 0.0), abs=1e-6), "far by marker size"
    rig.world = World.facing_box(0.14)
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, 0.0), "close by marker size: stop"
    rig.lidar_dead_zone = False

    rig.world = World.facing_box(0.26, yaw_deg=2.0)  # in band and aligned
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, 0.0)

    rig.camera_on = False
    graph.spin_for(0.8, every=rig.feed)
    rig.world = World.facing_box(0.8)
    graph.spin_for(0.2, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, 0.0), "camera silent: hold still"
    rig.camera_on = True
    assert graph.spin_until(lambda: vw(rig.cmds[-1])[0] > 0.0, timeout=2.0, every=rig.feed)

    rig.event("end success")
    assert graph.spin_until(lambda: vw(rig.cmds[-1]) == (0.0, 0.0), timeout=2.0, every=rig.feed)
    n = len(rig.cmds)
    graph.spin_for(0.5, every=rig.feed)
    assert len(rig.cmds) == n, "silent after the episode"


@pytest.mark.parametrize("hysteresis", [None, 0.0])
def test_servo_stop_latch_and_its_reset_at_each_start(graph, hysteresis):
    """Stopped at 0.26 m, the servo (stop_hysteresis 0.03 by default) stays stopped, v exactly 0,
    when pushed back to 0.29 m and drives again above 0.30 m; stop_hysteresis 0 is the v2/v3
    law, which creeps at 0.29 m. Each episode starts unlatched, and a scan from before the
    start (the previous pose) does not latch it again."""
    params = {} if hysteresis is None else {"stop_hysteresis": hysteresis}
    rig = Rig(graph, World.facing_box(0.26), image_topic=RAW_TEST_TOPIC, image_size=(320, 240))
    servo(graph, image_topic=RAW_TEST_TOPIC, **params)
    graph.spin_for(0.5, every=rig.feed)
    rig.event("start servo 1 1")
    assert graph.spin_until(lambda: len(rig.cmds) >= 5, timeout=3.0, every=rig.feed)
    graph.spin_for(0.3, every=rig.feed)
    assert vw(rig.cmds[-1]) == (0.0, 0.0), "stopped at 0.26 m"

    creep = SERVO.k_v * (0.29 - SERVO.target_range)
    rig.world = World.facing_box(0.29)  # pushed back into the hysteresis band
    n = len(rig.cmds)
    graph.spin_for(0.6, every=rig.feed)
    if hysteresis is None:
        assert len(rig.cmds) - n >= 8 and all(vw(c) == (0.0, 0.0) for c in rig.cmds[n:]), "latched"
    else:
        assert vw(rig.cmds[-1])[0] == pytest.approx(creep, rel=0.1), "v2/v3 law: creeps on"
    rig.world = World.facing_box(0.31)  # beyond the release range
    beyond = SERVO.k_v * (0.31 - SERVO.target_range)
    assert graph.spin_until(lambda: vw(rig.cmds[-1])[0] == pytest.approx(beyond, rel=0.1), 2.0, rig.feed)

    # Stopped at 0.26 m again; the next episode starts at 0.29 m. Until the lidar's next scan the
    # servo still has the old pose's scan (0.26 m, received before the start): it must not latch.
    rig.world = World.facing_box(0.26)
    assert graph.spin_until(lambda: vw(rig.cmds[-1]) == (0.0, 0.0), timeout=2.0, every=rig.feed)
    graph.spin_for(0.3, every=rig.feed)
    rig.lidar_on = False
    rig.world = World.facing_box(0.29)
    graph.spin_for(0.1)  # what was sent arrives before the start
    rig.event("start servo 2 1")
    graph.spin_for(0.25, every=rig.feed)  # frames of the new pose, the old scan (< scan_timeout_s)
    rig.lidar_on = True
    assert graph.spin_until(lambda: vw(rig.cmds[-1])[0] > 0.0, timeout=2.0, every=rig.feed)
    assert vw(rig.cmds[-1])[0] == pytest.approx(creep, rel=0.1), "a new episode starts unlatched"


def test_servo_stop_latch_uses_the_marker_size_when_the_lidar_has_no_reading(graph):
    """No valid lidar reading: the latch gets the range the law falls back on, the marker-size
    estimate (about 0.263, 0.294 and 0.336 m at 0.26, 0.29 and 0.33 m in a 320x240 frame). On at
    0.26 m, held at 0.29 m, off at 0.33 m; a frame received before a start (the previous pose)
    does not latch the new episode."""
    rig = Rig(graph, World.facing_box(0.33), image_topic=RAW_TEST_TOPIC, image_size=(320, 240))
    rig.lidar_dead_zone = True
    servo(graph, image_topic=RAW_TEST_TOPIC)
    graph.spin_for(0.3, every=rig.feed)
    rig.event("start servo 1 1")
    assert graph.spin_until(lambda: rig.cmds and vw(rig.cmds[-1])[0] > 0.05, timeout=3.0, every=rig.feed)
    rig.world = World.facing_box(0.26)
    assert graph.spin_until(lambda: vw(rig.cmds[-1]) == (0.0, 0.0), timeout=2.0, every=rig.feed)
    rig.world = World.facing_box(0.29)
    n = len(rig.cmds)
    graph.spin_for(0.6, every=rig.feed)
    assert len(rig.cmds) - n >= 8 and all(vw(c) == (0.0, 0.0) for c in rig.cmds[n:]), "latched by the marker size"
    rig.world = World.facing_box(0.33)
    assert graph.spin_until(lambda: vw(rig.cmds[-1])[0] > 0.05, timeout=2.0, every=rig.feed), "released"

    # Latched at 0.26 m, then the episode ends: the servo stops decoding, so the newest 0.26 m frame
    # is still pending when the next episode starts at 0.29 m.
    rig.world = World.facing_box(0.26)
    assert graph.spin_until(lambda: vw(rig.cmds[-1]) == (0.0, 0.0), timeout=2.0, every=rig.feed)
    rig.event("end success")
    graph.spin_for(0.3, every=rig.feed)
    rig.world = World.facing_box(0.29)
    graph.spin_for(0.05)  # what was sent arrives before the start
    rig.event("start servo 2 1")
    graph.spin_for(0.15)  # no new frame: the first ticks decode the pending 0.26 m frame
    assert graph.spin_until(lambda: vw(rig.cmds[-1])[0] > 0.0, timeout=2.0, every=rig.feed)
    creep = SERVO.k_v * (0.294 - SERVO.target_range)
    assert vw(rig.cmds[-1])[0] == pytest.approx(creep, abs=0.01), "a new episode starts unlatched"


def test_servo_always_on(graph):
    rig = Rig(graph, World.facing_box(0.8))
    servo(graph, always_on=True, rate_hz=10.0)
    assert graph.spin_until(lambda: len(rig.cmds) >= 5, timeout=4.0, every=rig.feed)
    assert vw(rig.cmds[-1]) == pytest.approx((SERVO.v_max, 0.0))
    rig.event("end fail timeout")
    graph.spin_for(0.5, every=rig.feed)
    assert vw(rig.cmds[-1]) == pytest.approx((SERVO.v_max, 0.0)), "always_on ignores episodes"


def test_servo_docks_in_closed_loop_and_the_trial_is_logged(graph, tmp_path):
    csv_path = tmp_path / "trials.csv"
    events = graph.record(String, EVENT_TOPIC, event_qos())
    status = graph.record(String, STATUS_TOPIC)
    # 0.75 m from the face, 12 cm off the centre line, heading 14 deg left: marker ~23 deg right.
    rig = Rig(graph, World(0.15, 0.12, 0.25))
    rig.move = True
    judge(graph)
    servo(graph)
    logger(graph, csv_path)
    assert graph.spin_until(lambda: status and statuses(status[-1:])[0]["marker_visible"], 8.0, rig.feed)
    assert statuses(status[-1:])[0]["bearing_deg"] < -15.0
    graph.spin_for(0.3, every=rig.feed)

    rig.event("start servo 7 1")
    assert graph.spin_until(lambda: ends(events), timeout=25.0, every=rig.feed), statuses(status[-1:])
    assert [text for text, _ in ends(events)] == ["end success"]
    final = statuses(status[-1:])[0]
    assert 0.18 <= final["front_range_m"] <= 0.32 and abs(final["bearing_deg"]) <= 10.0
    assert abs(rig.world.bearing_deg()) <= 10.0

    assert graph.spin_until(lambda: csv_path.exists() and read_rows(csv_path)[1], timeout=2.0)
    row = read_rows(csv_path)[1][0]
    assert (row["method"], row["pose_id"], row["round"], row["outcome"]) == ("servo", "7", "1", "success")
    assert 2.0 <= float(row["duration_s"]) <= 25.0
    assert 0.18 <= float(row["final_range_m"]) <= 0.32
    assert abs(float(row["final_bearing_deg"])) <= 10.0
