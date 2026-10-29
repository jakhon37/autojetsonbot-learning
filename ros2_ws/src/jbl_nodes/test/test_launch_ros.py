"""The three launch files on a live graph (runs under colcon test; needs jbl_safety installed).

Each test runs ``ros2 launch`` in its own process group, stops it with SIGINT, and talks to it
from a probe node on a private context. A random ROS_DOMAIN_ID keeps parallel runs on the same
host (docker --network host) apart.
"""

import csv
import json
import os
import random
import signal
import subprocess
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from rcl_interfaces.msg import ParameterType  # noqa: E402
from rcl_interfaces.srv import GetParameters  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CompressedImage  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from jbl_nodes.episode_cli import EVENT_TOPIC, event_qos  # noqa: E402

SIM_NODES = {"fake_robot", "safety_gate", "dock_judge", "trial_logger", "servo_baseline"}
ROBOT_NODES = {"camera_node", "safety_gate", "dock_judge", "trial_logger"}


class Probe(object):
    def __init__(self, context):
        self.node = Node("launch_probe", context=context)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        self.events = self.record(String, EVENT_TOPIC, event_qos())

    def record(self, msg_type, topic, qos=10):
        log = []
        self.node.create_subscription(msg_type, topic, log.append, qos)
        return log

    def spin_until(self, predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=0.02)
        return predicate()

    def wait_for_nodes(self, names, timeout=40.0):
        return self.spin_until(lambda: set(names) <= set(self.node.get_node_names()), timeout)

    def close(self):
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()


class Launch(object):
    """``ros2 launch jbl_nodes <file> k:=v ...`` with output in ``log_path``."""

    def __init__(self, launch_file, log_path, **args):
        self.log_path = str(log_path)
        self._log = open(self.log_path, "w")
        cmd = ["ros2", "launch", "jbl_nodes", launch_file] + ["%s:=%s" % kv for kv in args.items()]
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self, timeout=30.0):
        """SIGINT to ros2 launch, which forwards it to every process it started."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self._log.close()
        return self.proc.returncode

    def output(self):
        if not self._log.closed:
            self._log.flush()
        with open(self.log_path) as f:
            return f.read()


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(random.SystemRandom().randint(30, 99)))
    context = Context()
    rclpy.init(context=context)
    p = Probe(context)
    try:
        yield p
    finally:
        p.close()
        context.try_shutdown()


@pytest.fixture
def launches():
    started = []
    yield lambda *a, **k: started.append(Launch(*a, **k)) or started[-1]
    for launch in started:
        launch.stop()


def _param(probe, node, name):
    """(type, value) of a parameter, asked over the node's own get_parameters service."""
    client = probe.node.create_client(GetParameters, "/%s/get_parameters" % node)
    assert probe.spin_until(client.service_is_ready, 10.0), "no parameter service on %s" % node
    future = client.call_async(GetParameters.Request(names=[name]))
    assert probe.spin_until(future.done, 10.0)
    value = future.result().values[0]
    by_type = {
        ParameterType.PARAMETER_BOOL: value.bool_value,
        ParameterType.PARAMETER_INTEGER: value.integer_value,
        ParameterType.PARAMETER_DOUBLE: value.double_value,
        ParameterType.PARAMETER_STRING: value.string_value,
    }
    return value.type, by_type.get(value.type)


def _bag_counts(bag):
    out = subprocess.run(
        ["ros2", "bag", "info", str(bag)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
    ).stdout.decode()
    counts = {}
    for line in out.splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) >= 3 and "Topic:" in fields[0] and fields[2].startswith("Count:"):
            counts[fields[0].split()[-1]] = int(fields[2].split()[-1])
    return counts, out


def test_sim_dock_servo_episode_end_to_end(probe, launches, tmp_path):
    trials, bag = tmp_path / "trials.csv", tmp_path / "bag"
    launch = launches("sim_dock.launch.py", tmp_path / "launch.log", trials_csv=trials, bag=bag)
    assert probe.wait_for_nodes(SIM_NODES), launch.output()
    double = ParameterType.PARAMETER_DOUBLE
    # config/sim.yaml: the v4 stop rules (a still request is |v| <= 0.005 m/s; the servo's stop latch)
    assert _param(probe, "dock_judge", "request_still_v") == (double, 0.005)
    assert _param(probe, "servo_baseline", "stop_hysteresis") == (double, 0.03)
    time.sleep(2.0)  # the recorder subscribes to new topics on its next poll

    # The real CLI: probe, fake_robot, dock_judge, servo_baseline, trial_logger and the recorder.
    # The probe keeps spinning meanwhile: Foxy's DDS drops a sample nobody has taken yet when
    # its writer goes away, which the CLI's does when it exits.
    cli = subprocess.Popen(
        [
            "ros2",
            "run",
            "jbl_nodes",
            "episode",
            "--timeout",
            "10",
            "--expect",
            "6",
            "start",
            "servo",
            "0",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert probe.spin_until(lambda: cli.poll() is not None, 60.0), "episode CLI did not exit"
    assert cli.returncode == 0, cli.communicate()[0].decode()
    assert probe.spin_until(lambda: any(m.data.startswith("end ") for m in probe.events), 40.0), (
        launch.output()
    )
    assert [m.data for m in probe.events] == ["start servo 0 1", "end success"]
    assert probe.spin_until(lambda: trials.exists() and len(trials.read_text().splitlines()) == 2, 5.0)

    assert launch.stop() == 0, launch.output()
    with open(str(trials)) as f:
        (row,) = list(csv.DictReader(f))
    assert (row["method"], row["pose_id"], row["round"], row["outcome"]) == ("servo", "0", "1", "success")
    assert 0.18 <= float(row["final_range_m"]) <= 0.32 and abs(float(row["final_bearing_deg"])) <= 10.0

    assert (bag / "metadata.yaml").exists(), "the recorder was not shut down cleanly"
    counts, info = _bag_counts(bag)
    assert counts.get(EVENT_TOPIC) == 2, info
    for topic in (
        "/camera/image_raw/compressed",
        "/odom",
        "/scan",
        "/cmd_vel_in",
        "/cmd_vel",
        "/dock_judge/status",
        "/safety_gate/status",
        "/fake_robot/ground_truth",
    ):
        assert counts.get(topic, 0) > 0, "%s not recorded:\n%s" % (topic, info)


def test_robot_policy_constant_backend_drives_through_the_gate(probe, launches, tmp_path):
    extra = tmp_path / "extra.yaml"
    extra.write_text("policy_runner:\n  ros__parameters:\n    constant_action: [0.4, 0.5]\n")
    cmd_vel = probe.record(Twist, "/cmd_vel")
    frames = probe.record(CompressedImage, "/camera/image_raw/compressed")
    launch = launches(
        "robot_policy.launch.py",
        tmp_path / "launch.log",
        camera_source="test_pattern",
        backend="constant",
        extra_params_file=extra,
        trials_csv=tmp_path / "trials.csv",
    )
    assert probe.wait_for_nodes(ROBOT_NODES | {"policy_runner"}), launch.output()
    assert probe.spin_until(lambda: len(frames) >= 5, 10.0), "no frames from the test pattern"
    # config/robot.yaml: the Pi Camera v2 lens and its Argus pipeline (a string on every distro).
    assert _param(probe, "dock_judge", "hfov_rad") == (ParameterType.PARAMETER_DOUBLE, 1.0856)
    device_type, device = _param(probe, "camera_node", "device")
    assert device_type == ParameterType.PARAMETER_STRING and device.startswith("nvarguscamerasrc")

    events = probe.node.create_publisher(String, EVENT_TOPIC, event_qos())
    events.publish(String(data="start bc 0 1"))
    # No /scan on this bench: the gate blocks forward motion and passes the rotation (1.5 * 0.5).
    turning = lambda: cmd_vel and cmd_vel[-1].angular.z == pytest.approx(0.75) and cmd_vel[-1].linear.x == 0.0  # noqa: E731
    assert probe.spin_until(turning, 10.0), launch.output()
    events.publish(String(data="end abort bench test"))
    assert probe.spin_until(lambda: cmd_vel[-1].angular.z == 0.0, 2.0)
    assert launch.stop() == 0, launch.output()


def test_robot_record_shares_calibration_with_every_node(probe, launches, tmp_path):
    launch = launches(
        "robot_record.launch.py",
        tmp_path / "launch.log",
        camera_source="test_pattern",
        with_servo="true",
        forward_angle="3",  # an int on purpose: it must still reach the nodes as a double
        hfov_rad="1.085",
        trials_csv=tmp_path / "trials.csv",
    )
    assert probe.wait_for_nodes(ROBOT_NODES | {"servo_baseline"}), launch.output()
    double = ParameterType.PARAMETER_DOUBLE
    for node in ("safety_gate", "dock_judge", "servo_baseline"):
        assert _param(probe, node, "forward_angle") == (double, 3.0), node
    for node in ("dock_judge", "servo_baseline"):
        assert _param(probe, node, "hfov_rad") == (double, 1.085), node
    assert _param(probe, "safety_gate", "sector_half_angle") == (double, 0.35)  # jbl_safety's defaults
    assert _param(probe, "camera_node", "compressed_fps") == (double, 15.0)  # config/robot.yaml
    assert _param(probe, "dock_judge", "request_still_v") == (double, 0.005)
    assert _param(probe, "servo_baseline", "stop_hysteresis") == (double, 0.03)
    assert _param(probe, "camera_node", "source")[1] == "test_pattern"  # argument beats the node section
    status = probe.record(String, "/dock_judge/status")
    assert probe.spin_until(lambda: status, 5.0)
    assert json.loads(status[-1].data)["marker_visible"] is False
    assert launch.stop() == 0, launch.output()


@pytest.mark.parametrize(
    "launch_file, args, message",
    [
        ("robot_policy.launch.py", {"camera_source": "test_pattern"}, "model_path"),
        ("robot_policy.launch.py", {"method": "servo", "backend": "constant"}, "with_servo"),
        ("sim_dock.launch.py", {"method": "two words"}, "method"),
        ("sim_dock.launch.py", {"controller": "joystick"}, "controller"),
        ("sim_dock.launch.py", {"forward_angle": "ahead"}, "forward_angle"),
    ],
)
def test_bad_arguments_stop_the_launch(probe, launches, tmp_path, launch_file, args, message):
    launch = launches(launch_file, tmp_path / "launch.log", trials_csv=tmp_path / "t.csv", **args)
    try:
        launch.proc.wait(30.0)
    except subprocess.TimeoutExpired:
        pytest.fail("launch kept running:\n" + launch.output())
    assert launch.proc.returncode != 0
    assert message in launch.output()
    assert not set(probe.node.get_node_names()) & (SIM_NODES | ROBOT_NODES), "nodes were started"
