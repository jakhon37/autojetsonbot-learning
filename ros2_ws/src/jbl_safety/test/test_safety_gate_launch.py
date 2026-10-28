# Copyright 2026 Jakhongir Nodirov
# SPDX-License-Identifier: Apache-2.0
"""Run safety_gate_node with its installed config and drive it with fake /scan and /cmd_vel_in."""

import contextlib
import json
import math
import os
import signal
import subprocess
import time
import unittest

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
import domain_coordinator
from geometry_msgs.msg import Twist
import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.asserts
import pytest
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# add_launch_test has no ROS isolation, so do what ament_cmake_ros' isolated test runner does:
# claim a domain ID before anything starts, so that other ROS tests on the same host network
# cannot publish into /scan or /cmd_vel_in here. The launched node and the probe inherit it.
_ISOLATION = contextlib.ExitStack()  # holds the claim for the life of the test process
if 'ROS_DOMAIN_ID' not in os.environ and 'DISABLE_ROS_ISOLATION' not in os.environ:
    if hasattr(domain_coordinator, 'domain_id'):  # Humble and later
        _DOMAIN_ID = _ISOLATION.enter_context(domain_coordinator.domain_id())
    else:  # Foxy: the claim lasts as long as the returned object
        _DOMAIN_ID = domain_coordinator.get_coordinated_domain_id()
    os.environ['ROS_DOMAIN_ID'] = str(_DOMAIN_ID)

GATE_NODE = '/safety_gate'
INTERFACE_DEFAULTS = {
    'rate_hz': 50.0,
    'forward_angle': 0.0,
    'sector_half_angle': 0.35,
    'stop_distance': 0.17,
    'slow_distance': 0.45,
    'treat_invalid_as_close': True,
    'cmd_timeout': 0.3,
    'scan_timeout': 0.5,
    'max_linear': 0.25,
    'max_angular': 1.5,
}
STATUS_KEYS = {
    'state', 'front_min_m', 'scale', 'period_ms', 'missed_deadlines', 'ticks', 'cb_latency_us',
}
STATES = {'ok', 'slow', 'stop_obstacle', 'stop_cmd_timeout', 'stop_scan_timeout'}
DISCOVERY_TIMEOUT = 20.0


@pytest.mark.launch_test
def generate_test_description():
    config = os.path.join(get_package_share_directory('jbl_safety'), 'config', 'safety_gate.yaml')
    gate = launch_ros.actions.Node(
        package='jbl_safety',
        executable='safety_gate_node',
        name='safety_gate',
        parameters=[config],
        output='screen',
    )
    return launch.LaunchDescription([gate, launch_testing.actions.ReadyToTest()]), {'gate': gate}


def make_scan(front_range, range_min):
    """Build a 360-beam RPLidar A1 style scan: `front_range` within +-0.35 rad, 2 m elsewhere."""
    scan = LaserScan()
    scan.header.frame_id = 'laser_frame'
    scan.angle_min = -math.pi
    scan.angle_increment = 2.0 * math.pi / 360
    scan.angle_max = scan.angle_min + 359 * scan.angle_increment
    scan.range_min = range_min
    scan.range_max = 12.0
    ranges = []
    for i in range(360):
        angle = scan.angle_min + i * scan.angle_increment
        ranges.append(front_range if abs(angle) <= 0.35 else 2.0)
    scan.ranges = ranges
    return scan


def gate_executable():
    return os.path.join(get_package_prefix('jbl_safety'), 'lib', 'jbl_safety', 'safety_gate_node')


def is_twist(sample, v, w, tol=1e-6):
    return sample is not None and math.isclose(sample[0], v, abs_tol=tol) and math.isclose(
        sample[1], w, abs_tol=tol)


class GateDriver:
    """Feeds a gate fake scans (10 Hz) and commands (20 Hz) and records what it publishes."""

    def __init__(self, node, namespace=''):
        self.node = node
        self.front = 2.0
        self.range_min = 0.15
        self.cmd = None  # (v, w); None stops commanding
        self.outputs = []  # (monotonic receive time, v, w)
        self.statuses = []  # (monotonic receive time, JSON text)
        self._scan_pub = node.create_publisher(
            LaserScan, namespace + '/scan', qos_profile_sensor_data)
        self._cmd_pub = node.create_publisher(Twist, namespace + '/cmd_vel_in', 10)
        node.create_subscription(Twist, namespace + '/cmd_vel', self._on_output, 10)
        node.create_subscription(
            String, namespace + '/safety_gate/status', self._on_status, 10)

    def _on_output(self, msg):
        self.outputs.append((time.monotonic(), msg.linear.x, msg.angular.z))

    def _on_status(self, msg):
        self.statuses.append((time.monotonic(), msg.data))

    def latest(self):
        return self.outputs[-1][1:] if self.outputs else None

    def outputs_since(self, t):
        return [sample[1:] for sample in self.outputs if sample[0] >= t]

    def latest_status(self):
        return json.loads(self.statuses[-1][1]) if self.statuses else None

    def run(self, duration, until=None):
        """Spin for `duration` s, or until `until()` holds. Returns False only on a timeout."""
        deadline = time.monotonic() + duration
        next_scan = next_cmd = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_scan:
                self._scan_pub.publish(make_scan(self.front, self.range_min))
                next_scan = now + 0.1
            if self.cmd is not None and now >= next_cmd:
                twist = Twist()
                twist.linear.x, twist.angular.z = self.cmd
                self._cmd_pub.publish(twist)
                next_cmd = now + 0.05
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if until is not None and until():
                return True
        return until is None

    def converge(self, v, w, timeout=DISCOVERY_TIMEOUT, tol=1e-6):
        return self.run(timeout, until=lambda: is_twist(self.latest(), v, w, tol))


class TestSafetyGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('safety_gate_tester')
        self.driver = GateDriver(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def assert_holds(self, v, w, duration=0.5, tol=1e-6):
        start = time.monotonic()
        self.driver.run(duration)
        held = self.driver.outputs_since(start)
        self.assertGreater(len(held), 0.5 * duration * 50, 'gate stopped publishing at 50 Hz')
        for sample in held:
            self.assertTrue(is_twist(sample, v, w, tol), f'expected ({v}, {w}), got {sample}')

    def wait_for_state(self, state, timeout=3.0):
        self.driver.statuses.clear()
        ok = self.driver.run(timeout, until=lambda: (
            self.driver.latest_status() or {}).get('state') == state)
        self.assertTrue(ok, f'status never reported {state}: {self.driver.latest_status()}')

    def test_forwards_command_when_clear(self):
        d = self.driver
        d.front, d.cmd = 2.0, (0.2, 0.3)
        self.assertTrue(d.converge(0.2, 0.3), f'never forwarded, last output {d.latest()}')
        self.assert_holds(0.2, 0.3)
        self.wait_for_state('ok')

    def test_clamps_to_speed_limits(self):
        d = self.driver
        d.front, d.cmd = 2.0, (1.0, -4.0)
        self.assertTrue(d.converge(0.25, -1.5), f'not clamped, last output {d.latest()}')

    def test_blocks_forward_at_obstacle_but_allows_rotation_and_reverse(self):
        d = self.driver
        d.front, d.range_min, d.cmd = 0.12, 0.05, (0.2, 0.3)
        self.assertTrue(d.converge(0.0, 0.3), f'not blocked, last output {d.latest()}')
        self.assert_holds(0.0, 0.3)
        self.wait_for_state('stop_obstacle')
        self.assertAlmostEqual(d.latest_status()['front_min_m'], 0.12, places=5)
        self.assertEqual(d.latest_status()['scale'], 0.0)

        d.cmd = (-0.1, 0.0)
        self.assertTrue(d.converge(-0.1, 0.0), f'reverse blocked, last output {d.latest()}')

    def test_obstacle_inside_lidar_dead_zone_stays_blocked(self):
        # With the A1's range_min of 0.15 m a box at 0.12 m reads invalid; the close reading
        # just before must keep it blocked.
        d = self.driver
        d.front, d.range_min, d.cmd = 0.30, 0.15, (0.2, 0.0)
        slow_v = 0.2 * (0.30 - 0.17) / (0.45 - 0.17)
        self.assertTrue(d.converge(slow_v, 0.0, tol=1e-4), f'not slowed, got {d.latest()}')
        self.wait_for_state('slow')

        d.front = 0.12
        self.assertTrue(d.converge(0.0, 0.0, timeout=2.0), f'not blocked, got {d.latest()}')
        self.assert_holds(0.0, 0.0, duration=1.5)
        self.wait_for_state('stop_obstacle')

    def test_zero_after_command_timeout(self):
        d = self.driver
        d.front, d.cmd = 2.0, (0.15, -0.5)
        self.assertTrue(d.converge(0.15, -0.5), f'never forwarded, last output {d.latest()}')

        d.cmd = None
        stopped_commanding = time.monotonic()
        self.assertTrue(d.converge(0.0, 0.0, timeout=2.0), 'output not zeroed after cmd_timeout')
        elapsed = time.monotonic() - stopped_commanding
        self.assertLess(elapsed, 1.0)
        self.assertTrue(
            any(not is_twist(s, 0.0, 0.0) for s in d.outputs_since(stopped_commanding)),
            'zeroed immediately instead of after cmd_timeout')
        self.assert_holds(0.0, 0.0)
        self.wait_for_state('stop_cmd_timeout')

    def test_status_json_schema_and_rate(self):
        d = self.driver
        d.front, d.cmd = 2.0, (0.1, 0.0)
        self.assertTrue(d.converge(0.1, 0.0))
        d.statuses.clear()
        start = time.monotonic()
        d.run(3.0)
        window = time.monotonic() - start
        self.assertGreaterEqual(len(d.statuses), int(2.0 * window) - 2, 'status not at 2 Hz')
        self.assertLessEqual(len(d.statuses), int(2.0 * window) + 2, 'status not at 2 Hz')

        previous_ticks = -1
        for _, text in d.statuses:
            status = json.loads(text)
            self.assertEqual(set(status), STATUS_KEYS)
            self.assertIn(status['state'], STATES)
            self.assertIsInstance(status['front_min_m'], (float, int))
            self.assertIsInstance(status['scale'], (float, int))
            self.assertEqual(set(status['period_ms']), {'p50', 'p99', 'max'})
            self.assertEqual(set(status['cb_latency_us']), {'p50', 'p99'})
            self.assertIsInstance(status['missed_deadlines'], int)
            self.assertIsInstance(status['ticks'], int)
            self.assertGreater(status['ticks'], previous_ticks)
            previous_ticks = status['ticks']
            self.assertGreater(status['period_ms']['p50'], 15.0)
            self.assertLess(status['period_ms']['p50'], 25.0)
            self.assertGreaterEqual(status['period_ms']['max'], status['period_ms']['p99'])
            self.assertGreaterEqual(status['period_ms']['p99'], status['period_ms']['p50'])
            self.assertGreater(status['cb_latency_us']['p50'], 0.0)
            self.assertGreaterEqual(status['cb_latency_us']['p99'], status['cb_latency_us']['p50'])
            self.assertLessEqual(status['missed_deadlines'], status['ticks'])
        self.assertEqual(json.loads(d.statuses[-1][1])['state'], 'ok')

    def test_parameters_have_interface_defaults_and_are_read_only(self):
        get_client = self.node.create_client(GetParameters, GATE_NODE + '/get_parameters')
        self.assertTrue(get_client.wait_for_service(timeout_sec=DISCOVERY_TIMEOUT))
        names = list(INTERFACE_DEFAULTS)
        response = self.call(get_client, GetParameters.Request(names=names))
        for name, value in zip(names, response.values):
            expected = INTERFACE_DEFAULTS[name]
            if isinstance(expected, bool):
                self.assertEqual(value.type, ParameterType.PARAMETER_BOOL, name)
                self.assertEqual(value.bool_value, expected, name)
            else:
                self.assertEqual(value.type, ParameterType.PARAMETER_DOUBLE, name)
                self.assertAlmostEqual(value.double_value, expected, msg=name)

        set_client = self.node.create_client(SetParameters, GATE_NODE + '/set_parameters')
        self.assertTrue(set_client.wait_for_service(timeout_sec=DISCOVERY_TIMEOUT))
        change = Parameter(name='stop_distance', value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=0.05))
        result = self.call(set_client, SetParameters.Request(parameters=[change])).results[0]
        self.assertFalse(result.successful)

    def test_zero_twist_published_on_shutdown(self):
        # A second gate in its own namespace, so it can be stopped without touching the
        # launched one. Its topics move with the namespace, which also covers remapping.
        probe = subprocess.Popen([gate_executable(), '--ros-args', '-r', '__ns:=/probe'])
        try:
            d = GateDriver(self.node, namespace='/probe')
            d.front, d.cmd = 2.0, (0.2, 0.4)
            self.assertTrue(d.converge(0.2, 0.4), f'probe never forwarded: {d.latest()}')

            # The launched gate is stopped with SIGINT by launch; use SIGTERM here (docker stop,
            # systemd) to cover the other handled signal.
            interrupted = time.monotonic()
            probe.send_signal(signal.SIGTERM)
            d.run(10.0, until=lambda: probe.poll() is not None)
            self.assertEqual(probe.wait(timeout=5.0), 0)
            d.cmd = None
            d.run(0.5)
            self.assertTrue(is_twist(d.latest(), 0.0, 0.0), f'last output {d.latest()}')
            self.assertTrue(
                any(is_twist(s, 0.0, 0.0) for s in d.outputs_since(interrupted)),
                'no zero twist after SIGTERM')
        finally:
            if probe.poll() is None:
                probe.kill()
                probe.wait()

    def test_refuses_to_start_with_inconsistent_parameters(self):
        result = subprocess.run(
            [gate_executable(), '--ros-args', '-r', '__ns:=/bad', '-p', 'stop_distance:=0.5'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            timeout=30.0)
        self.assertEqual(result.returncode, 1)
        self.assertIn('slow_distance must be greater than stop_distance', result.stderr)

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertTrue(future.done(), 'service call timed out')
        return future.result()


@launch_testing.post_shutdown_test()
class TestGateShutdown(unittest.TestCase):

    def test_exits_cleanly_on_sigint(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
