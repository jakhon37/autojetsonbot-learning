"""Simulated autoJetsonBot: the docking pipeline end to end without hardware (sim_world inside).

Stands in for diffdrive_node, rplidar_ros and camera_node with the same topics, so safety_gate,
dock_judge, servo_baseline, policy_runner, teleop and ros2 bag run unchanged against it.

Subscribes
    /cmd_vel          geometry_msgs/Twist (best effort, so reliable and best-effort publishers
                      both match); the command drops to zero after cmd_timeout without one
    /episode/event    std_msgs/String, reliable + transient_local: "start <method> <pose_id>
                      <round>" teleports the robot to that start pose, at rest, and clears the
                      collided flag; every other event is ignored
Publishes, all stamped with the node clock (ROS time)
    /odom                         nav_msgs/Odometry odom -> base_footprint at rate_hz: the true
                                  pose in the world frame and the measured (v, w)
    /scan                         sensor_msgs/LaserScan laser_frame at scan_rate: 360 beams from
                                  -pi, range_min 0.15, range_max 12, inf = no valid return
    /camera/image_raw             sensor_msgs/Image bgr8 640x480 camera_link at camera_fps
    /camera/image_raw/compressed  sensor_msgs/CompressedImage jpeg (jpeg_quality), same frames
                                  and stamps; a frame is only rendered if either has a subscriber
    /fake_robot/ground_truth      std_msgs/String JSON at ground_truth_rate:
                                  {"x", "y", "yaw" (world, m / rad), "collided" (bool, latched
                                  since the last start), "bumper_to_box_m" (0 when touching)}

Parameters (defaults)
    start_poses_file    ""  (= share/jbl_nodes/config/start_poses.yaml)
    initial_pose_id     0   pose taken at startup, before any episode event
    randomize           false   wheel bias and slip, start jitter, lidar noise, exposure and
                                pixel noise (sim_world.NoiseConfig); false is exact
    seed                0   random seed for randomize
    box_x, box_y        1.5, 0.0   centre of the box front face on the floor (world, m)
    box_yaw             pi  direction the front face (marker) looks; the room is 4 x 3 m
                            centred on the origin
    rate_hz 50.0   scan_rate 10.0   camera_fps 10.0   ground_truth_rate 10.0   (0 disables a topic)
    jpeg_quality 85     cmd_timeout 0.5 s (0 = hold the last command forever, like the real
                        motor bridge)

The simulation advances in fixed 1/rate_hz steps that track the node clock: a late timer tick
runs the missed steps (up to MAX_CATCH_UP_S) instead of stretching one.
"""

import array
import json
import math
import os
import signal
import sys
from typing import List, Optional

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import Header, String

from jbl_nodes.camera_node import COMPRESSED_TOPIC, RAW_TOPIC, RateLimiter
from jbl_nodes.episode_cli import EVENT_TOPIC, event_qos, parse_episode_event
from jbl_nodes.imgconv import numpy_to_compressed, numpy_to_image
from jbl_nodes.sim_world import SimWorld, WorldConfig, load_start_poses, start_pose_to_world

CMD_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
SCAN_TOPIC = "/scan"
GROUND_TRUTH_TOPIC = "/fake_robot/ground_truth"
ODOM_FRAME = "odom"
BASE_FRAME = "base_footprint"
LASER_FRAME = "laser_frame"
CAMERA_FRAME = "camera_link"
MAX_CATCH_UP_S = 1.0  # physics steps are cheap; only a long stall or a clock jump is dropped
LOG_THROTTLE_S = 5.0


def default_start_poses_file() -> str:
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("jbl_nodes"), "config", "start_poses.yaml")


def ground_truth_json(sim: SimWorld) -> str:
    pose = sim.pose
    return json.dumps(
        {
            "x": round(pose.x, 5),
            "y": round(pose.y, 5),
            "yaw": round(pose.yaw, 5),
            "collided": sim.collided,
            "bumper_to_box_m": round(sim.bumper_to_box(), 5),
        }
    )


class FakeRobot(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("fake_robot", **kwargs)
        poses_file = str(self.declare_parameter("start_poses_file", "").value) or default_start_poses_file()
        initial_pose_id = int(self.declare_parameter("initial_pose_id", 0).value)
        randomize = bool(self.declare_parameter("randomize", False).value)
        seed = int(self.declare_parameter("seed", 0).value)
        box_x = float(self.declare_parameter("box_x", 1.5).value)
        box_y = float(self.declare_parameter("box_y", 0.0).value)
        box_yaw = float(self.declare_parameter("box_yaw", math.pi).value)
        rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        scan_rate = float(self.declare_parameter("scan_rate", 10.0).value)
        camera_fps = float(self.declare_parameter("camera_fps", 10.0).value)
        gt_rate = float(self.declare_parameter("ground_truth_rate", 10.0).value)
        self._jpeg_quality = int(self.declare_parameter("jpeg_quality", 85).value)
        cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)

        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive, got %s" % rate_hz)
        if min(scan_rate, camera_fps, gt_rate, cmd_timeout) < 0.0:
            raise ValueError("scan_rate, camera_fps, ground_truth_rate and cmd_timeout must be >= 0")
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100], got %d" % self._jpeg_quality)

        world = WorldConfig(box_x=box_x, box_y=box_y, box_yaw=box_yaw)
        self._sim = SimWorld(world=world, randomize=randomize, seed=seed, dt=1.0 / rate_hz)
        self._poses = {p.pose_id: p for p in load_start_poses(poses_file)}
        blocked = [i for i, p in self._poses.items() if self._sim.in_collision(start_pose_to_world(p, world))]
        if blocked:
            raise ValueError("start poses %s overlap the box or a wall with this box pose" % blocked)
        if initial_pose_id not in self._poses:
            raise ValueError(
                "initial_pose_id %d is not in %s (0..%d)"
                % (initial_pose_id, poses_file, len(self._poses) - 1)
            )
        self._sim.reset_to_start(self._poses[initial_pose_id])

        self._dt_ns = int(round(1e9 / rate_hz))
        self._cmd_timeout_ns = int(cmd_timeout * 1e9)
        self._last_cmd_ns: Optional[int] = None
        self._last_tick_ns: Optional[int] = None
        self._backlog_ns = 0
        # Half a tick of tolerance keeps 10 Hz out of 50 Hz ticks at exactly every 5th tick.
        tolerance = 0.5 / rate_hz
        self._scan_limiter = RateLimiter(scan_rate, tolerance)
        self._camera_limiter = RateLimiter(camera_fps, tolerance)
        self._gt_limiter = RateLimiter(gt_rate, tolerance)
        self._scan_time = 1.0 / scan_rate if scan_rate > 0.0 else 0.0

        reliable = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=5, reliability=QoSReliabilityPolicy.RELIABLE
        )
        self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self._scan_pub = self.create_publisher(LaserScan, SCAN_TOPIC, reliable)
        self._raw_pub = self.create_publisher(Image, RAW_TOPIC, reliable)
        self._compressed_pub = self.create_publisher(CompressedImage, COMPRESSED_TOPIC, reliable)
        self._gt_pub = self.create_publisher(String, GROUND_TRUTH_TOPIC, 10)
        best_effort = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(Twist, CMD_TOPIC, self._on_cmd, best_effort)
        self.create_subscription(String, EVENT_TOPIC, self._on_event, event_qos())
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            "%d start poses from %s; box front at (%.2f, %.2f) facing %.0f deg; %s; "
            "physics %.0f Hz, scan %.1f Hz, camera %.1f Hz"
            % (
                len(self._poses),
                poses_file,
                box_x,
                box_y,
                math.degrees(box_yaw),
                "randomized (seed %d)" % seed if randomize else "exact",
                rate_hz,
                scan_rate,
                camera_fps,
            )
        )

    # ---------------------------------------------------------------- inputs

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_cmd(self, msg: Twist) -> None:
        v, w = msg.linear.x, msg.angular.z
        if not (math.isfinite(v) and math.isfinite(w)):
            self.get_logger().warning(
                "ignoring non-finite %s (%s, %s)" % (CMD_TOPIC, v, w), throttle_duration_sec=LOG_THROTTLE_S
            )
            return
        self._sim.set_command(v, w)
        self._last_cmd_ns = self._now_ns()

    def _on_event(self, msg: String) -> None:
        words = msg.data.split()
        if not words or words[0] != "start":
            return
        try:
            event = parse_episode_event(msg.data)
        except ValueError as exc:
            self.get_logger().warning("ignoring episode event %r: %s" % (msg.data, exc))
            return
        start = self._poses.get(event.pose_id)
        if start is None:
            self.get_logger().error(
                "'%s': no start pose %d (have 0..%d); staying put"
                % (msg.data, event.pose_id, len(self._poses) - 1)
            )
            return
        self._sim.reset_to_start(start)
        self._last_cmd_ns = None
        pose = self._sim.pose
        self.get_logger().info(
            "'%s': teleported to start pose %d at (%.3f, %.3f, %.1f deg)"
            % (msg.data, start.pose_id, pose.x, pose.y, math.degrees(pose.yaw))
        )

    # ---------------------------------------------------------------- simulation

    def _tick(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        if self._last_tick_ns is None or now_ns < self._last_tick_ns:  # first tick or clock reset
            self._last_tick_ns = now_ns
            return
        self._backlog_ns += now_ns - self._last_tick_ns
        self._last_tick_ns = now_ns
        if self._backlog_ns > MAX_CATCH_UP_S * 1e9:
            self.get_logger().warning(
                "fell %.0f ms behind the clock; dropping the backlog" % (self._backlog_ns * 1e-6),
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            self._backlog_ns = self._dt_ns

        if (
            self._cmd_timeout_ns > 0
            and self._last_cmd_ns is not None
            and now_ns - self._last_cmd_ns > self._cmd_timeout_ns
        ):
            self._sim.set_command(0.0, 0.0)
            self._last_cmd_ns = None
        while self._backlog_ns >= self._dt_ns:
            self._sim.step()
            self._backlog_ns -= self._dt_ns

        t = now_ns * 1e-9
        stamp = now.to_msg()
        self._publish_odom(stamp)
        if self._scan_limiter.ready(t):
            self._publish_scan(stamp)
        if self._camera_limiter.ready(t):
            self._publish_camera(stamp)
        if self._gt_limiter.ready(t):
            self._gt_pub.publish(String(data=ground_truth_json(self._sim)))

    # ---------------------------------------------------------------- outputs

    def _publish_odom(self, stamp: TimeMsg) -> None:
        pose = self._sim.pose
        v, w = self._sim.velocity
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id=ODOM_FRAME)
        msg.child_frame_id = BASE_FRAME
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.orientation.z = math.sin(0.5 * pose.yaw)
        msg.pose.pose.orientation.w = math.cos(0.5 * pose.yaw)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w
        self._odom_pub.publish(msg)

    def _publish_scan(self, stamp: TimeMsg) -> None:
        lidar = self._sim.lidar
        ranges = self._sim.scan()
        msg = LaserScan()
        msg.header = Header(stamp=stamp, frame_id=LASER_FRAME)
        msg.angle_min = lidar.angle_min
        msg.angle_increment = lidar.angle_increment
        msg.angle_max = lidar.angle_min + (lidar.num_beams - 1) * lidar.angle_increment
        msg.scan_time = self._scan_time
        msg.time_increment = self._scan_time / lidar.num_beams
        msg.range_min = lidar.range_min
        msg.range_max = lidar.range_max
        msg.ranges = array.array("f", ranges.tobytes())
        self._scan_pub.publish(msg)

    def _publish_camera(self, stamp: TimeMsg) -> None:
        send_raw = self._raw_pub.get_subscription_count() > 0
        send_compressed = self._compressed_pub.get_subscription_count() > 0
        if not (send_raw or send_compressed):
            return
        frame = self._sim.render()
        if send_raw:
            msg = numpy_to_image(frame, "bgr8")
            msg.header = Header(stamp=stamp, frame_id=CAMERA_FRAME)
            self._raw_pub.publish(msg)
        if send_compressed:
            msg = numpy_to_compressed(frame, "jpeg", self._jpeg_quality)
            msg.header = Header(stamp=stamp, frame_id=CAMERA_FRAME)
            self._compressed_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    try:
        node = FakeRobot()
    except (ValueError, OSError, ParameterException) as exc:
        get_logger("fake_robot").fatal("cannot start: %s" % exc)
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ros2 launch forwards Ctrl-C again: finish cleanup
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
