"""Classical docking baseline: ArUco bearing + lidar range -> /cmd_vel_in (method "servo").

The control law is judge_logic.servo_command (pure, unit-tested): turn towards the marker,
drive forward proportionally to the lidar range error, slow down while the marker is far off
centre, stop in the band once aligned, rotate in place to search when the marker is not in
view. When the lidar has no valid front reading (e.g. closer than its 0.15 m minimum range),
the range is estimated from the marker's apparent size instead.

Subscribes
    image_topic       CompressedImage if the name ends in "/compressed", else Image; best
                      effort, depth 1; the newest frame is decoded once, on the next tick
    /scan             LaserScan (sensor-data QoS) -> front-sector median range
    /episode/event    reliable + transient_local: "start <method_name> ..." enables, any "end"
                      or a start for another method disables
Publishes
    /cmd_vel_in       geometry_msgs/Twist at rate_hz while enabled (or always with always_on);
                      one zero twist when disabled, then nothing, so teleop or a policy can
                      own the topic

While enabled: no frame for image_timeout_s publishes zero (camera fault). A marker that
disappears for less than marker_lost_s also publishes zero (a dropped detection is not a
reason to spin away); after that the robot searches, turning towards where it last saw it.

Parameters (defaults)
    method_name servo     always_on false     rate_hz 20.0
    image_topic /camera/image_raw/compressed  marker_id 7   hfov_rad 1.396
    forward_angle 0.0     sector_half_angle 0.17
    target_range 0.25     range_tol 0.02      k_v 1.0       k_w 1.2
    v_max 0.15            w_max 0.8           search_w 0.5
    slow_bearing_deg 25.0 bearing_deadband_deg 4.0
    marker_side_m 0.10    camera_ahead_of_lidar_m 0.033
    image_timeout_s 0.5   scan_timeout_s 0.5  marker_lost_s 0.3
"""

import math
import signal
from typing import Any, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from jbl_nodes.aruco_utils import MarkerObserver
from jbl_nodes.dock_judge import SCAN_TOPIC, Stamped, image_subscription
from jbl_nodes.episode_cli import EVENT_TOPIC, event_qos, parse_episode_event
from jbl_nodes.judge_logic import (
    MarkerObservation,
    ServoParams,
    front_range,
    range_from_marker_area,
    servo_command,
    validate_servo_params,
)

CMD_TOPIC = "/cmd_vel_in"
LOG_THROTTLE_S = 5.0


class ServoBaseline(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("servo_baseline", **kwargs)
        p = self.declare_parameter
        self._method = str(p("method_name", "servo").value)
        self._always_on = bool(p("always_on", False).value)
        rate_hz = float(p("rate_hz", 20.0).value)
        image_topic = str(p("image_topic", "/camera/image_raw/compressed").value)
        marker_id = int(p("marker_id", 7).value)
        self._hfov = float(p("hfov_rad", 1.396).value)
        self._forward_angle = float(p("forward_angle", 0.0).value)
        self._half_angle = float(p("sector_half_angle", 0.17).value)
        self._params = ServoParams(
            target_range=float(p("target_range", 0.25).value),
            range_tol=float(p("range_tol", 0.02).value),
            k_v=float(p("k_v", 1.0).value),
            k_w=float(p("k_w", 1.2).value),
            v_max=float(p("v_max", 0.15).value),
            w_max=float(p("w_max", 0.8).value),
            search_w=float(p("search_w", 0.5).value),
            slow_bearing_deg=float(p("slow_bearing_deg", 25.0).value),
            bearing_deadband_deg=float(p("bearing_deadband_deg", 4.0).value),
        )
        self._marker_side = float(p("marker_side_m", 0.10).value)
        self._camera_ahead = float(p("camera_ahead_of_lidar_m", 0.033).value)
        self._image_timeout_ns = int(float(p("image_timeout_s", 0.5).value) * 1e9)
        self._scan_timeout_ns = int(float(p("scan_timeout_s", 0.5).value) * 1e9)
        self._marker_lost_ns = int(float(p("marker_lost_s", 0.3).value) * 1e9)
        validate_servo_params(self._params)
        if rate_hz <= 0.0 or self._half_angle <= 0.0 or self._marker_side <= 0.0:
            raise ValueError("rate_hz, sector_half_angle and marker_side_m must be positive")

        self._observer = MarkerObserver(marker_id, self._hfov)
        self._enabled = self._always_on
        self._image = Stamped()
        self._image_pending = False
        self._range = Stamped()
        self._marker: Optional[MarkerObservation] = None  # from the newest decoded frame
        self._frame_size: Tuple[int, int] = (640, 480)
        self._last_seen_ns: Optional[int] = None
        self._search_sign = 1.0  # turn left first, then towards where the marker was last seen
        self._marker_visible: Optional[bool] = None  # for logging transitions only

        self._decode = image_subscription(self, image_topic, self._on_image)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(String, EVENT_TOPIC, self._on_event, event_qos())
        self._cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            "servo on marker %d from %s at %.0f Hz, %s"
            % (
                marker_id,
                image_topic,
                rate_hz,
                "always on" if self._always_on else "waiting for 'start %s ...'" % self._method,
            )
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_image(self, msg: Any) -> None:
        self._image.set(msg, self._now_ns())
        self._image_pending = True

    def _on_scan(self, msg: LaserScan) -> None:
        r = front_range(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            self._forward_angle,
            self._half_angle,
        )
        self._range.set(r, self._now_ns())

    def _on_event(self, msg: String) -> None:
        if self._always_on:
            return
        try:
            event = parse_episode_event(msg.data)
        except ValueError:
            return  # "intervention" and other notes are not ours to act on
        if event.kind == "start" and event.method == self._method:
            if not self._enabled:
                self.get_logger().info("enabled by '%s'" % msg.data)
            self._enabled = True
            self._marker = None
            self._last_seen_ns = None
            self._search_sign = 1.0
            return
        if self._enabled:
            self._enabled = False
            self._publish(0.0, 0.0)
            self.get_logger().info("disabled by '%s'" % msg.data)

    def _tick(self) -> None:
        if not self._enabled:
            return
        now_ns = self._now_ns()
        if self._image.t_ns is None or now_ns - self._image.t_ns > self._image_timeout_ns:
            self.get_logger().warning(
                "no camera frame for %.1f s; holding still" % (self._image_timeout_ns * 1e-9),
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            self._publish(0.0, 0.0)
            return
        if self._image_pending:
            self._image_pending = False
            self._update_marker(now_ns)
        if self._marker is not None:
            lidar = self._range.fresh(now_ns, self._scan_timeout_ns)
            range_m = lidar if lidar is not None else self._range_from_marker()
            v, w = servo_command(self._marker.bearing_deg, range_m, self._params)
        elif self._last_seen_ns is not None and now_ns - self._last_seen_ns < self._marker_lost_ns:
            v, w = 0.0, 0.0
        else:
            v, w = servo_command(None, None, self._params, self._search_sign)
        self._publish(v, w)

    def _update_marker(self, now_ns: int) -> None:
        try:
            frame = self._decode(self._image.value)
            self._frame_size = (frame.shape[1], frame.shape[0])
            self._marker = self._observer.observe(frame)
        except Exception as exc:  # a corrupt frame counts as "marker not seen"
            self.get_logger().error("cannot use image: %s" % exc, throttle_duration_sec=LOG_THROTTLE_S)
            self._marker = None
        if self._marker is not None:
            self._last_seen_ns = now_ns
            if abs(self._marker.bearing_deg) > self._params.bearing_deadband_deg:
                self._search_sign = math.copysign(1.0, self._marker.bearing_deg)
        visible = self._marker is not None
        if visible != self._marker_visible:  # throttled: a marker at the detection limit flickers
            self.get_logger().info(
                "marker %s" % ("in view" if visible else "lost"), throttle_duration_sec=1.0
            )
            self._marker_visible = visible

    def _range_from_marker(self) -> float:
        width, height = self._frame_size
        return range_from_marker_area(
            self._marker.area_frac, width, height, self._hfov, self._marker_side, self._camera_ahead
        )

    def _publish(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def destroy_node(self) -> None:
        if self._enabled and self.context.ok():
            try:  # best effort; safety_gate's cmd_timeout stops the robot regardless
                self._publish(0.0, 0.0)
            except Exception:
                pass
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ServoBaseline()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ros2 launch forwards Ctrl-C again: finish cleanup
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
