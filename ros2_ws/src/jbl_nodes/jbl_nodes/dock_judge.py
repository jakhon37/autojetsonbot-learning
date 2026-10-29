"""Referee for the docking task: decides success or failure of each episode (docs/INTERFACES.md).

Subscribes
    image_topic             CompressedImage if the name ends in "/compressed", else Image
                            (bgr8/rgb8/mono8); best effort, depth 1; only the newest frame is
                            decoded, once, on the next tick
    /scan                   LaserScan (sensor-data QoS) -> front-sector median range
    /odom                   Odometry: twist.linear.x, twist.angular.z decide "stopped"
    /cmd_vel_in             Twist (reliable, like the gate): the controller's request must be
                            still too for "stopped"; none within cmd_timeout_s counts as still
                            (the gate then outputs zero)
    /safety_gate/status     String JSON: "state" drives the pushed-into-obstacle rule
    /episode/event          reliable + transient_local: "start ..." begins judging, "end ..."
                            stops it, "intervention [...]" fails the running episode
Publishes
    /dock_judge/status      String JSON every tick (schema in INTERFACES.md); success is live:
                            the conditions have held for hold_s right now
    /episode/event          with auto_end: "end success" or "end fail <reason>" once per
                            episode, reason one of timeout, stop_obstacle, intervention
                            (reliable + transient_local, like every /episode/event writer)

Inputs older than their timeout count as unknown (no marker, no range, not stopped).

Parameters (defaults)
    image_topic /camera/image_raw/compressed   marker_id 7       hfov_rad 1.396
    forward_angle 0.0       sector_half_angle 0.17      band_min 0.18     band_max 0.32
    bearing_tol_deg 10.0    hold_s 1.0                  timeout_s 30.0    auto_end true
    stopped_v 0.02          stopped_w 0.05              obstacle_fail_s 2.0
    rate_hz 10.0            image_timeout_s 0.5         scan_timeout_s 0.5
    odom_timeout_s 0.5      gate_timeout_s 1.5          cmd_timeout_s 0.3 (= the gate's)
"""

import json
import signal
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String

from jbl_nodes.aruco_utils import MarkerObserver
from jbl_nodes.episode_cli import (
    EVENT_TOPIC,
    EpisodeEvent,
    event_qos,
    format_episode_event,
    parse_episode_event,
)
from jbl_nodes.imgconv import compressed_to_bgr, image_to_bgr
from jbl_nodes.judge_logic import (
    OUTCOME_SUCCESS,
    STATUS_TOPIC,
    DockJudge,
    JudgeConfig,
    MarkerObservation,
    Verdict,
    front_range,
    is_intervention,
)

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
CMD_IN_TOPIC = "/cmd_vel_in"
GATE_TOPIC = "/safety_gate/status"
LOG_THROTTLE_S = 5.0


def image_subscription(
    node: Node, topic: str, callback: Callable[[Any], None]
) -> Callable[[Any], np.ndarray]:
    """Subscribe to a raw or compressed image topic (by name); returns the matching BGR decoder."""
    compressed = topic.rstrip("/").endswith("/compressed")
    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT
    )
    node.create_subscription(CompressedImage if compressed else Image, topic, callback, qos)
    return compressed_to_bgr if compressed else image_to_bgr


class Stamped(object):
    """Latest value of an input with its arrival time (node clock, ns)."""

    def __init__(self) -> None:
        self.value: Any = None
        self.t_ns: Optional[int] = None

    def set(self, value: Any, t_ns: int) -> None:
        self.value, self.t_ns = value, t_ns

    def fresh(self, now_ns: int, timeout_ns: int) -> Any:
        """The value if it arrived within timeout_ns of now_ns, else None."""
        if self.t_ns is None or now_ns - self.t_ns > timeout_ns:
            return None
        return self.value


class DockJudgeNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("dock_judge", **kwargs)
        p = self.declare_parameter
        image_topic = str(p("image_topic", "/camera/image_raw/compressed").value)
        config = JudgeConfig(
            marker_id=int(p("marker_id", 7).value),
            band_min=float(p("band_min", 0.18).value),
            band_max=float(p("band_max", 0.32).value),
            bearing_tol_deg=float(p("bearing_tol_deg", 10.0).value),
            hold_s=float(p("hold_s", 1.0).value),
            timeout_s=float(p("timeout_s", 30.0).value),
            stopped_v=float(p("stopped_v", 0.02).value),
            stopped_w=float(p("stopped_w", 0.05).value),
            obstacle_fail_s=float(p("obstacle_fail_s", 2.0).value),
        )
        hfov_rad = float(p("hfov_rad", 1.396).value)
        self._forward_angle = float(p("forward_angle", 0.0).value)
        self._half_angle = float(p("sector_half_angle", 0.17).value)
        self._auto_end = bool(p("auto_end", True).value)
        rate_hz = float(p("rate_hz", 10.0).value)
        self._image_timeout_ns = int(float(p("image_timeout_s", 0.5).value) * 1e9)
        self._scan_timeout_ns = int(float(p("scan_timeout_s", 0.5).value) * 1e9)
        self._odom_timeout_ns = int(float(p("odom_timeout_s", 0.5).value) * 1e9)
        self._gate_timeout_ns = int(float(p("gate_timeout_s", 1.5).value) * 1e9)
        self._cmd_timeout_ns = int(float(p("cmd_timeout_s", 0.3).value) * 1e9)
        if rate_hz <= 0.0 or self._half_angle <= 0.0:
            raise ValueError("rate_hz and sector_half_angle must be positive")

        self._judge = DockJudge(config)  # validates the config
        self._observer = MarkerObserver(config.marker_id, hfov_rad)
        self._episode: Optional[EpisodeEvent] = None

        self._image = Stamped()  # newest message, decoded on the next tick
        self._image_pending = False
        self._marker = Stamped()  # observation from the newest decoded frame (None = not seen)
        self._range = Stamped()
        self._odom = Stamped()
        self._gate = Stamped()
        self._cmd = Stamped()

        self._decode = image_subscription(self, image_topic, self._on_image)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos_profile_sensor_data)
        self.create_subscription(String, GATE_TOPIC, self._on_gate, 10)
        self.create_subscription(Twist, CMD_IN_TOPIC, self._on_cmd, 10)
        self.create_subscription(String, EVENT_TOPIC, self._on_event, event_qos())
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self._event_pub = self.create_publisher(String, EVENT_TOPIC, event_qos())
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            "judging marker %d on %s: band [%.2f, %.2f] m, |bearing| <= %.1f deg, hold %.1f s, "
            "timeout %.0f s, auto_end %s"
            % (
                config.marker_id,
                image_topic,
                config.band_min,
                config.band_max,
                config.bearing_tol_deg,
                config.hold_s,
                config.timeout_s,
                self._auto_end,
            )
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    # ---------------------------------------------------------------- inputs

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

    def _on_odom(self, msg: Odometry) -> None:
        self._odom.set((msg.twist.twist.linear.x, msg.twist.twist.angular.z), self._now_ns())

    def _on_cmd(self, msg: Twist) -> None:
        self._cmd.set((msg.linear.x, msg.angular.z), self._now_ns())

    def _on_gate(self, msg: String) -> None:
        try:
            state = json.loads(msg.data)["state"]
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().warning(
                "ignoring malformed %s: %s" % (GATE_TOPIC, exc), throttle_duration_sec=LOG_THROTTLE_S
            )
            return
        self._gate.set(str(state), self._now_ns())

    def _on_event(self, msg: String) -> None:
        now = self._now_ns() * 1e-9
        if is_intervention(msg.data):
            verdict = self._judge.intervention(now)
            if verdict is not None:
                self._finish(verdict)
            return
        try:
            event = parse_episode_event(msg.data)
        except ValueError as exc:
            self.get_logger().warning("ignoring episode event %r: %s" % (msg.data, exc))
            return
        if event.kind == "start":
            self._episode = event
            self._judge.start_episode(now)
            self.get_logger().info("judging episode '%s'" % msg.data)
        elif self._judge.episode_active:
            self._judge.end_episode()
            self.get_logger().info("episode ended externally ('%s') before a verdict" % msg.data)
        else:
            self._judge.end_episode()

    # ---------------------------------------------------------------- tick

    def _observe(self, now_ns: int) -> Optional[MarkerObservation]:
        if self._image_pending:
            self._image_pending = False
            try:
                obs = self._observer.observe(self._decode(self._image.value))
            except Exception as exc:  # a corrupt frame must not take the referee down
                self.get_logger().error("cannot use image: %s" % exc, throttle_duration_sec=LOG_THROTTLE_S)
                obs = None
            self._marker.set(obs, self._image.t_ns)
        return self._marker.fresh(now_ns, self._image_timeout_ns)

    def _tick(self) -> None:
        now_ns = self._now_ns()
        marker = self._observe(now_ns)
        odom: Optional[Tuple[float, float]] = self._odom.fresh(now_ns, self._odom_timeout_ns)
        v, w = odom if odom is not None else (None, None)
        verdict = self._judge.update(
            now_ns * 1e-9,
            marker,
            self._range.fresh(now_ns, self._scan_timeout_ns),
            v,
            w,
            self._gate.fresh(now_ns, self._gate_timeout_ns),
            self._cmd.fresh(now_ns, self._cmd_timeout_ns),
        )
        self._status_pub.publish(String(data=json.dumps(self._judge.status())))
        if verdict is not None:
            self._finish(verdict)

    def _finish(self, verdict: Verdict) -> None:
        label = format_episode_event(self._episode) if self._episode is not None else "episode"
        self.get_logger().info(
            "'%s': %s%s after %.2f s"
            % (label, verdict.outcome, " (%s)" % verdict.reason if verdict.reason else "", verdict.duration_s)
        )
        if not self._auto_end:
            return
        reason = "" if verdict.outcome == OUTCOME_SUCCESS else verdict.reason
        text = format_episode_event(EpisodeEvent("end", outcome=verdict.outcome, reason=reason))
        self._event_pub.publish(String(data=text))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DockJudgeNode()
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
