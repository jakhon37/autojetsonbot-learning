"""Drive the robot with a learned policy: newest camera frame -> /cmd_vel_in at a fixed rate.

Subscribes
    image_topic       CompressedImage if the name ends in "/compressed", else Image
                      (bgr8/rgb8/mono8); depth 1, decoded only when used
    /episode/event    reliable + transient_local: "start ..." enables, "end ..." disables
Publishes
    /cmd_vel_in       geometry_msgs/Twist every tick while enabled; nothing while disabled,
                      so teleop or the baseline can own the topic
    /policy/latency   std_msgs/String JSON per inference:
                      {"preprocess_ms", "infer_ms", "total_ms", "image_age_ms", "backend"}
                      preprocess_ms = decode + preprocess, total_ms = tick start to publish,
                      image_age_ms = clock now - header.stamp at tick start

Each tick while enabled: no frame yet, a frame older than max_image_age, a stamp in the
future by more than max_image_age (unsynchronised clocks) or an inference error all publish
a zero twist. A frame with a zero stamp (an unstamped camera) is aged from its arrival.

Parameters
    model_path        model file; policy_meta.json beside it is checked when present
    backend           onnxruntime | tensorrt | torch | constant
    image_topic       /camera/image_raw/compressed
    rate_hz           10.0
    max_image_age     0.2 s
    enabled_on_start  false
    constant_action   [0.0, 0.0]  normalised action of the constant backend
    method_filter     ""  comma-separated methods whose "start" enables this runner (empty: any);
                      a "start" for another method disables it
    image_reliability auto | best_effort | reliable. auto = best effort for compressed frames
                      (newest-only, no retransmission stalls) and reliable for raw ones: a 640x480
                      raw frame exceeds Fast DDS's default shared-memory segment, goes out as
                      fragmented UDP, and best effort then delivers only ~5 of 30 frames/s across
                      processes (measured on Humble defaults); reliable delivers all 30. Best effort
                      matches any publisher; reliable needs a reliable one (camera_node is).
"""

import json
import signal
import sys
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from jbl_nodes.episode_cli import EVENT_TOPIC, EpisodeEvent, event_qos, parse_episode_event
from jbl_nodes.imgconv import compressed_to_bgr, image_to_bgr
from jbl_nodes.policy_backends import INPUT_SHAPE, META_FILENAME, create_backend
from jbl_nodes.preprocess import action_to_twist, preprocess

CMD_TOPIC = "/cmd_vel_in"
LATENCY_TOPIC = "/policy/latency"
WARMUP_RUNS = 2
LOG_THROTTLE_S = 2.0
RELIABILITY = {
    "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
    "reliable": QoSReliabilityPolicy.RELIABLE,
}


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


class PolicyRunner(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("policy_runner", **kwargs)
        model_path = str(self.declare_parameter("model_path", "").value)
        backend_name = str(self.declare_parameter("backend", "onnxruntime").value)
        self._image_topic = str(self.declare_parameter("image_topic", "/camera/image_raw/compressed").value)
        rate_hz = float(self.declare_parameter("rate_hz", 10.0).value)
        self._max_age_ns = int(float(self.declare_parameter("max_image_age", 0.2).value) * 1e9)
        self._enabled = bool(self.declare_parameter("enabled_on_start", False).value)
        constant_action = [float(v) for v in self.declare_parameter("constant_action", [0.0, 0.0]).value]
        method_filter = str(self.declare_parameter("method_filter", "").value)
        self._methods = [m.strip() for m in method_filter.split(",") if m.strip()]
        reliability = str(self.declare_parameter("image_reliability", "auto").value)
        if rate_hz <= 0 or self._max_age_ns <= 0:
            raise ValueError("rate_hz and max_image_age must be positive")
        if reliability not in ("auto",) + tuple(RELIABILITY):
            raise ValueError("image_reliability must be auto, best_effort or reliable, got %r" % reliability)

        self._backend = create_backend(backend_name, model_path, constant_action=constant_action)
        self._report_backend(model_path)
        self._warm_up()

        self._latest: Optional[Tuple[Any, int]] = None
        self._unstamped_warned = False
        compressed = self._image_topic.rstrip("/").endswith("/compressed")
        self._decode: Callable[[Any], np.ndarray] = compressed_to_bgr if compressed else image_to_bgr
        if reliability == "auto":
            reliability = "best_effort" if compressed else "reliable"
        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=RELIABILITY[reliability]
        )
        self.create_subscription(
            CompressedImage if compressed else Image, self._image_topic, self._on_image, image_qos
        )
        self.create_subscription(String, EVENT_TOPIC, self._on_event, event_qos())
        self._cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self._latency_pub = self.create_publisher(String, LATENCY_TOPIC, 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            "%s on %s (%s) at %.1f Hz, max image age %.0f ms, %s"
            % (
                self._backend.description,
                self._image_topic,
                reliability,
                rate_hz,
                self._max_age_ns * 1e-6,
                "enabled" if self._enabled else "waiting for an episode start",
            )
        )

    def _report_backend(self, model_path: str) -> None:
        if self._backend.name == "constant":
            return
        if self._backend.meta is None:
            self.get_logger().warning(
                "no %s beside %s: input size and normalisation not verified" % (META_FILENAME, model_path)
            )
        else:
            self.get_logger().info(
                "%s checked: %s" % (META_FILENAME, ", ".join(self._backend.meta_checked) or "nothing")
            )

    def _warm_up(self) -> None:
        # The first runs allocate and autotune; keep that out of the first real tick.
        x = np.zeros(INPUT_SHAPE, dtype=np.float32)
        for _ in range(WARMUP_RUNS):
            t0 = time.perf_counter()
            self._backend.infer(x)
        self.get_logger().info("warm-up inference %.1f ms" % ((time.perf_counter() - t0) * 1e3))

    def _on_image(self, msg: Any) -> None:
        self._latest = (msg, self.get_clock().now().nanoseconds)

    def _on_event(self, msg: String) -> None:
        try:
            event = parse_episode_event(msg.data)
        except ValueError as exc:
            self.get_logger().warning("ignoring episode event %r: %s" % (msg.data, exc))
            return
        if event.kind == "start" and self._accepts(event):
            if not self._enabled:
                self.get_logger().info("enabled by '%s'" % msg.data)
            self._enabled = True
            return
        if event.kind == "start":
            self.get_logger().info(
                "'%s' is for another method (method_filter: %s)" % (msg.data, ",".join(self._methods))
            )
        if self._enabled:
            self._enabled = False
            self._publish_twist(0.0, 0.0)
            self.get_logger().info("disabled by '%s'" % msg.data)

    def _accepts(self, event: EpisodeEvent) -> bool:
        return not self._methods or event.method in self._methods

    def _tick(self) -> None:
        if not self._enabled:
            return
        t0 = time.perf_counter()
        age_ns = self._image_age_ns()
        if age_ns is None:
            self._publish_twist(0.0, 0.0)
            return
        try:
            t_pre = time.perf_counter()
            frame = self._decode(self._latest[0])
            x = preprocess(frame, is_bgr=True)
            t1 = time.perf_counter()
            action = self._backend.infer(x)
            t2 = time.perf_counter()
        except Exception as exc:  # a bad frame or a backend fault must stop the robot, not the node
            self.get_logger().error(
                "inference failed, publishing zero: %s" % exc, throttle_duration_sec=LOG_THROTTLE_S
            )
            self._publish_twist(0.0, 0.0)
            return
        v, w = action_to_twist(action)
        self._publish_twist(v, w)
        t3 = time.perf_counter()
        report = {
            "preprocess_ms": round((t1 - t_pre) * 1e3, 3),
            "infer_ms": round((t2 - t1) * 1e3, 3),
            "total_ms": round((t3 - t0) * 1e3, 3),
            "image_age_ms": round(age_ns * 1e-6, 3),
            "backend": self._backend.name,
        }
        self._latency_pub.publish(String(data=json.dumps(report)))

    def _image_age_ns(self) -> Optional[int]:
        """Age of the newest frame, or None (with a throttled reason) if it must not be used."""
        log = self.get_logger()
        if self._latest is None:
            log.warning(
                "no image on %s yet; publishing zero" % self._image_topic,
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return None
        msg, received_ns = self._latest
        stamp_ns = _stamp_ns(msg.header.stamp)
        if stamp_ns == 0:
            if not self._unstamped_warned:
                log.warning("images on %s have no header stamp; ageing them from arrival" % self._image_topic)
                self._unstamped_warned = True
            stamp_ns = received_ns
        age_ns = self.get_clock().now().nanoseconds - stamp_ns
        if age_ns > self._max_age_ns:
            log.warning(
                "newest image is %.0f ms old (limit %.0f ms); publishing zero"
                % (age_ns * 1e-6, self._max_age_ns * 1e-6),
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return None
        if age_ns < -self._max_age_ns:
            log.warning(
                "image stamp is %.0f ms in the future: camera and runner clocks are not synchronised "
                "(run chrony/NTP); publishing zero" % (-age_ns * 1e-6),
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return None
        return age_ns

    def _publish_twist(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def destroy_node(self) -> None:
        if self._enabled and self.context.ok():
            try:  # best effort; safety_gate's cmd_timeout stops the robot regardless
                self._publish_twist(0.0, 0.0)
            except Exception:
                pass
        self._backend.close()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    try:
        node = PolicyRunner()
    except (ValueError, OSError, ImportError, RuntimeError, ParameterException) as exc:
        get_logger("policy_runner").fatal("cannot start: %s" % exc)
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
