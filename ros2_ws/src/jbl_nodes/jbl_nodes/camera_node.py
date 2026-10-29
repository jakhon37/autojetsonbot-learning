"""Camera publisher with capture-time stamps (replaces jetson_bot_detection/camera_go.py).

Publishes
    /camera/image_raw             sensor_msgs/Image, bgr8, width x height, at most ``fps``
    /camera/image_raw/compressed  sensor_msgs/CompressedImage, jpeg, at most ``compressed_fps``
Both carry header.stamp = the moment the frame was grabbed and header.frame_id = ``frame_id``;
a compressed frame has the same stamp as the raw frame it was encoded from. Nothing is
converted or encoded for a topic that has no subscriber.

Parameters
    source          "camera" or "test_pattern" (moving synthetic frames, no hardware needed)
    device          index ("0", default, or an int), a device path / video file / URL, or a
                    GStreamer pipeline (any string containing '!'), e.g. for the Pi Camera v2 on
                    the Nano (config/robot.yaml uses this one):
                    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1640,height=1232,
                    framerate=30/1 ! nvvidconv ! video/x-raw,width=640,height=480,format=BGRx !
                    videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
                    1640x1232 is the IMX219's full-sensor binned 4:3 mode, so the image keeps
                    the lens's whole field of view and its aspect ratio. Its 1280x720 and
                    1920x1080 modes are centre crops: a narrower HFOV, and a VFOV too small to
                    see the 10 cm marker when docked. The CSI camera is not a V4L2 device that
                    OpenCV can read (/dev/video0 is raw RG10 Bayer): nvarguscamerasrc needs the
                    container started with --runtime nvidia and -v /tmp/argus_socket:/tmp/argus_socket
                    (docs/PROTOCOL.md section 4).
    width, height   published size (int, 640 x 480); other capture sizes are resized
    fps             raw publish rate cap and the rate requested from the driver (double, 30.0)
    compressed_fps  jpeg publish rate cap (double, 15.0; 0 disables the topic)
    jpeg_quality    1-100 (int, 85)
    frame_id        "camera_link"

Capture runs on its own thread so a blocking read never stalls the executor. Read failures
are logged (throttled); after a run of them the device is closed and reopened with backoff.
"""

import functools
import re
import signal
import sys
import threading
import time
from typing import List, Optional, Union

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from jbl_nodes.imgconv import numpy_to_compressed, numpy_to_image

RAW_TOPIC = "/camera/image_raw"
COMPRESSED_TOPIC = "/camera/image_raw/compressed"
SOURCES = ("camera", "test_pattern")
READ_RETRY_S = 0.05
MAX_CONSECUTIVE_FAILURES = 10
REOPEN_BACKOFF_MIN_S = 0.5
REOPEN_BACKOFF_MAX_S = 5.0
LOG_THROTTLE_S = 5.0

Device = Union[int, str]


def parse_device(value: object) -> Device:
    """Parameter value -> camera index (int) or a path / URL / GStreamer pipeline (str)."""
    if isinstance(value, bool):
        raise ValueError("device must be an int or a string, got %r" % (value,))
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("device must not be empty")
    return int(text) if re.match(r"^\d+$", text) else text


class RateLimiter(object):
    """Lets through at most ``rate_hz`` events per second from a stream of timestamps.

    A tolerance of half an input frame keeps 30 -> 15 Hz decimation at exactly every second
    frame despite capture jitter; after a gap the schedule restarts instead of bursting.
    """

    def __init__(self, rate_hz: float, tolerance_s: float) -> None:
        self._period = 1.0 / rate_hz if rate_hz > 0 else None
        self._tolerance = tolerance_s
        self._next: Optional[float] = None

    def ready(self, t: float) -> bool:
        if self._period is None:
            return False
        if self._next is not None and t < self._next - self._tolerance:
            return False
        on_schedule = self._next is not None and t - self._next < self._period
        self._next = self._next + self._period if on_schedule else t + self._period
        return True


@functools.lru_cache(maxsize=2)
def _gradient(width: int, height: int) -> np.ndarray:
    blue = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(height, 0)
    green = np.linspace(0, 255, height, dtype=np.float32)[:, None].repeat(width, 1)
    red = np.full((height, width), 96, dtype=np.float32)
    return np.dstack([blue, green, red]).astype(np.uint8)


def render_test_pattern(index: int, width: int, height: int) -> np.ndarray:
    """Synthetic BGR frame: a colour gradient (blue grows to the right, green downwards, so a
    channel swap is visible), a white square sweeping left-right, and the frame index."""
    frame = _gradient(width, height).copy()
    side = max(4, height // 4)
    travel = max(1, width - side)
    x = (index * 8) % (2 * travel)
    x = x if x < travel else 2 * travel - x
    y = (height - side) // 2
    cv2.rectangle(frame, (x, y), (x + side - 1, y + side - 1), (255, 255, 255), -1)
    cv2.putText(
        frame,
        str(index),
        (8, max(16, height // 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, height / 960.0),
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


class OpenCvSource(object):
    """cv2.VideoCapture on an index, path/URL or GStreamer pipeline."""

    def __init__(self, device: Device, width: int, height: int, fps: float) -> None:
        self._device = device
        self._pipeline = isinstance(device, str) and "!" in device
        self._width, self._height, self._fps = width, height, fps
        self._cap: Optional[cv2.VideoCapture] = None

    def describe(self) -> str:
        return "GStreamer pipeline" if self._pipeline else "device %r" % (self._device,)

    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> bool:
        self.close()
        cap = (
            cv2.VideoCapture(self._device, cv2.CAP_GSTREAMER)
            if self._pipeline
            else cv2.VideoCapture(self._device)
        )
        if not cap.isOpened():
            cap.release()
            return False
        if not self._pipeline:  # a pipeline fixes its own caps; drivers take requests
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not one queued in the driver
        self._cap = cap
        return True

    def grab(self) -> bool:
        return self._cap is not None and self._cap.grab()

    def retrieve(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.retrieve()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource(object):
    """Frames from render_test_pattern, paced at ``fps`` like a real camera."""

    def __init__(self, width: int, height: int, fps: float, stop: threading.Event) -> None:
        self._width, self._height = width, height
        self._period = 1.0 / fps
        self._stop = stop
        self._next: Optional[float] = None
        self._index = 0

    def describe(self) -> str:
        return "test pattern"

    def is_open(self) -> bool:
        return True

    def open(self) -> bool:
        return True

    def grab(self) -> bool:
        now = time.monotonic()
        if self._next is None or now - self._next > self._period:
            self._next = now  # first frame, or we fell behind: restart the schedule
        elif self._stop.wait(self._next - now):
            return False
        self._next += self._period
        return True

    def retrieve(self) -> Optional[np.ndarray]:
        frame = render_test_pattern(self._index, self._width, self._height)
        self._index += 1
        return frame

    def close(self) -> None:
        pass


class CameraNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("camera_node", **kwargs)
        device_descriptor = ParameterDescriptor(description="camera index, path/URL, or GStreamer pipeline")
        if hasattr(device_descriptor, "dynamic_typing"):  # Galactic+: accept int or string
            device_descriptor.dynamic_typing = True
        # A string default, so that a pipeline from a params file has the declared type on every
        # distro (Foxy has no dynamic_typing); parse_device turns "0" into the index 0.
        device = self.declare_parameter("device", "0", device_descriptor).value
        self._width = int(self.declare_parameter("width", 640).value)
        self._height = int(self.declare_parameter("height", 480).value)
        fps = float(self.declare_parameter("fps", 30.0).value)
        compressed_fps = float(self.declare_parameter("compressed_fps", 15.0).value)
        self._jpeg_quality = int(self.declare_parameter("jpeg_quality", 85).value)
        self._frame_id = str(self.declare_parameter("frame_id", "camera_link").value)
        source = str(self.declare_parameter("source", "camera").value)

        if self._width <= 0 or self._height <= 0:
            raise ValueError("width and height must be positive, got %dx%d" % (self._width, self._height))
        if fps <= 0 or compressed_fps < 0:
            raise ValueError("fps must be > 0 and compressed_fps >= 0, got %s and %s" % (fps, compressed_fps))
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100], got %d" % self._jpeg_quality)
        if source not in SOURCES:
            raise ValueError("source must be one of %s, got %r" % (SOURCES, source))

        # Reliable so that default-QoS subscribers and ros2 bag match; a small depth bounds the
        # memory a slow reader can pin. Best-effort subscribers (policy_runner) match as well.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=5, reliability=QoSReliabilityPolicy.RELIABLE
        )
        self._raw_pub = self.create_publisher(Image, RAW_TOPIC, qos)
        self._compressed_pub = self.create_publisher(CompressedImage, COMPRESSED_TOPIC, qos)
        tolerance = 0.5 / fps
        self._raw_limiter = RateLimiter(fps, tolerance)
        self._compressed_limiter = RateLimiter(compressed_fps, tolerance)
        self._size_warned = False

        self._stop = threading.Event()
        if source == "test_pattern":
            self._source = SyntheticSource(self._width, self._height, fps, self._stop)
        else:
            self._source = OpenCvSource(parse_device(device), self._width, self._height, fps)
        self.get_logger().info(
            "%s -> %s (%dx%d @ %.1f Hz), %s (jpeg q%d @ %.1f Hz)"
            % (
                self._source.describe(),
                RAW_TOPIC,
                self._width,
                self._height,
                fps,
                COMPRESSED_TOPIC,
                self._jpeg_quality,
                compressed_fps,
            )
        )
        self._thread = threading.Thread(target=self._capture_loop, name="camera_capture", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        log = self.get_logger()
        failures = 0
        backoff = REOPEN_BACKOFF_MIN_S
        while not self._stop.is_set():
            if not self._source.is_open():
                if not self._source.open():
                    log.warning(
                        "cannot open %s; retrying in %.1f s" % (self._source.describe(), backoff),
                        throttle_duration_sec=LOG_THROTTLE_S,
                    )
                    self._stop.wait(backoff)
                    backoff = min(2.0 * backoff, REOPEN_BACKOFF_MAX_S)
                    continue
                log.info("opened %s" % self._source.describe())

            frame = None
            if self._source.grab():
                stamp = self.get_clock().now()
                mono = time.monotonic()
                frame = self._source.retrieve()
            if self._stop.is_set():
                break
            if frame is None:
                failures += 1
                log.warning(
                    "frame read failed (%d in a row)" % failures, throttle_duration_sec=LOG_THROTTLE_S
                )
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    log.warning(
                        "reopening %s after %d failed reads" % (self._source.describe(), failures),
                        throttle_duration_sec=LOG_THROTTLE_S,
                    )
                    self._source.close()
                    failures = 0
                    self._stop.wait(backoff)
                    backoff = min(2.0 * backoff, REOPEN_BACKOFF_MAX_S)
                else:
                    self._stop.wait(READ_RETRY_S)
                continue

            failures = 0
            backoff = REOPEN_BACKOFF_MIN_S
            try:
                self._publish(frame, stamp, mono)
            except Exception as exc:  # keep capturing; a shut-down context ends the loop
                if not self.context.ok():
                    break
                log.error("publishing frame failed: %s" % exc, throttle_duration_sec=LOG_THROTTLE_S)
        self._source.close()

    def _publish(self, frame: np.ndarray, stamp: Time, mono: float) -> None:
        send_raw = self._raw_limiter.ready(mono) and self._raw_pub.get_subscription_count() > 0
        send_compressed = (
            self._compressed_limiter.ready(mono) and self._compressed_pub.get_subscription_count() > 0
        )
        if not (send_raw or send_compressed):
            return
        frame = self._normalise(frame)
        header = Header(stamp=stamp.to_msg(), frame_id=self._frame_id)
        if send_raw:
            msg = numpy_to_image(frame, "bgr8")
            msg.header = header
            self._raw_pub.publish(msg)
        if send_compressed:
            msg = numpy_to_compressed(frame, "jpeg", self._jpeg_quality)
            msg.header = header
            self._compressed_pub.publish(msg)

    def _normalise(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape[:2] != (self._height, self._width):
            if not self._size_warned:
                self.get_logger().warning(
                    "capture delivers %dx%d; resizing to %dx%d"
                    % (frame.shape[1], frame.shape[0], self._width, self._height)
                )
                self._size_warned = True
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame

    def destroy_node(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    try:
        node = CameraNode()
    except (ValueError, ParameterException) as exc:
        get_logger("camera_node").fatal("cannot start: %s" % exc)
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
