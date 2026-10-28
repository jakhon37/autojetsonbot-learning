"""sensor_msgs/Image and CompressedImage <-> numpy without cv_bridge.

Supported raw encodings: bgr8, rgb8, mono8 (row padding via ``step`` is honoured).
Supported compressed formats: jpeg, png.

Message classes are imported lazily, so the module (and its tests) work without a ROS
installation: every builder accepts ``msg=`` with any object that has the same fields.
Headers are left to the caller.
"""

import array
from typing import Any, Optional

import cv2
import numpy as np

CHANNELS = {"bgr8": 3, "rgb8": 3, "mono8": 1}
COMPRESSED_FORMATS = ("jpeg", "png")


def _new_image_msg() -> Any:
    from sensor_msgs.msg import Image

    return Image()


def _new_compressed_msg() -> Any:
    from sensor_msgs.msg import CompressedImage

    return CompressedImage()


def _as_uint8_buffer(data: Any) -> np.ndarray:
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except TypeError:  # a plain list of ints
        return np.asarray(data, dtype=np.uint8)


def _to_msg_data(arr: np.ndarray) -> array.array:
    # rclpy's uint8[] setter takes array('B') as is; any other sequence is validated element
    # by element in Python, which costs tens of milliseconds for a 640x480 frame.
    return array.array("B", arr.tobytes())


def image_to_numpy(msg: Any) -> np.ndarray:
    """Image -> uint8 array, HxWx3 for bgr8/rgb8 (channel order unchanged) or HxW for mono8.

    The result may share memory with ``msg.data``; copy it if the message is reused.
    """
    encoding = msg.encoding.lower()
    if encoding not in CHANNELS:
        raise ValueError("unsupported encoding %r (supported: %s)" % (msg.encoding, ", ".join(CHANNELS)))
    channels = CHANNELS[encoding]
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError("step %d is smaller than width * channels = %d" % (step, row_bytes))
    buf = _as_uint8_buffer(msg.data)
    if buf.size < height * step:
        raise ValueError(
            "data has %d bytes, expected at least height * step = %d" % (buf.size, height * step)
        )
    rows = buf[: height * step].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        return rows
    return rows.reshape(height, width, channels)


def image_to_bgr(msg: Any) -> np.ndarray:
    """Image in any supported encoding -> HxWx3 uint8 BGR."""
    arr = image_to_numpy(msg)
    encoding = msg.encoding.lower()
    if encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR)
    return arr


def numpy_to_image(arr: np.ndarray, encoding: str = "bgr8", msg: Optional[Any] = None) -> Any:
    """uint8 HxWx3 (bgr8/rgb8) or HxW / HxWx1 (mono8) array -> Image (header untouched)."""
    encoding = encoding.lower()
    if encoding not in CHANNELS:
        raise ValueError("unsupported encoding %r (supported: %s)" % (encoding, ", ".join(CHANNELS)))
    if arr.dtype != np.uint8:
        raise ValueError("expected a uint8 array, got %s" % arr.dtype)
    channels = CHANNELS[encoding]
    if channels == 1 and arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    expected_ndim = 2 if channels == 1 else 3
    if arr.ndim != expected_ndim or (channels == 3 and arr.shape[2] != 3):
        raise ValueError("array of shape %s does not match encoding %s" % (arr.shape, encoding))
    msg = _new_image_msg() if msg is None else msg
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels
    msg.data = _to_msg_data(arr)
    return msg


def compressed_to_bgr(msg: Any) -> np.ndarray:
    """CompressedImage (jpeg or png) -> HxWx3 uint8 BGR."""
    img = cv2.imdecode(_as_uint8_buffer(msg.data), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "could not decode compressed image (format %r, %d bytes)" % (msg.format, len(msg.data))
        )
    return img


def numpy_to_compressed(
    img: np.ndarray, fmt: str = "jpeg", quality: int = 85, msg: Optional[Any] = None
) -> Any:
    """BGR HxWx3 or mono HxW uint8 array -> CompressedImage (header untouched).

    ``quality`` is the JPEG quality (1-100); for png it is ignored and the OpenCV default
    compression level is used. ``format`` is set to the plain "jpeg"/"png" string, which
    image_transport's compressed plugin and the bag tools all accept.
    """
    fmt = fmt.lower()
    if fmt == "jpeg":
        if not 1 <= int(quality) <= 100:
            raise ValueError("jpeg quality must be in [1, 100], got %r" % quality)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    elif fmt == "png":
        ok, buf = cv2.imencode(".png", img)
    else:
        raise ValueError(
            "unsupported compressed format %r (supported: %s)" % (fmt, ", ".join(COMPRESSED_FORMATS))
        )
    if not ok:
        raise ValueError("OpenCV failed to encode a %s image of shape %s" % (fmt, img.shape))
    msg = _new_compressed_msg() if msg is None else msg
    msg.format = fmt
    msg.data = _to_msg_data(buf)
    return msg
