"""imgconv: raw and compressed image conversion (runs on the host without ROS)."""

import array
import importlib.util

import numpy as np
import pytest

from jbl_nodes import imgconv


class FakeImage(object):
    def __init__(self, height=0, width=0, encoding="", step=0, data=b""):
        self.height, self.width, self.encoding, self.step, self.data = height, width, encoding, step, data
        self.is_bigendian = 0


class FakeCompressed(object):
    def __init__(self, fmt="", data=b""):
        self.format, self.data = fmt, data


def _frame(h=48, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _smooth_frame(h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.dstack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)]).astype(np.uint8)


@pytest.mark.parametrize("encoding", ["bgr8", "rgb8"])
def test_color_round_trip(encoding):
    img = _frame()
    msg = imgconv.numpy_to_image(img, encoding, msg=FakeImage())
    assert (msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian) == (48, 64, 64 * 3, encoding, 0)
    assert isinstance(msg.data, array.array) and msg.data.typecode == "B"
    np.testing.assert_array_equal(imgconv.image_to_numpy(msg), img)


@pytest.mark.parametrize("shape", [(30, 40), (30, 40, 1)])
def test_mono_round_trip(shape):
    img = np.random.default_rng(1).integers(0, 256, size=shape, dtype=np.uint8)
    msg = imgconv.numpy_to_image(img, "mono8", msg=FakeImage())
    assert msg.step == 40
    np.testing.assert_array_equal(imgconv.image_to_numpy(msg), img.reshape(30, 40))


@pytest.mark.parametrize("encoding,channels", [("bgr8", 3), ("rgb8", 3), ("mono8", 1)])
def test_row_padding_is_skipped(encoding, channels):
    h, w, pad = 5, 7, 11
    img = np.random.default_rng(2).integers(0, 256, size=(h, w, channels), dtype=np.uint8)
    rows = np.full((h, w * channels + pad), 0xAB, dtype=np.uint8)
    rows[:, : w * channels] = img.reshape(h, w * channels)
    msg = FakeImage(h, w, encoding, w * channels + pad, rows.tobytes() + b"\x00" * 3)  # trailing junk too
    out = imgconv.image_to_numpy(msg)
    np.testing.assert_array_equal(out, img if channels == 3 else img[:, :, 0])


def test_data_as_list_and_bytes():
    img = _frame(4, 5)
    for data in (img.tobytes(), list(img.tobytes()), bytearray(img.tobytes())):
        np.testing.assert_array_equal(imgconv.image_to_numpy(FakeImage(4, 5, "bgr8", 15, data)), img)


def test_upper_case_encoding_is_accepted():
    img = _frame(4, 5)
    np.testing.assert_array_equal(imgconv.image_to_numpy(FakeImage(4, 5, "BGR8", 15, img.tobytes())), img)


def test_image_to_bgr_converts_rgb_and_mono():
    bgr = _frame()
    rgb_msg = imgconv.numpy_to_image(bgr[:, :, ::-1], "rgb8", msg=FakeImage())
    np.testing.assert_array_equal(imgconv.image_to_bgr(rgb_msg), bgr)

    gray = bgr[:, :, 0].copy()
    mono = imgconv.image_to_bgr(imgconv.numpy_to_image(gray, "mono8", msg=FakeImage()))
    assert mono.shape == (48, 64, 3)
    for c in range(3):
        np.testing.assert_array_equal(mono[:, :, c], gray)

    bgr_msg = imgconv.numpy_to_image(bgr, "bgr8", msg=FakeImage())
    np.testing.assert_array_equal(imgconv.image_to_bgr(bgr_msg), bgr)


def test_padded_bgr_view_is_usable_by_opencv():
    import cv2

    h, w, pad = 6, 8, 4
    img = _frame(h, w)
    rows = np.zeros((h, w * 3 + pad), dtype=np.uint8)
    rows[:, : w * 3] = img.reshape(h, w * 3)
    view = imgconv.image_to_bgr(FakeImage(h, w, "bgr8", w * 3 + pad, rows.tobytes()))
    np.testing.assert_array_equal(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), img[:, :, ::-1])


@pytest.mark.parametrize(
    "msg,match",
    [
        (FakeImage(2, 2, "bgr16", 12, b"\x00" * 24), "unsupported encoding"),
        (FakeImage(2, 2, "bgr8", 5, b"\x00" * 12), "step 5 is smaller"),
        (FakeImage(2, 2, "bgr8", 6, b"\x00" * 11), "data has 11 bytes"),
    ],
)
def test_invalid_raw_messages(msg, match):
    with pytest.raises(ValueError, match=match):
        imgconv.image_to_numpy(msg)


@pytest.mark.parametrize(
    "arr,encoding,match",
    [
        (np.zeros((2, 2, 3), np.uint16), "bgr8", "uint8"),
        (np.zeros((2, 2), np.uint8), "bgr8", "does not match"),
        (np.zeros((2, 2, 4), np.uint8), "rgb8", "does not match"),
        (np.zeros((2, 2, 3), np.uint8), "mono8", "does not match"),
        (np.zeros((2, 2, 3), np.uint8), "yuv422", "unsupported encoding"),
    ],
)
def test_invalid_arrays(arr, encoding, match):
    with pytest.raises(ValueError, match=match):
        imgconv.numpy_to_image(arr, encoding, msg=FakeImage())


def test_jpeg_round_trip_is_close():
    img = _smooth_frame()
    msg = imgconv.numpy_to_compressed(img, "jpeg", 90, msg=FakeCompressed())
    assert msg.format == "jpeg"
    assert isinstance(msg.data, array.array)
    assert bytes(msg.data[:2]) == b"\xff\xd8"  # JPEG SOI marker
    out = imgconv.compressed_to_bgr(msg)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 3.0


def test_jpeg_quality_changes_size():
    img = _frame(120, 160)
    small = imgconv.numpy_to_compressed(img, "jpeg", 20, msg=FakeCompressed())
    large = imgconv.numpy_to_compressed(img, "jpeg", 95, msg=FakeCompressed())
    assert len(small.data) < len(large.data)


def test_png_round_trip_is_exact():
    img = _frame()
    msg = imgconv.numpy_to_compressed(img, "png", msg=FakeCompressed())
    assert msg.format == "png"
    np.testing.assert_array_equal(imgconv.compressed_to_bgr(msg), img)


def test_mono_compressed_decodes_to_bgr():
    gray = _frame()[:, :, 1].copy()
    out = imgconv.compressed_to_bgr(imgconv.numpy_to_compressed(gray, "png", msg=FakeCompressed()))
    assert out.shape == (48, 64, 3)
    np.testing.assert_array_equal(out[:, :, 2], gray)


def test_compressed_errors():
    img = _frame()
    with pytest.raises(ValueError, match="unsupported compressed format"):
        imgconv.numpy_to_compressed(img, "webp", msg=FakeCompressed())
    with pytest.raises(ValueError, match="quality"):
        imgconv.numpy_to_compressed(img, "jpeg", 0, msg=FakeCompressed())
    with pytest.raises(ValueError, match="could not decode"):
        imgconv.compressed_to_bgr(FakeCompressed("jpeg", b"not an image"))


@pytest.mark.skipif(importlib.util.find_spec("sensor_msgs") is None, reason="needs ROS sensor_msgs")
def test_real_ros_messages():
    from sensor_msgs.msg import CompressedImage, Image

    img = _frame(480, 640)
    msg = imgconv.numpy_to_image(img, "bgr8")
    assert isinstance(msg, Image)
    np.testing.assert_array_equal(imgconv.image_to_numpy(msg), img)
    cmsg = imgconv.numpy_to_compressed(img, "jpeg", 85)
    assert isinstance(cmsg, CompressedImage) and cmsg.format == "jpeg"
    assert imgconv.compressed_to_bgr(cmsg).shape == img.shape
