"""ArUco helpers on synthetic camera frames with a known marker pose.

Runs on the host venv (OpenCV >= 4.7, new API) and under colcon in the Foxy / Humble / Jazzy
containers (OpenCV 4.2 / 4.5 / 4.6, old API), so both code paths are exercised.
"""

import math

import cv2
import numpy as np
import pytest

from jbl_nodes import aruco_utils as au
from jbl_nodes.judge_logic import range_from_marker_area

HFOV = 1.396
W, H = 640, 480


def view(x, z, yaw=0.0, marker_id=7, y=0.0, size=(W, H), jpeg_quality=85):
    """Camera frame of a 0.10 m marker centred at (x, y, z) in the camera frame, JPEG round trip."""
    img = au.render_marker_view(marker_id, au.marker_corners_camera((x, y, z), yaw), size, HFOV)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        assert ok
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def expected_bearing(x, z):
    return math.degrees(math.atan2(-x, z))


def test_api_selection_matches_the_installed_opencv():
    major, minor = (int(p) for p in cv2.__version__.split(".")[:2])
    assert au.NEW_API == ((major, minor) >= (4, 7))
    assert au.NEW_API == hasattr(cv2.aruco, "ArucoDetector")


def test_generate_marker_is_a_decodable_binary_square():
    img = au.generate_marker(7, 120)
    assert img.shape == (120, 120) and img.dtype == np.uint8
    assert set(np.unique(img)) <= {0, 255}
    cell = 120 // 6  # 4x4 bits + 1-bit black border on each side
    assert (img[:cell] == 0).all() and (img[:, :cell] == 0).all()
    assert (img[-cell:] == 0).all() and (img[:, -cell:] == 0).all()
    assert not np.array_equal(img, au.generate_marker(8, 120))
    padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    found = au.MarkerDetector().detect(padded)
    assert [d.marker_id for d in found] == [7]
    corners = found[0].corners
    assert corners[0] == pytest.approx([39.5, 39.5], abs=1.0)  # top-left, pixel-edge coordinates
    assert corners[2] == pytest.approx([159.5, 159.5], abs=1.0)  # bottom-right
    assert found[0].center == pytest.approx((99.5, 99.5), abs=0.5)


def test_bearing_pinhole_model_and_sign():
    cx = (W - 1) / 2.0
    assert au.bearing_deg(cx, W, HFOV) == pytest.approx(0.0)
    # The image edges are at -0.5 and W - 0.5: half the HFOV either way, left positive.
    assert au.bearing_deg(-0.5, W, HFOV) == pytest.approx(math.degrees(HFOV / 2))
    assert au.bearing_deg(W - 0.5, W, HFOV) == pytest.approx(-math.degrees(HFOV / 2))
    assert au.bearing_deg(100.0, W, HFOV) > 0.0 > au.bearing_deg(500.0, W, HFOV)
    f = au.focal_length_px(W, HFOV)
    assert au.bearing_deg(cx - f, W, HFOV) == pytest.approx(45.0)
    # resolution independent: the same relative column gives the same bearing
    assert au.bearing_deg(79.5, 320, HFOV) == pytest.approx(au.bearing_deg(159.5, W, HFOV))


def test_area_and_center_of_quads():
    square = np.array([[10, 20], [110, 20], [110, 120], [10, 120]], dtype=float)
    assert au.polygon_area(square) == pytest.approx(10000.0)
    assert au.polygon_area(square[::-1]) == pytest.approx(10000.0)
    assert au.area_fraction(square, W, H) == pytest.approx(10000.0 / (W * H))
    assert au.marker_center(square) == pytest.approx((60.0, 70.0))
    # Perspective: the diagonal intersection, not the corner mean, is the centre's image.
    corners = au.marker_corners_camera((0.0, 0.0, 0.4), yaw_rad=0.9)
    quad = au.project_points(corners, W, H, HFOV)
    true_center = au.project_points(corners.mean(axis=0, keepdims=True), W, H, HFOV)[0]
    assert au.marker_center(quad) == pytest.approx(tuple(true_center), abs=1e-6)
    assert abs(quad[:, 0].mean() - true_center[0]) > 1.0


def in_view(x, z, yaw, margin_px=8.0):
    """The marker's corners all lie at least margin_px inside the image."""
    uv = au.project_points(au.marker_corners_camera((x, 0.0, z), yaw), W, H, HFOV)
    return bool((uv >= margin_px - 0.5).all() and (uv[:, 0] <= W - 0.5 - margin_px).all())


@pytest.mark.parametrize("z", [0.2, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("yaw_deg", [-30.0, 0.0, 30.0])
def test_bearing_from_detection_matches_geometry(z, yaw_deg):
    """Bearing within 1.5 deg across the view, for every pose where the whole marker is in view."""
    observer = au.MarkerObserver(7, HFOV)
    yaw = math.radians(yaw_deg)
    half_width = z * math.tan(0.5 * HFOV)
    poses = [float(x) for x in np.linspace(-half_width, half_width, 21) if in_view(float(x), z, yaw)]
    assert len(poses) >= 8
    for x in poses:
        obs = observer.observe(view(x, z, yaw))
        assert obs is not None, "marker at x=%.3f z=%.2f yaw=%s not detected" % (x, z, yaw_deg)
        assert obs.marker_id == 7
        assert obs.bearing_deg == pytest.approx(expected_bearing(x, z), abs=1.5), (x, z, yaw_deg)
        if abs(x) > 0.02:
            assert math.copysign(1.0, obs.bearing_deg) == math.copysign(1.0, -x), "left must be positive"


def test_area_fraction_tracks_distance():
    observer = au.MarkerObserver(7, HFOV)
    for z in (0.217, 0.4, 1.0):
        obs = observer.observe(view(0.0, z))
        side_px = 0.10 * au.focal_length_px(W, HFOV) / z
        # ArUco puts corners on the outermost black pixels, half a pixel inside the true edges.
        assert math.sqrt(obs.area_frac * W * H) == pytest.approx(side_px, abs=1.5)
        assert range_from_marker_area(obs.area_frac, W, H, HFOV) - 0.033 == pytest.approx(z, rel=0.05)


def test_observer_ignores_other_ids_and_picks_the_largest_copy():
    observer = au.MarkerObserver(7, HFOV)
    assert observer.observe(view(0.0, 0.5, marker_id=3)) is None
    assert observer.observe(np.full((H, W, 3), 128, dtype=np.uint8)) is None

    frame = au.render_marker_view(3, au.marker_corners_camera((0.03, 0.0, 0.35)), (W, H), HFOV)
    frame = au.render_marker_view(7, au.marker_corners_camera((0.3, 0.05, 0.9)), hfov_rad=HFOV, frame=frame)
    frame = au.render_marker_view(7, au.marker_corners_camera((-0.17, 0.0, 0.5)), hfov_rad=HFOV, frame=frame)
    detections = au.MarkerDetector().detect(frame)
    assert sorted(d.marker_id for d in detections) == [3, 7, 7]
    obs = observer.observe(frame)
    assert obs.bearing_deg == pytest.approx(expected_bearing(-0.17, 0.5), abs=1.5)


def test_grayscale_small_and_dim_frames():
    observer = au.MarkerObserver(7, HFOV)
    small = view(0.1, 0.6, size=(320, 240))
    obs = observer.observe(small)
    assert obs.bearing_deg == pytest.approx(expected_bearing(0.1, 0.6), abs=1.5)
    gray = cv2.cvtColor(view(-0.1, 0.6), cv2.COLOR_BGR2GRAY)
    assert observer.observe(gray).bearing_deg == pytest.approx(expected_bearing(-0.1, 0.6), abs=1.5)
    dim = (view(0.0, 0.8).astype(np.float32) * 0.35 + 10).astype(np.uint8)
    assert observer.observe(dim) is not None


def test_render_keeps_a_face_behind_the_camera_out():
    behind = au.render_marker_view(7, au.marker_corners_camera((0.0, 0.0, 0.03), yaw_rad=1.4), (W, H), HFOV)
    assert (behind == 90).all()
    with pytest.raises(ValueError):
        au.MarkerObserver(7, 0.0)
