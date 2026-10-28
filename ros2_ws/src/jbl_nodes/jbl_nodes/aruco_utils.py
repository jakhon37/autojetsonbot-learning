"""ArUco DICT_4X4_50 detection, marker images and pinhole geometry for the docking marker (no ROS).

Works with both OpenCV ArUco APIs:
    OpenCV <= 4.6 (Foxy 4.2, Humble 4.5, Jazzy 4.6): cv2.aruco.detectMarkers, Dictionary_get,
        DetectorParameters_create, drawMarker
    OpenCV >= 4.7 (host venv): cv2.aruco.ArucoDetector, getPredefinedDictionary,
        DetectorParameters, generateImageMarker

Image coordinates follow OpenCV: pixel centres at integer coordinates, so the optical centre
of an ideal W x H camera is ((W - 1) / 2, (H - 1) / 2) and the image spans [-0.5, W - 0.5].
The horizontal field of view spans the full image width. Camera frame: x right, y down,
z forward (optical frame).

render_marker_view() draws what an ideal camera sees of the marker; tests and the kinematic
simulator use it to produce frames with a known bearing.
"""

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from jbl_nodes.judge_logic import MarkerObservation

if not hasattr(cv2, "aruco"):
    raise ImportError(
        "this OpenCV build has no aruco module: install python3-opencv (Ubuntu) or "
        "opencv-contrib-python / opencv-python >= 4.7"
    )

DICT_4X4_50 = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_ID = 7
DEFAULT_HFOV_RAD = 1.396
NEW_API = hasattr(cv2.aruco, "ArucoDetector")  # OpenCV >= 4.7


def get_dictionary(dictionary_id: int = DICT_4X4_50):
    if NEW_API:
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def generate_marker(
    marker_id: int, side_px: int, dictionary_id: int = DICT_4X4_50, border_bits: int = 1
) -> np.ndarray:
    """side_px x side_px uint8 marker image (black border included, no white quiet zone).

    A detector needs white around the marker: print it on a white face, or pad the image.
    """
    dictionary = get_dictionary(dictionary_id)
    if NEW_API:
        return cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(side_px), borderBits=border_bits)
    return cv2.aruco.drawMarker(dictionary, int(marker_id), int(side_px), borderBits=border_bits)


class MarkerDetection(NamedTuple):
    marker_id: int
    corners: np.ndarray  # 4x2 float64 pixels: top-left, top-right, bottom-right, bottom-left as printed
    center: Tuple[float, float]  # image of the marker's centre (diagonal intersection)


def marker_center(corners: np.ndarray) -> Tuple[float, float]:
    """Intersection of the quad's diagonals: the exact image of a planar square's centre under
    perspective (the corner mean is biased towards the nearer edge)."""
    q = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    d1 = q[2] - q[0]
    d2 = q[3] - q[1]
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:  # degenerate quad
        c = q.mean(axis=0)
        return float(c[0]), float(c[1])
    r = q[1] - q[0]
    t = (r[0] * d2[1] - r[1] * d2[0]) / denom
    c = q[0] + t * d1
    return float(c[0]), float(c[1])


def polygon_area(corners: np.ndarray) -> float:
    """Area (px^2) of a simple polygon given as Nx2 vertices (shoelace)."""
    q = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def area_fraction(corners: np.ndarray, image_width: int, image_height: int) -> float:
    """Marker area as a fraction of the image area."""
    return polygon_area(corners) / float(image_width * image_height)


def focal_length_px(image_width: int, hfov_rad: float) -> float:
    return 0.5 * image_width / math.tan(0.5 * hfov_rad)


def bearing_deg(center_x: float, image_width: int, hfov_rad: float = DEFAULT_HFOV_RAD) -> float:
    """Horizontal angle (deg) of an image column from the optical axis; positive = left of centre.

    Pinhole model: bearing = atan((cx - x) / f), cx = (W - 1) / 2, f = (W / 2) / tan(HFOV / 2).
    """
    cx = 0.5 * (image_width - 1)
    return math.degrees(math.atan((cx - center_x) / focal_length_px(image_width, hfov_rad)))


class MarkerDetector(object):
    """All markers of one dictionary in a BGR or grayscale image."""

    def __init__(self, dictionary_id: int = DICT_4X4_50) -> None:
        self._dictionary = get_dictionary(dictionary_id)
        if NEW_API:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, cv2.aruco.DetectorParameters())
        else:
            self._params = cv2.aruco.DetectorParameters_create()

    def detect(self, image: np.ndarray) -> List[MarkerDetection]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if NEW_API:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary, parameters=self._params)
        if ids is None:
            return []
        detections = []
        for quad, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
            detections.append(MarkerDetection(int(marker_id), q, marker_center(q)))
        return detections


class MarkerObserver(object):
    """One marker id -> MarkerObservation (bearing and area) per frame."""

    def __init__(
        self,
        marker_id: int = DEFAULT_MARKER_ID,
        hfov_rad: float = DEFAULT_HFOV_RAD,
        dictionary_id: int = DICT_4X4_50,
    ) -> None:
        if not 0.0 < hfov_rad < math.pi:
            raise ValueError("hfov_rad must be in (0, pi), got %s" % hfov_rad)
        self.marker_id = int(marker_id)
        self.hfov_rad = float(hfov_rad)
        self._detector = MarkerDetector(dictionary_id)

    def observe(self, image: np.ndarray) -> Optional[MarkerObservation]:
        """The marker in this frame, or None. With several copies in view, the largest wins."""
        height, width = image.shape[:2]
        matches = [d for d in self._detector.detect(image) if d.marker_id == self.marker_id]
        if not matches:
            return None
        best = max(matches, key=lambda d: polygon_area(d.corners))
        return MarkerObservation(
            best.marker_id,
            bearing_deg(best.center[0], width, self.hfov_rad),
            area_fraction(best.corners, width, height),
        )


# ------------------------------------------------------------------ synthetic views


def project_points(
    points_cam: Sequence[Sequence[float]], image_width: int, image_height: int, hfov_rad: float
) -> np.ndarray:
    """Nx3 camera-frame points (x right, y down, z forward, z > 0) -> Nx2 pixel coordinates of an
    ideal pinhole camera with square pixels and no distortion."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    f = focal_length_px(image_width, hfov_rad)
    u = 0.5 * (image_width - 1) + f * p[:, 0] / p[:, 2]
    v = 0.5 * (image_height - 1) + f * p[:, 1] / p[:, 2]
    return np.stack([u, v], axis=1)


def marker_corners_camera(
    center_cam: Sequence[float], yaw_rad: float = 0.0, side_m: float = 0.10
) -> np.ndarray:
    """4x3 camera-frame corners (top-left, top-right, bottom-right, bottom-left as printed) of an
    upright marker centred at center_cam, facing the camera when yaw_rad = 0 and turned by
    yaw_rad about the camera's vertical (y) axis otherwise."""
    h = 0.5 * side_m
    local = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.asarray(center_cam, dtype=np.float64).reshape(1, 3) + local @ rot_y.T


def render_marker_view(
    marker_id: int,
    corners_cam: np.ndarray,
    image_size: Tuple[int, int] = (640, 480),
    hfov_rad: float = DEFAULT_HFOV_RAD,
    face_scale: float = 2.0,
    background: Tuple[int, int, int] = (90, 90, 90),
    dictionary_id: int = DICT_4X4_50,
    marker_px: int = 240,
    frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BGR frame of a marker printed on a white face, as an ideal pinhole camera sees it.

    corners_cam: 4x3 marker corners in the camera frame (see marker_corners_camera). The
    white face is the marker scaled by face_scale about its centre (2.0 = half a marker side
    of white on each side). It is drawn over a copy of ``frame`` (HxWx3 uint8, e.g. to put
    several markers in one view) or over a uniform ``background`` of image_size. If any face
    corner is not at least 1 cm in front of the camera the face is not drawn: callers must keep
    it entirely in front, which holds for a robot facing the box. Edges are anti-aliased.
    """
    if frame is None:
        width, height = image_size
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:] = background
    else:
        height, width = frame.shape[:2]
        frame = frame.copy()
    corners = np.asarray(corners_cam, dtype=np.float64).reshape(4, 3)
    centre = corners.mean(axis=0)
    face = centre + face_scale * (corners - centre)
    if np.any(face[:, 2] <= 0.01):
        return frame

    pad = int(round(0.5 * (face_scale - 1.0) * marker_px))
    size = marker_px + 2 * pad
    canvas = np.full((size, size), 255, dtype=np.uint8)
    canvas[pad : pad + marker_px, pad : pad + marker_px] = generate_marker(
        marker_id, marker_px, dictionary_id
    )
    # Canvas pixel edges (not centres) map to the face corners, so the marker centre lands on
    # the projected centre exactly.
    src = np.array(
        [[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]], dtype=np.float32
    )
    dst = project_points(face, width, height, hfov_rad).astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, homography, (width, height), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(
        np.ones((size, size), dtype=np.float32), homography, (width, height), flags=cv2.INTER_LINEAR
    )[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32)[:, :, None] * alpha
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)
