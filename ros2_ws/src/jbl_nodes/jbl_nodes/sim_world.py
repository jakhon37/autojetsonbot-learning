"""Kinematic simulator of the docking task: robot, room, marker box, lidar and camera (no ROS).

World frame (metres, radians, z up): a room of room_length x room_width centred on the origin
(walls at x = +-2, y = +-1.5 by default) and a box standing on the floor. The box is placed by
its *front face*: (box_x, box_y) is the centre of the face's bottom edge and box_yaw the
direction the face looks (its outward normal). ArUco DICT_4X4_50 id 7 (0.10 m) is printed on a
white sheet centred on that face at marker_height, which equals the camera height so a robot
facing the marker sees it on the image's middle row.

Robot frame (base_link): x forward, y left. A unicycle driven by (v, w) commands, integrated
every ``dt`` (50 Hz) with acceleration limits. The footprint is a 0.194 x 0.178 m rectangle
centred footprint_center_x ahead of base_link (0 by default, which puts the front bumper
0.097 m ahead as docs/INTERFACES.md states). A step that would make the footprint overlap the
box or leave the room is cut short at the point of contact, stops the robot and latches
``collided`` until the next reset.

Sensors:
    scan()    360-beam lidar from (lidar_x, 0) in the robot frame, beam i at angle
              angle_min + i * 2 pi / 360 (0 = forward, counter-clockwise positive). Like the
              RPLidar driver, a return closer than range_min (or none at all) reads +inf.
    render()  640x480 BGR frame of a pinhole camera at (camera_x, 0, camera_height) looking
              along the robot's x axis with horizontal FOV 1.396 rad: flat-shaded walls, floor,
              ceiling and box, the marker sheet texture-mapped with an exact homography.

randomize=True adds the small disturbances a policy should be robust to: per-episode wheel
speed bias and start pose jitter, per-step wheel slip, lidar range noise, and per-episode
brightness plus per-frame pixel noise. randomize=False is exact and deterministic.

Python 3.8 compatible; numpy, OpenCV and PyYAML only.
"""

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from jbl_nodes.aruco_utils import (
    DEFAULT_HFOV_RAD,
    DEFAULT_MARKER_ID,
    DICT_4X4_50,
    focal_length_px,
    generate_marker,
)

TWO_PI = 2.0 * math.pi
MARKER_BITS = 6  # DICT_4X4_50: 4 data bits + a 1-bit black border on each side
_EPS = 1e-9


class Pose2D(NamedTuple):
    x: float
    y: float
    yaw: float


class RobotConfig(NamedTuple):
    wheel_separation: float = 0.212
    max_linear_accel: float = 1.0  # m/s^2
    max_angular_accel: float = 2.0  # rad/s^2
    footprint_length: float = 0.194
    footprint_width: float = 0.178
    footprint_center_x: float = 0.0  # footprint centre ahead of base_link
    lidar_x: float = 0.064
    camera_x: float = 0.10
    camera_height: float = 0.106  # base_link is 0.056 above the floor, camera 0.05 above that

    @property
    def bumper_x(self) -> float:
        return self.footprint_center_x + 0.5 * self.footprint_length


class WorldConfig(NamedTuple):
    room_length: float = 4.0  # along x
    room_width: float = 3.0  # along y
    wall_height: float = 1.0
    box_x: float = 1.5  # centre of the front face's bottom edge
    box_y: float = 0.0
    box_yaw: float = math.pi  # direction the front face (and marker) looks
    box_width: float = 0.30
    box_depth: float = 0.30
    box_height: float = 0.30
    marker_id: int = DEFAULT_MARKER_ID
    marker_side: float = 0.10
    marker_height: float = 0.106  # marker centre above the floor
    marker_margin_bits: int = 2  # white quiet zone around the marker, in marker bits

    @property
    def sheet_side(self) -> float:
        """Side of the white sheet the marker is printed on."""
        return self.marker_side * (MARKER_BITS + 2 * self.marker_margin_bits) / MARKER_BITS


class LidarConfig(NamedTuple):
    num_beams: int = 360
    angle_min: float = -math.pi
    range_min: float = 0.15
    range_max: float = 12.0

    @property
    def angle_increment(self) -> float:
        return TWO_PI / self.num_beams


class CameraConfig(NamedTuple):
    width: int = 640
    height: int = 480
    hfov: float = DEFAULT_HFOV_RAD
    near: float = 0.01  # m; geometry closer than this to the camera plane is clipped


class Appearance(NamedTuple):
    """Colours (BGR) and the single directional light of the flat shading."""

    floor: Tuple[int, int, int] = (105, 118, 128)
    ceiling: Tuple[int, int, int] = (215, 215, 210)
    wall: Tuple[int, int, int] = (196, 186, 176)
    box: Tuple[int, int, int] = (80, 130, 175)  # cardboard
    sheet: int = 235  # grey level of the white sheet
    ink: int = 25  # grey level of the printed marker
    light_dir: Tuple[float, float, float] = (-0.45, 0.35, 0.8)  # towards the light, world frame
    ambient: float = 0.6  # brightness of a face turned away from the light


class NoiseConfig(NamedTuple):
    """Magnitudes of the disturbances applied when randomize=True."""

    wheel_bias_std: float = 0.02  # per episode, relative wheel speed error per wheel
    wheel_slip_std: float = 0.03  # per step, relative wheel speed error per wheel
    scan_noise_std: float = 0.005  # m, per beam
    pose_jitter_m: float = 0.03  # uniform +- on x and y of a start pose
    yaw_jitter_rad: float = math.radians(3.0)  # uniform +- on a start heading
    brightness_gain: Tuple[float, float] = (0.7, 1.3)  # uniform per episode
    brightness_offset: Tuple[float, float] = (-20.0, 20.0)  # grey levels, uniform per episode
    pixel_noise_std: float = 3.0  # grey levels, per pixel and frame


class StartPose(NamedTuple):
    """A start pose relative to the box front face, as marked on the floor (see start_poses.yaml)."""

    pose_id: int
    distance_m: float  # base_link to the front face plane, along the face normal
    lateral_m: float  # along the face from its centre; positive = robot's left when facing the box
    heading_deg: float  # from facing the face squarely; positive = turned left (counter-clockwise)


def wrap_angle(angle: float) -> float:
    """Wrap radians into [-pi, pi)."""
    return (angle + math.pi) % TWO_PI - math.pi


def load_start_poses(path: str) -> List[StartPose]:
    """Read start poses from a YAML file with a ``start_poses`` list (see config/start_poses.yaml).

    Raises ValueError unless the ids are exactly 0..N-1 and every number is finite.
    """
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("start_poses"), list):
        raise ValueError("%s: expected a mapping with a 'start_poses' list" % path)
    poses = []
    for i, entry in enumerate(doc["start_poses"]):
        try:
            pose = StartPose(
                pose_id=int(entry["pose_id"]),
                distance_m=float(entry["distance_m"]),
                lateral_m=float(entry["lateral_m"]),
                heading_deg=float(entry["heading_deg"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("%s: start_poses[%d] is malformed (%s)" % (path, i, exc)) from None
        if not all(math.isfinite(v) for v in pose[1:]):
            raise ValueError("%s: start_poses[%d] has a non-finite value" % (path, i))
        poses.append(pose)
    poses.sort(key=lambda p: p.pose_id)
    ids = [p.pose_id for p in poses]
    if ids != list(range(len(poses))):
        raise ValueError("%s: pose_id values must be 0..N-1 without gaps or repeats, got %s" % (path, ids))
    return poses


def start_pose_to_world(start: StartPose, world: WorldConfig = WorldConfig()) -> Pose2D:
    """World pose of base_link for a start pose given relative to the box front face."""
    nx, ny = math.cos(world.box_yaw), math.sin(world.box_yaw)
    # A robot facing the box looks along -n; its left is -n turned 90 deg counter-clockwise.
    left_x, left_y = ny, -nx
    x = world.box_x + start.distance_m * nx + start.lateral_m * left_x
    y = world.box_y + start.distance_m * ny + start.lateral_m * left_y
    return Pose2D(x, y, wrap_angle(world.box_yaw + math.pi + math.radians(start.heading_deg)))


def box_corners(world: WorldConfig) -> np.ndarray:
    """4x2 box outline on the floor, counter-clockwise, starting with the front face's right end
    as seen by a robot facing it."""
    n = np.array([math.cos(world.box_yaw), math.sin(world.box_yaw)])
    t = np.array([-n[1], n[0]])
    front = np.array([world.box_x, world.box_y])
    hw = 0.5 * world.box_width
    back = front - world.box_depth * n
    return np.array([front - hw * t, front + hw * t, back + hw * t, back - hw * t])


def room_corners(world: WorldConfig) -> np.ndarray:
    """4x2 room outline, counter-clockwise."""
    hx, hy = 0.5 * world.room_length, 0.5 * world.room_width
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])


def _to_world(pose: Pose2D, points: np.ndarray) -> np.ndarray:
    """Nx2 robot-frame points -> world frame."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.stack([pose.x + c * p[:, 0] - s * p[:, 1], pose.y + s * p[:, 0] + c * p[:, 1]], axis=1)


def _convex_polygons_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Separating axis test for two convex polygons (Nx2, Mx2); touching counts as overlap."""
    for poly in (p, q):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        pp, qq = p @ axes.T, q @ axes.T
        if np.any(pp.max(axis=0) < qq.min(axis=0)) or np.any(qq.max(axis=0) < pp.min(axis=0)):
            return False
    return True


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), _EPS), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segment_polygon_distance(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> float:
    """Shortest distance between segment ab and a convex polygon outline (0 if they overlap)."""
    if _convex_polygons_overlap(np.array([a, b]), poly):
        return 0.0
    best = math.inf
    for i in range(len(poly)):
        c, d = poly[i], poly[(i + 1) % len(poly)]
        best = min(
            best,
            _point_segment_distance(a, c, d),
            _point_segment_distance(b, c, d),
            _point_segment_distance(c, a, b),
        )
    return best


def cast_rays(
    origin: Sequence[float], angles: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from ``origin`` along each world angle to the nearest of the segments
    seg_a[k] -> seg_b[k], and the index of the segment hit (-1 and inf where none is)."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # M x 2
    e = seg_b - seg_a  # S x 2
    w = seg_a - o
    denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # cross(d, e), M x S
    t_num = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0])[None, :]  # cross(w, e)
    s_num = w[None, :, 0] * d[:, 1:2] - w[None, :, 1] * d[:, 0:1]  # cross(w, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = t_num / denom
        s = s_num / denom
    hit = (np.abs(denom) > _EPS) & (t > _EPS) & (s >= -_EPS) & (s <= 1.0 + _EPS)
    t = np.where(hit, t, np.inf)
    index = np.argmin(t, axis=1)
    dist = t[np.arange(len(index)), index]
    return dist, np.where(np.isfinite(dist), index, -1)


class _Face(NamedTuple):
    corners: np.ndarray  # Nx3 world, planar convex polygon
    normal: np.ndarray  # outward unit normal, world
    color: Tuple[float, float, float]  # shaded BGR
    antialias: bool


class CameraRenderer(object):
    """Flat-shaded pinhole view of the room, box and marker sheet.

    Faces are clipped in 3D against the near plane and a guard band around the image frustum
    before projection, so geometry that is partly behind the camera is cut exactly where it
    leaves the view instead of being mirrored, and pixel coordinates stay small. The room is
    convex and the box convex and inside it, so painting walls, then box faces turned to the
    camera, then the sheet resolves visibility without a depth buffer.
    """

    _GUARD_PX = 2.0
    _SUBPIXEL_BITS = 4

    def __init__(
        self,
        world: WorldConfig = WorldConfig(),
        robot: RobotConfig = RobotConfig(),
        camera: CameraConfig = CameraConfig(),
        appearance: Appearance = Appearance(),
    ) -> None:
        self.camera = camera
        self._robot = robot
        w, h = camera.width, camera.height
        self._f = focal_length_px(w, camera.hfov)
        self._cx, self._cy = 0.5 * (w - 1), 0.5 * (h - 1)
        self._k = np.array([[self._f, 0.0, self._cx], [0.0, self._f, self._cy], [0.0, 0.0, 1.0]])
        g = self._GUARD_PX
        # Keep n . p + d >= 0 in the camera frame (x right, y down, z forward).
        self._planes = [
            (np.array([0.0, 0.0, 1.0]), -camera.near),
            (np.array([self._f, 0.0, self._cx + g]), 0.0),
            (np.array([-self._f, 0.0, (w - 1) + g - self._cx]), 0.0),
            (np.array([0.0, self._f, self._cy + g]), 0.0),
            (np.array([0.0, -self._f, (h - 1) + g - self._cy]), 0.0),
        ]
        light = np.asarray(appearance.light_dir, dtype=np.float64)
        self._light = light / np.linalg.norm(light)
        self._ambient = appearance.ambient
        self._background = self._make_background(appearance)
        self._faces = self._make_faces(world, appearance)

        n = np.array([math.cos(world.box_yaw), math.sin(world.box_yaw), 0.0])
        self._sheet_normal = n
        self._sheet_center = np.array([world.box_x, world.box_y, world.marker_height])
        self._sheet_right = np.array([-n[1], n[0], 0.0])  # the printed marker's left-to-right
        self._sheet_down = np.array([0.0, 0.0, -1.0])
        self._sheet_side = world.sheet_side
        shade = self._shade(n)
        self._sheet_lut = np.clip(np.arange(256) * shade + 0.5, 0, 255).astype(np.uint8)
        self._textures = self._make_textures(world, appearance)

    # ------------------------------------------------------------ scene setup

    def _shade(self, normal: np.ndarray) -> float:
        return self._ambient + (1.0 - self._ambient) * max(0.0, float(np.dot(normal, self._light)))

    def _make_background(self, appearance: Appearance) -> np.ndarray:
        h, w = self.camera.height, self.camera.width
        rows = np.arange(h, dtype=np.float64)
        # The floor darkens towards the horizon, a cheap depth cue; the ceiling is uniform.
        below = np.clip((rows - self._cy) / max(h - 1 - self._cy, 1.0), 0.0, 1.0)
        floor = np.outer(0.8 + 0.2 * below, appearance.floor)
        colors = np.where((rows > self._cy)[:, None], floor, np.asarray(appearance.ceiling, dtype=np.float64))
        column = np.clip(colors + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(column[:, None, :], w, axis=1))

    def _make_faces(self, world: WorldConfig, appearance: Appearance) -> List[_Face]:
        faces = []

        def add(corners2d: np.ndarray, z0: float, z1: float, normal2d: np.ndarray, color, aa: bool) -> None:
            a, b = corners2d
            corners = np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])
            normal = np.array([normal2d[0], normal2d[1], 0.0])
            faces.append(_Face(corners, normal, self._shaded(color, normal), aa))

        room = room_corners(world)
        for i in range(4):
            a, b = room[i], room[(i + 1) % 4]
            edge = (b - a) / np.linalg.norm(b - a)
            inward = np.array([-edge[1], edge[0]])  # counter-clockwise outline: inside is on the left
            add(np.array([a, b]), 0.0, world.wall_height, inward, appearance.wall, False)

        box = box_corners(world)
        for i in range(4):
            a, b = box[i], box[(i + 1) % 4]
            edge = (b - a) / np.linalg.norm(b - a)
            outward = np.array([edge[1], -edge[0]])
            add(np.array([a, b]), 0.0, world.box_height, outward, appearance.box, True)
        top = np.column_stack([box, np.full(4, world.box_height)])
        up = np.array([0.0, 0.0, 1.0])
        faces.append(_Face(top, up, self._shaded(appearance.box, up), True))
        return faces

    def _shaded(self, color: Sequence[int], normal: np.ndarray) -> Tuple[float, float, float]:
        s = self._shade(normal)
        return tuple(float(min(255.0, c * s)) for c in color)  # type: ignore[return-value]

    def _make_textures(self, world: WorldConfig, appearance: Appearance) -> List[np.ndarray]:
        """Sheet textures at 2..32 px per marker bit, a small mip chain so that warping never
        minifies by much more than 2x (bilinear warps alias badly beyond that)."""
        textures = []
        bits = MARKER_BITS + 2 * world.marker_margin_bits
        for px_per_bit in (2, 4, 8, 16, 32):
            side = bits * px_per_bit
            sheet = np.full((side, side), appearance.sheet, dtype=np.uint8)
            marker = generate_marker(world.marker_id, MARKER_BITS * px_per_bit, DICT_4X4_50)
            m = world.marker_margin_bits * px_per_bit
            inner = sheet[m : m + marker.shape[0], m : m + marker.shape[1]]
            inner[marker < 128] = appearance.ink
            textures.append(sheet)
        return textures

    # ------------------------------------------------------------ per frame

    def camera_pose(self, pose: Pose2D) -> Tuple[np.ndarray, np.ndarray]:
        """(R, c): world -> camera rotation and the camera centre in the world."""
        cos_y, sin_y = math.cos(pose.yaw), math.sin(pose.yaw)
        forward = np.array([cos_y, sin_y, 0.0])
        left = np.array([-sin_y, cos_y, 0.0])
        rotation = np.stack([-left, np.array([0.0, 0.0, -1.0]), forward])
        centre = np.array(
            [
                pose.x + self._robot.camera_x * cos_y,
                pose.y + self._robot.camera_x * sin_y,
                self._robot.camera_height,
            ]
        )
        return rotation, centre

    def _clip(self, poly: np.ndarray) -> np.ndarray:
        """Sutherland-Hodgman against the near plane and the guard-banded frustum sides."""
        for normal, offset in self._planes:
            dist = poly @ normal + offset
            inside = dist >= 0.0
            if inside.all():
                continue
            if not inside.any():
                return poly[:0]
            out = []
            n = len(poly)
            for i in range(n):
                j = (i + 1) % n
                if inside[i]:
                    out.append(poly[i])
                if inside[i] != inside[j]:
                    t = dist[i] / (dist[i] - dist[j])
                    out.append(poly[i] + t * (poly[j] - poly[i]))
            poly = np.array(out)
        return poly

    def _project_fixed(self, poly_cam: np.ndarray) -> np.ndarray:
        """Camera-frame points (all in front) -> fixed-point pixel coordinates for cv2.fillPoly."""
        z = poly_cam[:, 2]
        uv = np.stack(
            [self._cx + self._f * poly_cam[:, 0] / z, self._cy + self._f * poly_cam[:, 1] / z], axis=1
        )
        return np.round(uv * (1 << self._SUBPIXEL_BITS)).astype(np.int32)

    def render(
        self,
        pose: Pose2D,
        gain: float = 1.0,
        offset: float = 0.0,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """BGR uint8 frame seen from robot pose ``pose``; gain/offset/noise_std adjust exposure
        and add Gaussian pixel noise (rng required when noise_std > 0)."""
        frame = self._background.copy()
        rotation, centre = self.camera_pose(pose)
        for face in self._faces:
            if np.dot(face.normal, centre - face.corners[0]) <= 0.0:
                continue  # turned away from the camera
            poly = self._clip((face.corners - centre) @ rotation.T)
            if len(poly) < 3:
                continue
            pts = self._project_fixed(poly)
            line = cv2.LINE_AA if face.antialias else cv2.LINE_8
            cv2.fillPoly(frame, [pts.reshape(-1, 1, 2)], face.color, line, self._SUBPIXEL_BITS)
        if np.dot(self._sheet_normal, centre - self._sheet_center) > 0.0:
            self._draw_sheet(frame, rotation, centre)
        if gain != 1.0 or offset != 0.0:
            cv2.addWeighted(frame, float(gain), frame, 0.0, float(offset), dst=frame)
        if noise_std > 0.0:
            if rng is None:
                raise ValueError("noise_std > 0 needs an rng")
            noise = rng.standard_normal(frame.shape[:2], dtype=np.float32)
            noise *= noise_std
            frame = cv2.add(frame, cv2.merge([noise, noise, noise]), dtype=cv2.CV_8U)
        return frame

    def _draw_sheet(self, frame: np.ndarray, rotation: np.ndarray, centre: np.ndarray) -> None:
        half = 0.5 * self._sheet_side
        r, d = self._sheet_right, self._sheet_down
        c = self._sheet_center
        corners = np.array(
            [
                c - half * r - half * d,
                c + half * r - half * d,
                c + half * r + half * d,
                c - half * r + half * d,
            ]
        )
        poly = self._clip((corners - centre) @ rotation.T)
        if len(poly) < 3:
            return
        pts = self._project_fixed(poly)
        scale = float(1 << self._SUBPIXEL_BITS)
        height, width = frame.shape[:2]
        x0 = max(int(math.floor(pts[:, 0].min() / scale)), 0)
        y0 = max(int(math.floor(pts[:, 1].min() / scale)), 0)
        x1 = min(int(math.ceil(pts[:, 0].max() / scale)) + 1, width)
        y1 = min(int(math.ceil(pts[:, 1].max() / scale)) + 1, height)
        if x1 <= x0 or y1 <= y0:
            return

        c_cam = rotation @ (c - centre)
        if c_cam[2] > self.camera.near:
            size_px = self._f * self._sheet_side / c_cam[2]
        else:
            size_px = float(max(x1 - x0, y1 - y0))
        texture = next((t for t in self._textures if t.shape[0] >= size_px), self._textures[-1])

        # Homography texture pixel (s, t) -> image, built from the 3D sheet pose rather than from
        # projected corners, so it stays exact when corners are behind the camera; the clipped
        # outline then masks the pixels that really see the sheet.
        step = self._sheet_side / texture.shape[0]
        r_cam, d_cam = rotation @ r, rotation @ d
        origin = c_cam + (0.5 * step - half) * (r_cam + d_cam)  # centre of texel (0, 0)
        homography = self._k @ np.column_stack([step * r_cam, step * d_cam, origin])
        to_roi = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        warped = cv2.warpPerspective(
            texture,
            to_roi @ homography,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        offset = np.array([x0, y0], dtype=np.int32) << self._SUBPIXEL_BITS
        cv2.fillPoly(mask, [(pts - offset).reshape(-1, 1, 2)], 255, cv2.LINE_8, self._SUBPIXEL_BITS)
        shaded = cv2.LUT(warped, self._sheet_lut)
        np.copyto(frame[y0:y1, x0:x1], shaded[:, :, None], where=mask[:, :, None] > 0)


class SimWorld(object):
    """The robot in the room: kinematics, collisions, lidar and camera."""

    def __init__(
        self,
        world: WorldConfig = WorldConfig(),
        robot: RobotConfig = RobotConfig(),
        lidar: LidarConfig = LidarConfig(),
        camera: CameraConfig = CameraConfig(),
        appearance: Appearance = Appearance(),
        noise: NoiseConfig = NoiseConfig(),
        randomize: bool = False,
        seed: Optional[int] = 0,
        dt: float = 0.02,
    ) -> None:
        _validate(world, robot, lidar, dt)
        self.world, self.robot, self.lidar, self.noise = world, robot, lidar, noise
        self.randomize = bool(randomize)
        self.dt = float(dt)
        self.renderer = CameraRenderer(world, robot, camera, appearance)
        self._rng = np.random.default_rng(seed)

        self._box = box_corners(world)
        room = room_corners(world)
        self._room_min, self._room_max = room.min(axis=0), room.max(axis=0)
        self._seg_a = np.vstack([room, self._box])
        self._seg_b = np.vstack([np.roll(room, -1, axis=0), np.roll(self._box, -1, axis=0)])
        self._num_wall_segments = len(room)
        self.beam_angles = lidar.angle_min + lidar.angle_increment * np.arange(lidar.num_beams)
        hl, hw = 0.5 * robot.footprint_length, 0.5 * robot.footprint_width
        fx = robot.footprint_center_x
        self._footprint = np.array([[fx - hl, -hw], [fx + hl, -hw], [fx + hl, hw], [fx - hl, hw]])
        self._bumper = self._footprint[1:3]

        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._cmd = (0.0, 0.0)
        self._v_ref = self._w_ref = 0.0  # acceleration-limited command the motors track
        self._velocity = (0.0, 0.0)
        self._collided = False
        self._time = 0.0
        self._wheel_bias = (0.0, 0.0)
        self._gain, self._offset = 1.0, 0.0

    # ------------------------------------------------------------ state

    @property
    def pose(self) -> Pose2D:
        return self._pose

    @property
    def velocity(self) -> Tuple[float, float]:
        """Measured body velocity (v m/s, w rad/s) over the last step, what odometry reports."""
        return self._velocity

    @property
    def collided(self) -> bool:
        """True once a step was blocked by the box or a wall since the last reset."""
        return self._collided

    @property
    def time(self) -> float:
        return self._time

    def set_command(self, v: float, w: float) -> None:
        self._cmd = (float(v), float(w))

    def reset(self, pose: Pose2D) -> None:
        """Teleport to ``pose`` at rest; clears ``collided`` and draws the per-episode
        disturbances when randomizing. ValueError if the footprint would overlap anything."""
        pose = Pose2D(float(pose.x), float(pose.y), wrap_angle(float(pose.yaw)))
        if self.in_collision(pose):
            raise ValueError("pose %s puts the robot inside the box or a wall" % (tuple(pose),))
        self._pose = pose
        self._cmd = (0.0, 0.0)
        self._v_ref = self._w_ref = 0.0
        self._velocity = (0.0, 0.0)
        self._collided = False
        if self.randomize:
            n = self.noise
            self._wheel_bias = tuple(self._rng.normal(0.0, n.wheel_bias_std, 2))  # type: ignore[assignment]
            self._gain = float(self._rng.uniform(*n.brightness_gain))
            self._offset = float(self._rng.uniform(*n.brightness_offset))

    def reset_to_start(self, start: StartPose) -> Pose2D:
        """Teleport to a start pose (jittered when randomizing) and return the pose used."""
        pose = start_pose_to_world(start, self.world)
        if self.randomize:
            n = self.noise
            jittered = Pose2D(
                pose.x + self._rng.uniform(-n.pose_jitter_m, n.pose_jitter_m),
                pose.y + self._rng.uniform(-n.pose_jitter_m, n.pose_jitter_m),
                pose.yaw + self._rng.uniform(-n.yaw_jitter_rad, n.yaw_jitter_rad),
            )
            if not self.in_collision(jittered):
                pose = jittered
        self.reset(pose)
        return self._pose

    # ------------------------------------------------------------ kinematics

    def step(self) -> None:
        """Advance by dt: slew the motors towards the command, move, stop at contact."""
        dt = self.dt
        v_cmd, w_cmd = self._cmd
        dv_max, dw_max = self.robot.max_linear_accel * dt, self.robot.max_angular_accel * dt
        self._v_ref += min(max(v_cmd - self._v_ref, -dv_max), dv_max)
        self._w_ref += min(max(w_cmd - self._w_ref, -dw_max), dw_max)
        v, w = self._wheel_response(self._v_ref, self._w_ref)

        target = _integrate(self._pose, v, w, dt)
        if self.in_collision(target):
            # Bisect for the last collision-free fraction of the step: the robot ends up touching.
            lo, hi = 0.0, 1.0
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if self.in_collision(_integrate(self._pose, v, w, mid * dt)):
                    hi = mid
                else:
                    lo = mid
            target = _integrate(self._pose, v, w, lo * dt)
            self._v_ref = self._w_ref = 0.0
            v = w = 0.0
            self._collided = True
        self._pose = target
        self._velocity = (v, w)
        self._time += dt

    def _wheel_response(self, v: float, w: float) -> Tuple[float, float]:
        if not self.randomize:
            return v, w
        half = 0.5 * self.robot.wheel_separation
        slip = self._rng.normal(0.0, self.noise.wheel_slip_std, 2)
        left = (v - w * half) * (1.0 + self._wheel_bias[0]) * (1.0 + slip[0])
        right = (v + w * half) * (1.0 + self._wheel_bias[1]) * (1.0 + slip[1])
        return 0.5 * (left + right), (right - left) / self.robot.wheel_separation

    # ------------------------------------------------------------ geometry

    def footprint(self, pose: Optional[Pose2D] = None) -> np.ndarray:
        """4x2 world corners of the robot footprint, counter-clockwise."""
        return _to_world(self._pose if pose is None else pose, self._footprint)

    def in_collision(self, pose: Pose2D) -> bool:
        corners = self.footprint(pose)
        if np.any(corners < self._room_min) or np.any(corners > self._room_max):
            return True
        return _convex_polygons_overlap(corners, self._box)

    def bumper_to_box(self, pose: Optional[Pose2D] = None) -> float:
        """Shortest distance (m) from the front bumper edge to the box outline; 0 when touching."""
        a, b = _to_world(self._pose if pose is None else pose, self._bumper)
        return _segment_polygon_distance(a, b, self._box)

    def marker_bearing_deg(self, pose: Optional[Pose2D] = None) -> Optional[float]:
        """True horizontal angle of the marker centre from the optical axis (positive = left), or
        None when the marker centre is behind the camera."""
        rotation, centre = self.renderer.camera_pose(self._pose if pose is None else pose)
        marker = np.array([self.world.box_x, self.world.box_y, self.world.marker_height])
        x, _, z = rotation @ (marker - centre)
        if z <= 0.0:
            return None
        return math.degrees(math.atan2(-x, z))

    def lidar_origin(self, pose: Optional[Pose2D] = None) -> np.ndarray:
        return _to_world(self._pose if pose is None else pose, np.array([[self.robot.lidar_x, 0.0]]))[0]

    def cast(self, origin: Sequence[float], angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distances along world angles from ``origin`` to the walls or the box, and whether each
        ray ended on the box."""
        dist, index = cast_rays(origin, np.asarray(angles, dtype=np.float64), self._seg_a, self._seg_b)
        return dist, index >= self._num_wall_segments

    # ------------------------------------------------------------ sensors

    def scan(self) -> np.ndarray:
        """float32 ranges of one lidar revolution at the current pose (+inf = no valid return)."""
        ranges, _ = self.cast(self.lidar_origin(), self._pose.yaw + self.beam_angles)
        if self.randomize and self.noise.scan_noise_std > 0.0:
            ranges = ranges + self._rng.normal(0.0, self.noise.scan_noise_std, ranges.shape)
        with np.errstate(invalid="ignore"):
            invalid = ~np.isfinite(ranges) | (ranges < self.lidar.range_min) | (ranges > self.lidar.range_max)
        return np.where(invalid, np.inf, ranges).astype(np.float32)

    def render(self) -> np.ndarray:
        """BGR uint8 camera frame at the current pose."""
        if not self.randomize:
            return self.renderer.render(self._pose)
        return self.renderer.render(
            self._pose, self._gain, self._offset, self.noise.pixel_noise_std, self._rng
        )


def _integrate(pose: Pose2D, v: float, w: float, dt: float) -> Pose2D:
    """Exact unicycle motion over dt at constant (v, w)."""
    yaw1 = pose.yaw + w * dt
    if abs(w) < 1e-9:
        dx, dy = v * dt * math.cos(pose.yaw), v * dt * math.sin(pose.yaw)
    else:
        r = v / w
        dx, dy = r * (math.sin(yaw1) - math.sin(pose.yaw)), -r * (math.cos(yaw1) - math.cos(pose.yaw))
    return Pose2D(pose.x + dx, pose.y + dy, wrap_angle(yaw1))


def _validate(world: WorldConfig, robot: RobotConfig, lidar: LidarConfig, dt: float) -> None:
    if dt <= 0.0:
        raise ValueError("dt must be positive, got %s" % dt)
    if min(world.room_length, world.room_width, world.wall_height) <= 0.0:
        raise ValueError("room dimensions must be positive")
    if min(world.box_width, world.box_depth, world.box_height, world.marker_side) <= 0.0:
        raise ValueError("box and marker dimensions must be positive")
    half = 0.5 * world.sheet_side
    if world.sheet_side > world.box_width or not half <= world.marker_height <= world.box_height - half:
        raise ValueError(
            "the %.3f m marker sheet at height %.3f m does not fit on the %.2f x %.2f m box face"
            % (world.sheet_side, world.marker_height, world.box_width, world.box_height)
        )
    lo, hi = room_corners(world).min(axis=0), room_corners(world).max(axis=0)
    box = box_corners(world)
    if np.any(box <= lo) or np.any(box >= hi):
        raise ValueError("the box must stand inside the room")
    if min(robot.footprint_length, robot.footprint_width, robot.wheel_separation) <= 0.0:
        raise ValueError("robot dimensions must be positive")
    if robot.max_linear_accel <= 0.0 or robot.max_angular_accel <= 0.0:
        raise ValueError("acceleration limits must be positive")
    if lidar.num_beams <= 0 or not 0.0 <= lidar.range_min < lidar.range_max:
        raise ValueError("need num_beams > 0 and 0 <= range_min < range_max")
