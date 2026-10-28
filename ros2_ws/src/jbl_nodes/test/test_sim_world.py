"""Kinematic simulator: kinematics, collisions, lidar geometry, camera rendering, start poses.

Pure numpy / OpenCV: runs on the host venv (OpenCV >= 4.7, new ArUco API) and under colcon in
the Foxy / Humble / Jazzy containers (old ArUco API), so marker generation and detection are
exercised with both APIs.
"""

import math
import os
import time

import cv2
import numpy as np
import pytest

from jbl_nodes import aruco_utils as au
from jbl_nodes.sim_world import (
    Appearance,
    Pose2D,
    RobotConfig,
    SimWorld,
    StartPose,
    WorldConfig,
    box_corners,
    load_start_poses,
    start_pose_to_world,
)

ROBOT = RobotConfig()
LIDAR_X = 0.064
BUMPER_X = 0.097
DT = 0.02


def start_poses_path():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "start_poses.yaml")
    if os.path.exists(here):
        return here
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("jbl_nodes"), "config", "start_poses.yaml")


def facing_box(distance, lateral=0.0, heading_deg=0.0, world=WorldConfig()):
    return start_pose_to_world(StartPose(0, distance, lateral, heading_deg), world)


def run(sim, v, w, seconds):
    sim.set_command(v, w)
    for _ in range(int(round(seconds / sim.dt))):
        sim.step()


def jpeg_round_trip(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- start poses


def test_start_pose_frame_conventions():
    world = WorldConfig()  # front face centre at (1.5, 0) looking along -x
    assert facing_box(1.0) == pytest.approx(Pose2D(0.5, 0.0, 0.0))
    assert facing_box(1.2, lateral=0.3) == pytest.approx(Pose2D(0.3, 0.3, 0.0))  # robot's left is +y
    assert facing_box(1.0, heading_deg=20.0).yaw == pytest.approx(math.radians(20.0))

    turned = world._replace(box_x=0.0, box_y=1.0, box_yaw=-math.pi / 2)  # face looks along -y
    pose = facing_box(1.0, lateral=0.3, heading_deg=-10.0, world=turned)
    assert (pose.x, pose.y) == pytest.approx((-0.3, 0.0))  # robot faces +y, its left is -x
    assert pose.yaw == pytest.approx(math.pi / 2 - math.radians(10.0))


def test_shipped_start_poses_meet_the_protocol():
    poses = load_start_poses(start_poses_path())
    assert [p.pose_id for p in poses] == list(range(10))
    sim = SimWorld()
    observer = au.MarkerObserver()
    needs_search = 0
    for start in poses:
        assert 0.8 <= start.distance_m <= 1.6 and abs(start.lateral_m) <= 0.5
        assert abs(start.heading_deg) <= 40.0
        pose = sim.reset_to_start(start)  # raises if the robot would overlap the box or a wall
        bearing = sim.marker_bearing_deg()
        assert bearing is not None
        if observer.observe(sim.render()) is not None:
            continue
        needs_search += 1
        # Not detected at the start: turning in place towards the marker finds it within 90 deg.
        side = math.copysign(1.0, bearing)
        found = None
        for deg in range(1, 91):
            sim.reset(Pose2D(pose.x, pose.y, pose.yaw + side * math.radians(deg)))
            if observer.observe(sim.render()) is not None:
                found = deg
                break
        assert found is not None, "pose %d: marker not found within a 90 deg turn" % start.pose_id
    assert 3 <= needs_search <= 6, "expected a mix of visible and search starts"


def test_load_start_poses_rejects_bad_files(tmp_path):
    def write(text):
        path = tmp_path / "poses.yaml"
        path.write_text(text)
        return str(path)

    good = "start_poses:\n  - {pose_id: 0, distance_m: 1.0, lateral_m: 0.0, heading_deg: 0.0}\n"
    assert load_start_poses(write(good)) == [StartPose(0, 1.0, 0.0, 0.0)]
    for bad in (
        "poses: []\n",
        "start_poses:\n  - {pose_id: 1, distance_m: 1.0, lateral_m: 0.0, heading_deg: 0.0}\n",
        "start_poses:\n  - {pose_id: 0, distance_m: 1.0, lateral_m: 0.0}\n",
        "start_poses:\n  - {pose_id: 0, distance_m: .nan, lateral_m: 0.0, heading_deg: 0.0}\n",
        good + "  - {pose_id: 0, distance_m: 1.2, lateral_m: 0.0, heading_deg: 0.0}\n",
    ):
        with pytest.raises(ValueError):
            load_start_poses(write(bad))


# ---------------------------------------------------------------- kinematics


def test_acceleration_limits_and_straight_drive():
    sim = SimWorld()
    sim.reset(Pose2D(-1.0, 0.0, 0.0))
    run(sim, 0.25, 0.0, DT)
    assert sim.velocity == pytest.approx((0.02, 0.0))  # 1.0 m/s^2 for one 50 Hz step
    run(sim, 0.25, 0.0, 0.24)
    assert sim.velocity[0] == pytest.approx(0.25)
    run(sim, 0.25, 0.0, 1.74)  # 2.0 s in total
    # 2 s at 0.25 m/s minus what the 0.25 s ramp loses (v^2 / 2a), to within a step's slack.
    assert sim.pose.x - (-1.0) == pytest.approx(0.25 * 2.0 - 0.25**2 / 2.0, abs=0.003)
    assert sim.pose.y == pytest.approx(0.0, abs=1e-12) and sim.pose.yaw == pytest.approx(0.0, abs=1e-12)
    assert sim.time == pytest.approx(2.0)

    run(sim, 0.0, 0.0, 0.2)
    assert sim.velocity[0] == pytest.approx(0.05)
    run(sim, 0.0, 0.0, 0.06)
    assert sim.velocity == (0.0, 0.0) and not sim.collided


def test_turning_follows_an_exact_circle():
    sim = SimWorld()
    sim.reset(Pose2D(-1.0, -0.5, 0.0))
    run(sim, 0.0, 1.5, DT)
    assert sim.velocity[1] == pytest.approx(0.04)  # 2.0 rad/s^2
    run(sim, 0.2, 0.5, 1.0)  # well past both ramps: steady arc of radius v / w = 0.4 m
    p0 = sim.pose
    centre = np.array([p0.x - 0.4 * math.sin(p0.yaw), p0.y + 0.4 * math.cos(p0.yaw)])
    radii = []
    for _ in range(int(round(2 * math.pi / 0.5 / DT))):
        sim.step()
        radii.append(np.hypot(sim.pose.x - centre[0], sim.pose.y - centre[1]))
    assert np.allclose(radii, 0.4, atol=1e-9)
    assert (sim.pose.x, sim.pose.y) == pytest.approx((p0.x, p0.y), abs=0.01)  # back after one turn


def test_randomized_world_is_noisy_but_reproducible_and_stops_cleanly():
    def drive(seed):
        sim = SimWorld(randomize=True, seed=seed)
        sim.reset_to_start(StartPose(0, 1.6, 0.0, 0.0))
        start = sim.pose
        run(sim, 0.2, 0.0, 2.0)
        moved = sim.pose
        run(sim, 0.0, 0.0, 0.5)
        return start, moved, sim

    start_a, moved_a, sim_a = drive(1)
    start_b, moved_b, _ = drive(1)
    _, moved_c, _ = drive(2)
    assert moved_a == moved_b and moved_a != moved_c
    nominal = start_pose_to_world(StartPose(0, 1.6, 0.0, 0.0))
    assert abs(start_a.x - nominal.x) <= 0.03 and abs(start_a.yaw - nominal.yaw) <= math.radians(3.0)
    travelled = math.hypot(moved_a.x - start_a.x, moved_a.y - start_a.y)
    assert travelled == pytest.approx(0.2 * 2.0 - 0.02, rel=0.1)
    assert sim_a.velocity == (0.0, 0.0), "wheel noise must not make a stopped robot creep"


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="inside the box"):
        SimWorld().reset(facing_box(0.05))
    with pytest.raises(ValueError, match="inside the room"):
        SimWorld(WorldConfig(box_x=2.1))
    with pytest.raises(ValueError, match="does not fit"):
        SimWorld(WorldConfig(marker_height=0.02))


# ---------------------------------------------------------------- collisions


def test_driving_into_the_box_stops_at_contact():
    sim = SimWorld()
    sim.reset(facing_box(0.6))
    assert sim.bumper_to_box() == pytest.approx(0.6 - BUMPER_X)
    sim.set_command(0.25, 0.0)
    for _ in range(int(3.0 / DT)):
        sim.step()
        assert not sim.in_collision(sim.pose)
    assert sim.collided
    assert sim.bumper_to_box() < 1e-3
    assert sim.velocity == (0.0, 0.0)
    x_stuck = sim.pose.x
    run(sim, 0.25, 0.0, 0.5)
    assert sim.pose.x == pytest.approx(x_stuck, abs=1e-4)
    assert sim.scan()[180] == np.inf, "the box is inside the lidar's minimum range"

    run(sim, -0.1, 0.0, 1.0)
    assert sim.pose.x < x_stuck - 0.05 and sim.velocity[0] < 0.0
    assert sim.collided, "the flag latches until the next reset"
    sim.reset(facing_box(1.0))
    assert not sim.collided


def test_turning_in_place_beside_the_box_is_blocked():
    sim = SimWorld()
    # Parked alongside the box's +y side, 5 mm off it: turning right swings the nose into it.
    sim.reset(Pose2D(1.65, 0.15 + 0.089 + 0.005, 0.0))
    run(sim, 0.0, -1.0, 1.0)
    assert sim.collided and not sim.in_collision(sim.pose)
    assert -0.1 < sim.pose.yaw < 0.0
    stuck = sim.pose
    run(sim, -0.1, 0.0, 0.5)
    assert sim.pose.x < stuck.x - 0.02 and sim.pose.y > stuck.y, "backing away is free"


def test_walls_stop_the_robot():
    sim = SimWorld()
    sim.reset(Pose2D(-1.5, 0.0, math.pi))
    run(sim, 0.25, 0.0, 3.0)
    assert sim.collided
    assert sim.footprint()[:, 0].min() == pytest.approx(-2.0, abs=1e-3)


# ---------------------------------------------------------------- lidar


def test_scan_layout_and_room_ranges():
    sim = SimWorld()
    sim.reset(Pose2D(-1.0, 0.0, 0.0))
    ranges = sim.scan()
    assert ranges.shape == (360,) and ranges.dtype == np.float32
    assert sim.beam_angles[0] == pytest.approx(-math.pi) and sim.beam_angles[180] == pytest.approx(0.0)
    lidar_x = -1.0 + LIDAR_X
    assert ranges[0] == pytest.approx(lidar_x + 2.0, abs=1e-5)  # behind: wall at x = -2
    assert ranges[90] == pytest.approx(1.5, abs=1e-5)  # right: wall at y = -1.5
    assert ranges[270] == pytest.approx(1.5, abs=1e-5)  # left: wall at y = +1.5
    assert ranges[180] == pytest.approx(1.5 - lidar_x, abs=1e-5)  # ahead: box front face at x = 1.5


@pytest.mark.parametrize(
    "distance, lateral, heading_deg",
    [
        (0.25, 0.0, 0.0),
        (0.314, 0.0, 0.0),
        (0.6, 0.05, 10.0),
        (1.0, -0.1, -20.0),
        (1.6, 0.1, 35.0),
        (2.5, 0.0, 0.0),
    ],
)
def test_box_face_range_matches_geometry(distance, lateral, heading_deg):
    world = WorldConfig()
    sim = SimWorld(world)
    sim.reset(facing_box(distance, lateral, heading_deg))
    ranges = sim.scan()
    origin = sim.lidar_origin()
    face = np.array([world.box_x, world.box_y])
    normal = np.array([math.cos(world.box_yaw), math.sin(world.box_yaw)])
    tangent = np.array([-normal[1], normal[0]])
    checked = 0
    for i, angle in enumerate(sim.pose.yaw + sim.beam_angles):
        d = np.array([math.cos(angle), math.sin(angle)])
        if np.dot(d, normal) >= -1e-6:
            continue
        t = np.dot(face - origin, normal) / np.dot(d, normal)  # beam meets the face plane
        along = np.dot(origin + t * d - face, tangent)
        if t <= 0 or abs(along) > 0.5 * world.box_width - 0.005:
            continue
        expected = t if t >= 0.15 else np.inf
        assert ranges[i] == pytest.approx(expected, abs=0.01)
        checked += 1
    assert checked >= 3


def test_scan_noise_when_randomized():
    sim = SimWorld(randomize=True, seed=3)
    sim.reset(Pose2D(-1.0, 0.0, 0.0))
    exact = SimWorld()
    exact.reset(sim.pose)
    err = sim.scan().astype(np.float64) - exact.scan()
    assert 0.003 < float(np.std(err)) < 0.008 and abs(float(np.mean(err))) < 0.002


# ---------------------------------------------------------------- camera


@pytest.mark.parametrize(
    "distance, lateral, heading_deg",
    [
        (0.314, 0.0, 0.0),  # docked: lidar 0.25 m from the face
        (0.4, 0.05, 10.0),
        (0.8, 0.25, -20.0),
        (1.0, 0.0, 0.0),
        (1.2, -0.3, 0.0),
        (1.6, 0.5, 0.0),
        (1.0, -0.4, 45.0),
        (2.5, 0.2, -5.0),
    ],
)
def test_rendered_marker_is_detected_at_its_true_bearing(distance, lateral, heading_deg):
    sim = SimWorld()
    sim.reset(facing_box(distance, lateral, heading_deg))
    truth = sim.marker_bearing_deg()
    assert truth is not None and abs(truth) < 35.0
    frame = sim.render()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    for img in (frame, jpeg_round_trip(frame)):
        seen = au.MarkerObserver(marker_id=7).observe(img)
        assert seen is not None, "marker not detected"
        assert seen.bearing_deg == pytest.approx(truth, abs=2.0)
        assert abs(seen.bearing_deg - truth) < 0.5  # the 2 deg budget is mostly unused


@pytest.mark.parametrize("distance", [0.5, 1.0, 2.0])
def test_marker_is_centred_and_sized_by_the_pinhole_model(distance):
    sim = SimWorld()
    sim.reset(facing_box(distance))
    frame = sim.render()
    det = [d for d in au.MarkerDetector().detect(frame) if d.marker_id == 7]
    assert len(det) == 1
    assert det[0].center == pytest.approx((319.5, 239.5), abs=0.5)  # camera at marker height

    # Width of the marker's top border row, to a fraction of a pixel: sum of each pixel's
    # darkness between sheet white and ink black (the detector's corners are only whole pixels).
    side_px = au.focal_length_px(640, 1.396) * 0.10 / (distance - ROBOT.camera_x)
    row = int(round(239.5 - side_px / 2 + side_px / 12))  # middle of the first bit row: all black
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)[row]
    sheet, ink = grey[int(319.5 - 0.75 * side_px)], grey[320]
    darkness = np.clip((sheet - grey) / (sheet - ink), 0.0, 1.0)
    span = slice(int(319.5 - 0.8 * side_px), int(319.5 + 0.8 * side_px))
    assert darkness[span].sum() == pytest.approx(side_px, abs=0.5)


def _distance_to_rect(point, corners):
    """Distance from a 2D point to a convex polygon (negative inside)."""
    worst = -np.inf
    best = np.inf
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        edge = b - a
        outward = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
        worst = max(worst, float(np.dot(point - a, outward)))
        t = np.clip(np.dot(point - a, edge) / np.dot(edge, edge), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (a + t * edge))))
    return worst if worst < 0 else best


@pytest.mark.parametrize("box_yaw", [math.pi, 2.2])
def test_box_silhouette_matches_ray_casting_from_any_pose(box_yaw):
    """At the horizon row each column shows whatever its horizontal ray hits first. This holds
    only if faces and the marker sheet are clipped correctly when they are partly behind or right
    beside the camera (a wrong clip mirrors them into the image)."""
    world = WorldConfig(box_x=1.0, box_y=0.2, box_yaw=box_yaw)
    colors = Appearance(wall=(40, 200, 40), box=(40, 40, 200))
    sim = SimWorld(world, appearance=colors)
    box = box_corners(world)
    face = np.array([world.box_x, world.box_y])
    normal = np.array([math.cos(box_yaw), math.sin(box_yaw)])
    tangent = np.array([-normal[1], normal[0]])
    along = math.atan2(tangent[1], tangent[0])
    # 4 cm in front of the face, beside the sheet's centre, looking (almost) along the face: the
    # near half of the sheet is behind the camera plane.
    cameras = [
        (face + 0.04 * normal + side * 0.03 * tangent, along + turn)
        for side in (-1.0, 1.0)
        for turn in (0.15, -0.15, math.pi + 0.15, math.pi - 0.15)
    ]
    rng = np.random.default_rng(0)
    while len(cameras) < 80:
        cam = box.mean(axis=0) + rng.uniform(-0.3, 0.3, 2) * (3.0 if len(cameras) > 30 else 1.0)
        inside_room = (
            abs(cam[0]) < 0.5 * world.room_length - 0.03 and abs(cam[1]) < 0.5 * world.room_width - 0.03
        )
        if inside_room and _distance_to_rect(cam, box) >= 0.03:
            cameras.append((cam, rng.uniform(-math.pi, math.pi)))

    f = au.focal_length_px(640, 1.396)
    columns = np.arange(640)
    for cam, yaw in cameras:
        pose = Pose2D(cam[0] - ROBOT.camera_x * math.cos(yaw), cam[1] - ROBOT.camera_x * math.sin(yaw), yaw)
        row = sim.renderer.render(pose)[240].astype(int)
        seen_box = ~((row[:, 1] > row[:, 2] + 60) & (row[:, 1] > row[:, 0] + 60))
        _, hits_box = sim.cast(cam, yaw + np.arctan2(319.5 - columns, f))
        wrong = np.flatnonzero(seen_box != hits_box)
        edges = np.flatnonzero(np.diff(hits_box.astype(int))) + 0.5
        for u in wrong:  # only the antialiased silhouette edges may disagree
            assert edges.size and np.min(np.abs(edges - u)) <= 1.5, (pose, u)


def test_box_behind_the_camera_leaves_no_trace():
    behind = SimWorld().renderer.render(Pose2D(1.3, 0.0, math.pi))  # box at x >= 1.5, camera looks along -x
    elsewhere = SimWorld(WorldConfig(box_y=1.0)).renderer.render(Pose2D(1.3, 0.0, math.pi))
    assert np.array_equal(behind, elsewhere)


def test_randomized_exposure_and_noise():
    plain = SimWorld()
    noisy = SimWorld(randomize=True, seed=5)
    pose = noisy.reset_to_start(StartPose(0, 1.0, 0.0, 0.0))
    plain.reset(pose)
    a, b = plain.render().astype(np.float64).ravel(), noisy.render().astype(np.float64).ravel()
    unclipped = (b > 15) & (b < 240)
    gain, offset = np.polyfit(a[unclipped], b[unclipped], 1)
    assert 0.7 <= gain <= 1.3 and -21 <= offset <= 21
    assert abs(gain - 1.0) > 0.01 or abs(offset) > 1.0, "no exposure change drawn"
    residual = b[unclipped] - (gain * a[unclipped] + offset)
    assert 2.5 < residual.std() < 3.5  # Gaussian pixel noise of 3 grey levels
    assert not np.array_equal(noisy.render(), noisy.render()), "noise is drawn per frame"
    assert au.MarkerObserver().observe(noisy.render()) is not None


def test_render_time_budget():
    for sim in (SimWorld(), SimWorld(randomize=True)):
        sim.reset(facing_box(0.314))  # docked: the sheet fills a third of the frame
        sim.render()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            sim.render()
            times.append(time.perf_counter() - t0)
        assert float(np.median(times)) < 0.015, "median %.1f ms" % (1e3 * float(np.median(times)))
