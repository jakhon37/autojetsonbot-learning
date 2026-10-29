"""The launch params files and the simulated demo start poses (pure Python, no ROS needed)."""

import ast
import math
import os

import pytest
import yaml

from jbl_nodes import aruco_utils as au
from jbl_nodes.sim_world import CameraConfig as SimCameraConfig
from jbl_nodes.sim_world import Pose2D, SimWorld, load_start_poses, start_pose_to_world

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PKG, "config")
NODES = os.path.join(PKG, "jbl_nodes")


def _declared_types():
    """{node module: {parameter: python type of its declared default}} from the node sources."""
    types = {}
    for module in (
        "camera_node",
        "dock_judge",
        "servo_baseline",
        "policy_runner",
        "fake_robot",
        "trial_logger",
    ):
        with open(os.path.join(NODES, module + ".py")) as f:
            tree = ast.parse(f.read())
        found = {}
        for call in ast.walk(tree):
            func = getattr(call, "func", None)
            name = getattr(func, "attr", getattr(func, "id", None))
            if isinstance(call, ast.Call) and name in ("declare_parameter", "p") and len(call.args) >= 2:
                key, default = call.args[0], call.args[1]
                if isinstance(key, ast.Constant) and isinstance(default, ast.Constant):
                    found[key.value] = type(default.value)
                elif isinstance(key, ast.Constant) and isinstance(default, ast.Attribute):
                    found[key.value] = float  # math.pi
        types[module] = found
    return types


DECLARED = _declared_types()


@pytest.mark.parametrize("name", ["sim.yaml", "robot.yaml"])
def test_params_files_match_declared_parameter_types(name):
    with open(os.path.join(CONFIG, name)) as f:
        doc = yaml.safe_load(f)
    shared = doc.pop("/**")["ros__parameters"]
    assert set(shared) == {"forward_angle", "hfov_rad"}
    assert all(isinstance(v, float) for v in shared.values())
    for node, section in doc.items():
        declared = DECLARED[node]
        for key, value in section["ros__parameters"].items():
            assert key in declared, "%s: %s.%s is not a parameter of the node" % (name, node, key)
            # Humble and newer refuse an int for a double parameter (and the reverse).
            assert type(value) is declared[key], "%s: %s.%s=%r should be a %s" % (
                name,
                node,
                key,
                value,
                declared[key].__name__,
            )


def test_robot_and_sim_cameras_are_explicit():
    with open(os.path.join(CONFIG, "robot.yaml")) as f:
        robot = yaml.safe_load(f)
    with open(os.path.join(CONFIG, "sim.yaml")) as f:
        sim = yaml.safe_load(f)
    # The robot: a stock Raspberry Pi Camera v2 lens (62.2 deg), read through Argus in the
    # full-sensor 4:3 mode (not a cropped 16:9 one) and scaled to 640x480.
    assert robot["/**"]["ros__parameters"]["hfov_rad"] == pytest.approx(math.radians(62.2), abs=1e-3)
    device = robot["camera_node"]["ros__parameters"]["device"]
    assert device.startswith("nvarguscamerasrc") and "width=1640,height=1232" in device
    assert "width=640,height=480" in device and "appsink" in device
    # The simulator: its own lens, stated next to what the judge and the servo assume.
    camera_hfov = sim["fake_robot"]["ros__parameters"]["camera_hfov_rad"]
    assert camera_hfov == sim["/**"]["ros__parameters"]["hfov_rad"] == pytest.approx(SimCameraConfig().hfov)


def test_sim_demo_poses_keep_the_test_poses_held_out():
    tests = load_start_poses(os.path.join(CONFIG, "start_poses.yaml"))
    poses = load_start_poses(os.path.join(CONFIG, "sim_demo_poses.yaml"))
    assert poses[:10] == tests, "ids 0-9 must stay identical to start_poses.yaml"
    spots = poses[10:]
    assert [p.pose_id for p in spots] == list(range(10, 30))

    sim = SimWorld()
    observer = au.MarkerObserver()
    test_xy = [start_pose_to_world(t, sim.world)[:2] for t in tests]
    out_of_view = 0
    for spot in spots:
        assert 0.70 <= spot.distance_m <= 1.70 and abs(spot.lateral_m) <= 0.55 and abs(spot.heading_deg) <= 45
        pose = sim.reset_to_start(spot)  # raises if it overlaps the box or a wall
        assert min(math.hypot(pose.x - x, pose.y - y) for x, y in test_xy) >= 0.10, spot
        if observer.observe(sim.render()) is not None:
            continue
        out_of_view += 1
        side = math.copysign(1.0, sim.marker_bearing_deg())
        assert _found_within(sim, observer, pose, side), (
            "spot %d: marker not found within 90 deg" % spot.pose_id
        )
    assert out_of_view == 5


def _found_within(sim, observer, pose, side, max_deg=90):
    for deg in range(1, max_deg + 1):
        sim.reset(Pose2D(pose.x, pose.y, pose.yaw + side * math.radians(deg)))
        if observer.observe(sim.render()) is not None:
            return True
    return False


def test_sim_eval_wide_poses_lie_outside_the_training_distribution():
    """scripts/sim_eval.sh condition "wide": every pose outside the demonstrations' box on at
    least one axis, placeable, and the marker findable within a 90 deg turn at both lenses."""
    spots = load_start_poses(os.path.join(CONFIG, "sim_demo_poses.yaml"))[10:]
    poses = load_start_poses(os.path.join(CONFIG, "sim_eval_wide_poses.yaml"))
    assert [p.pose_id for p in poses] == list(range(10))
    d_lo, d_hi = min(s.distance_m for s in spots), max(s.distance_m for s in spots)
    lat_hi = max(abs(s.lateral_m) for s in spots)
    head_hi = max(abs(s.heading_deg) for s in spots)
    for p in poses:
        assert (
            p.distance_m < d_lo or p.distance_m > d_hi or abs(p.lateral_m) > lat_hi or abs(p.heading_deg) > head_hi
        ), "pose %d is inside the demonstrations' distance / lateral / heading box" % p.pose_id
    for hfov in (SimCameraConfig().hfov, 1.0856):
        sim = SimWorld(camera=SimCameraConfig(hfov=hfov))
        observer = au.MarkerObserver(hfov_rad=hfov)
        for p in poses:
            pose = sim.reset_to_start(p)  # raises if it overlaps the box or a wall
            if observer.observe(sim.render()) is not None:
                continue
            side = math.copysign(1.0, sim.marker_bearing_deg())
            assert _found_within(sim, observer, pose, side), "pose %d at HFOV %.3f" % (p.pose_id, hfov)
