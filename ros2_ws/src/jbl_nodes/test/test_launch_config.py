"""The launch params files and the simulated demo start poses (pure Python, no ROS needed)."""

import ast
import math
import os

import pytest
import yaml

from jbl_nodes import aruco_utils as au
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
