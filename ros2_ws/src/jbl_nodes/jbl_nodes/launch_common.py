"""Building blocks shared by the jbl_nodes launch files (launch / launch_ros APIs present in Foxy).

Launch arguments arrive as strings. The helpers here turn them into typed parameter values, so
``forward_angle:=3`` reaches the nodes as the double 3.0: Humble and newer reject an int for a
double parameter.

Parameter precedence for every node, lowest first: the node's own defaults, the package's
params file (``params_file``; node-name sections plus a ``/**`` section for values several nodes
must share), ``extra_params_file`` if given, then explicit launch arguments. The arguments are
written to a params file under the node's own name: rcl applies a node-name section after any
``/**`` section whatever the file order, so launch_ros's plain dict parameters (written under
``/**``) would lose to a node section of params_file.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "jbl_nodes"

# docs/INTERFACES.md topic list; /camera/image_raw is left out on purpose (the compressed stream
# is what gets recorded and converted).
RECORD_TOPICS = (
    "/camera/image_raw/compressed",
    "/odom",
    "/cmd_vel_in",
    "/cmd_vel",
    "/scan",
    "/episode/event",
    "/dock_judge/status",
    "/safety_gate/status",
    "/policy/latency",
)

# Every /episode/event writer is transient_local. Requesting transient_local too lets the
# recorder still catch an event whose writer it matched only after the publish; rosbag2 picks
# transient_local by itself only while every publisher offers it (one `ros2 topic pub` on the
# topic is enough to lose that), so the request is pinned here.
RECORD_QOS_OVERRIDES = """\
/episode/event:
  history: keep_last
  depth: 10
  reliability: reliable
  durability: transient_local
"""


def share_file(package: str, *parts: str) -> str:
    return os.path.join(get_package_share_directory(package), *parts)


def declare(name: str, default: str, description: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default, description=description)


def arg(context: LaunchContext, name: str) -> str:
    return LaunchConfiguration(name).perform(context).strip()


def bool_arg(context: LaunchContext, name: str) -> bool:
    text = arg(context, name).lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError("%s:=%s is not a boolean (true or false)" % (name, text))


def float_arg(context: LaunchContext, name: str) -> Optional[float]:
    """The argument as a float, or None when it is empty (= keep the params file value)."""
    text = arg(context, name)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError("%s:=%s is not a number" % (name, text)) from None


def path_arg(context: LaunchContext, name: str) -> str:
    text = arg(context, name)
    return os.path.abspath(os.path.expanduser(text)) if text else ""


class Params(object):
    """Per-node parameter lists: params files first, then shared and per-node overrides."""

    def __init__(self, files: Sequence[str], shared: Optional[Dict[str, Any]] = None) -> None:
        for path in files:
            if not os.path.isfile(path):
                raise FileNotFoundError("params file %s does not exist" % path)
        self._files = list(files)
        self._shared = dict(shared or {})
        self._dir: Optional[str] = None

    def for_node(self, name: str, base_files: Sequence[str] = (), **overrides: Any) -> List[str]:
        values = dict(self._shared)
        values.update({k: v for k, v in overrides.items() if v is not None})
        files = list(base_files) + self._files
        if values:
            files.append(self._write(name, values))
        return files

    def _write(self, name: str, values: Dict[str, Any]) -> str:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="jbl_launch_")
        path = os.path.join(self._dir, name + ".yaml")
        with open(path, "w") as f:
            yaml.safe_dump({name: {"ros__parameters": values}}, f, default_flow_style=False)
        return path


def common_arguments(default_params_file: str, default_trials_csv: str) -> List[DeclareLaunchArgument]:
    return [
        declare(
            "params_file",
            share_file(PACKAGE, "config", default_params_file),
            "node parameters (node-name sections and a /** section shared by all nodes)",
        ),
        declare("extra_params_file", "", "optional params file applied after params_file"),
        declare("forward_angle", "", "scan angle (rad) that points forward; empty = params file"),
        declare("hfov_rad", "", "camera horizontal field of view (rad); empty = params file"),
        declare("trials_csv", default_trials_csv, "trial_logger output (appended to)"),
        declare("bag", "", "record the INTERFACES topics to this new bag directory; empty = no bag"),
    ]


def make_params(context: LaunchContext) -> Params:
    files = [path_arg(context, "params_file")]
    extra = path_arg(context, "extra_params_file")
    if extra:
        files.append(extra)
    shared = {
        "forward_angle": float_arg(context, "forward_angle"),
        "hfov_rad": float_arg(context, "hfov_rad"),
    }
    return Params(files, {k: v for k, v in shared.items() if v is not None})


def jbl_node(executable: str, params: Params, **overrides: Any) -> Node:
    """A jbl_nodes executable as node ``executable`` with its parameters and overrides."""
    return Node(
        package=PACKAGE,
        executable=executable,
        name=executable,
        output="screen",
        emulate_tty=True,
        parameters=params.for_node(executable, **overrides),
    )


def safety_gate(params: Params, respawn: bool = False) -> Node:
    """jbl_safety's gate with its installed defaults, then the shared params.

    On the real robot ``respawn`` is on: diffdrive_node has no command watchdog, so a gate that
    dies without SIGINT/SIGTERM/SIGHUP/SIGQUIT (SIGKILL, a crash) leaves /cmd_vel at its last
    value. The respawned gate publishes zero from its first tick, but on Foxy / Fast DDS
    diffdrive_node only receives it after rediscovery, measured at 2.5-3.6 s after the kill.
    Respawn restores gating; it does not stop the robot quickly. Only a motor-side command
    timeout (the ESP32's 500 ms 'm' timeout, or a watchdog in diffdrive_node) does that:
    docs/PROTOCOL.md section 4 checks it before any trial.
    """
    return Node(
        package="jbl_safety",
        executable="safety_gate_node",
        name="safety_gate",
        output="screen",
        emulate_tty=True,
        parameters=params.for_node("safety_gate", [share_file("jbl_safety", "config", "safety_gate.yaml")]),
        respawn=respawn,
        respawn_delay=1.0 if respawn else None,
    )


def judge_and_logger(context: LaunchContext, params: Params) -> List[Node]:
    trials_csv = path_arg(context, "trials_csv")
    if not trials_csv:
        raise ValueError("trials_csv must not be empty")
    return [jbl_node("dock_judge", params), jbl_node("trial_logger", params, path=trials_csv)]


def policy_runner(context: LaunchContext, params: Params, method: str) -> Node:
    """policy_runner that answers to ``start <method> ...``; checks the arguments first."""
    backend = arg(context, "backend")
    model_path = path_arg(context, "model_path")
    if backend != "constant" and not model_path:
        raise ValueError("method:=%s runs policy_runner and needs model_path:=<model file>" % method)
    return jbl_node("policy_runner", params, model_path=model_path, backend=backend, method_filter=method)


def robot_arguments() -> List[DeclareLaunchArgument]:
    return common_arguments("robot.yaml", "~/jbl_trials/trials.csv") + [
        declare("camera_source", "camera", "camera | test_pattern (bench test without a camera)"),
        declare("with_servo", "false", "also run servo_baseline (drives only on 'start servo ...')"),
    ]


def robot_nodes(context: LaunchContext, params: Params) -> List[Node]:
    """Camera, gate (respawned), judge, logger and optionally the servo baseline."""
    nodes = [jbl_node("camera_node", params, source=arg(context, "camera_source")), safety_gate(params, True)]
    nodes += judge_and_logger(context, params)
    if bool_arg(context, "with_servo"):
        nodes.append(jbl_node("servo_baseline", params, method_name="servo"))
    return nodes


def bag_recorder(context: LaunchContext, extra_topics: Sequence[str] = ()) -> List[ExecuteProcess]:
    """``ros2 bag record`` of RECORD_TOPICS (+ extra_topics) to the ``bag`` argument, if set.

    The QoS override file is written next to the bag directory (rosbag2 refuses an existing
    output directory) and kept, so the recording documents how it was made.
    """
    bag = path_arg(context, "bag")
    if not bag:
        return []
    if os.path.exists(bag):
        raise FileExistsError("bag:=%s already exists; rosbag2 needs a new directory" % bag)
    parent = os.path.dirname(bag)
    os.makedirs(parent, exist_ok=True)
    qos_file = os.path.join(parent, os.path.basename(bag) + "_qos.yaml")
    with open(qos_file, "w") as f:
        f.write(RECORD_QOS_OVERRIDES)
    # setsid: Ctrl-C in the terminal reaches only ros2 launch, which then stops the recorder
    # once; a second SIGINT while rosbag2 closes the bag makes it exit with an error.
    cmd = ["setsid", "ros2", "bag", "record", "-o", bag, "--qos-profile-overrides-path", qos_file]
    return [
        ExecuteProcess(
            cmd=cmd + list(RECORD_TOPICS) + list(extra_topics),
            name="bag_record",
            output="screen",
            sigterm_timeout="10",  # let rosbag2 finish writing metadata.yaml
        )
    ]
