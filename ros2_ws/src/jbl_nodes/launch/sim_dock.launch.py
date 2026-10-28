"""Docking in simulation: fake_robot + safety gate + judge + one controller + trial logger.

    ros2 launch jbl_nodes sim_dock.launch.py                       # servo baseline
    ros2 launch jbl_nodes sim_dock.launch.py method:=bc model_path:=runs/bc/policy.onnx
    ros2 launch jbl_nodes sim_dock.launch.py method:=none          # drive it yourself (teleop)

Then start episodes with ``ros2 run jbl_nodes episode --timeout 5 start <method> <pose_id>
<round>``: fake_robot teleports to the start pose, the controller drives, dock_judge ends the
episode and trial_logger writes the row.

method is the name episodes are started with; controller picks what drives (auto by default):
    servo    servo_baseline answering to "start <method> ..." (auto for method servo)
    policy   policy_runner with method_filter=<method>, model_path and backend (auto for any
             other method name)
    none     nothing; publish /cmd_vel_in yourself (auto for method none)
e.g. method:=servo_demo controller:=servo runs the baseline under another name, which keeps
scripted demonstrations apart from baseline trials in trials.csv and the dataset.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from jbl_nodes.episode_cli import parse_event_args
from jbl_nodes.launch_common import (
    arg,
    bag_recorder,
    bool_arg,
    common_arguments,
    declare,
    jbl_node,
    judge_and_logger,
    make_params,
    path_arg,
    policy_runner,
    safety_gate,
)


def _controller(context, params):
    method = arg(context, "method")
    parse_event_args(["start", method, "0", "0"])  # same naming rule as the episode CLI
    controller = arg(context, "controller")
    if controller == "auto":
        controller = method if method in ("servo", "none") else "policy"
    if controller == "none":
        return []
    if controller == "servo":
        return [jbl_node("servo_baseline", params, method_name=method)]
    if controller == "policy":
        return [policy_runner(context, params, method)]
    raise ValueError("controller:=%s is not one of auto, servo, policy, none" % controller)


def _setup(context):
    params = make_params(context)
    fake_robot = jbl_node(
        "fake_robot",
        params,
        randomize=bool_arg(context, "randomize"),
        seed=int(arg(context, "seed")),
        start_poses_file=path_arg(context, "start_poses_file") or None,
    )
    return (
        [fake_robot, safety_gate(params)]
        + judge_and_logger(context, params)
        + _controller(context, params)
        + bag_recorder(context, extra_topics=["/fake_robot/ground_truth"])
    )


def generate_launch_description():
    return LaunchDescription(
        common_arguments("sim.yaml", "~/jbl_trials/sim/trials.csv")
        + [
            declare("method", "servo", "method name used in 'start <method> ...' events"),
            declare("controller", "auto", "auto | servo | policy | none (auto: from method)"),
            declare("model_path", "", "policy model for policy_runner (ONNX, TorchScript or engine)"),
            declare(
                "backend", "onnxruntime", "policy_runner backend: onnxruntime | tensorrt | torch | constant"
            ),
            declare("randomize", "false", "fake_robot wheel slip/bias, start jitter, sensor and image noise"),
            declare("seed", "0", "fake_robot random seed"),
            declare("start_poses_file", "", "fake_robot start poses; empty = config/start_poses.yaml"),
            OpaqueFunction(function=_setup),
        ]
    )
