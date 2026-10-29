"""Real robot, on-board policy: camera + safety gate + judge + policy_runner + trial logger (+ bag).

    ros2 launch jbl_nodes robot_policy.launch.py model_path:=/models/bc/policy.engine backend:=tensorrt
    ros2 launch jbl_nodes robot_policy.launch.py model_path:=/models/bc/policy.onnx \
        with_servo:=true trials_csv:=$D/trials/2026-11-02/trials.csv bag:=$D/bags/trials_r1

with D=/autonomous_ROS/autojetsonbot-learning/data/robot (on the /autonomous_ROS bind mount;
$HOME inside the container is not persistent).

policy_runner drives only after ``start <method> ...`` (method_filter = the method argument)
and stops on any ``end``. with_servo:=true adds servo_baseline for interleaved paired trials
(docs/PROTOCOL.md section 8). policy_meta.json must sit beside the model file.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from jbl_nodes.episode_cli import parse_event_args
from jbl_nodes.launch_common import (
    arg,
    bag_recorder,
    declare,
    make_params,
    policy_runner,
    robot_arguments,
    robot_nodes,
)


def _setup(context):
    method = arg(context, "method")
    parse_event_args(["start", method, "0", "0"])  # same naming rule as the episode CLI
    if method == "servo":
        raise ValueError("method:=servo is the baseline; add with_servo:=true next to a policy method")
    params = make_params(context)
    return robot_nodes(context, params) + [policy_runner(context, params, method)] + bag_recorder(context)


def generate_launch_description():
    return LaunchDescription(
        robot_arguments()
        + [
            declare("method", "bc", "method name this policy answers to in 'start <method> ...'"),
            declare(
                "model_path", "", "ONNX (onnxruntime), TensorRT engine (tensorrt) or TorchScript (torch)"
            ),
            declare("backend", "onnxruntime", "onnxruntime | tensorrt | torch | constant"),
            OpaqueFunction(function=_setup),
        ]
    )
