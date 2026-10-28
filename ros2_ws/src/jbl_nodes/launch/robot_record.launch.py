"""Real robot, demonstrations: camera + safety gate + judge + trial logger (+ bag).

    ros2 launch jbl_nodes robot_record.launch.py bag:=$HOME/jbl_bags/demos_$(date +%Y%m%d_%H%M)

Runs next to the robot's own bring-up (diffdrive_node on /cmd_vel, rplidar on /scan). Teleop
is external and must publish /cmd_vel_in, e.g.

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in

Episodes are marked with ``ros2 run jbl_nodes episode --timeout 5 start teleop <spot> <k>``;
dock_judge ends them. The same graph serves off-board inference: run policy_runner on a laptop
against it (README, "Deployment").

Calibrate forward_angle and hfov_rad once (docs/PROTOCOL.md sections 4 and 5) and write them in
the params file's /** section, or pass them as arguments.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from jbl_nodes.launch_common import bag_recorder, make_params, robot_arguments, robot_nodes


def _setup(context):
    return robot_nodes(context, make_params(context)) + bag_recorder(context)


def generate_launch_description():
    return LaunchDescription(robot_arguments() + [OpaqueFunction(function=_setup)])
