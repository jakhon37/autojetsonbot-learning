#!/usr/bin/env bash
# Record teleop demonstrations on autoJetsonBot (docs/PROTOCOL.md section 6).
#
# Run inside the robot's Foxy container (autoJetsonBot: ./robot.sh shell) after the robot's own
# bring-up (./robot.sh robot: diffdrive_node on /cmd_vel, the RPLidar on /scan) and after
# building this workspace there (cd ros2_ws && colcon build --symlink-install).
#
#   bash scripts/record_robot.sh                                  # calibration from config/robot.yaml
#   bash scripts/record_robot.sh forward_angle:=3.14 hfov_rad:=1.086
#
# Starts robot_record.launch.py (camera_node, safety_gate, dock_judge, trial_logger) with a bag
# of the INTERFACES topics in $BAG_DIR/demos_<date>_<time> and the per-episode log in
# $TRIALS_DIR/<date>/demos.csv. Extra name:=value arguments go to the launch file. Ctrl-C stops
# everything and closes the bag. Teleop and the episode CLI run in other terminals (printed below).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
BAG_DIR=${BAG_DIR:-$HOME/jbl_bags}
TRIALS_DIR=${TRIALS_DIR:-$HOME/jbl_trials}
MIN_FREE_GB=5  # the 15 Hz JPEG stream is about 2 GB per hour

warn() { printf '\033[1;33m[record_robot] %s\033[0m\n' "$*" >&2; }
die() { printf '[record_robot] error: %s\n' "$*" >&2; exit 1; }

[ -n "${ROS_DISTRO:-}" ] || die "source /opt/ros/<distro>/setup.bash first"
[ -f "$REPO/ros2_ws/install/setup.bash" ] || die "build the workspace first: cd $REPO/ros2_ws && colcon build"
set +u  # ROS setup scripts read unset variables
# shellcheck source=/dev/null
. "$REPO/ros2_ws/install/setup.bash"
set -u

stamp=$(date +%Y%m%d_%H%M)
bag=$BAG_DIR/demos_$stamp
csv=$TRIALS_DIR/$(date +%Y-%m-%d)/demos.csv
mkdir -p "$BAG_DIR" "$(dirname "$csv")"

free_gb=$(df -Pk "$BAG_DIR" | awk 'NR == 2 {print int($4 / 1048576)}')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || warn "only ${free_gb} GB free in $BAG_DIR (about 2 GB per hour of recording)"

# Robot bring-up: someone must publish /scan and /odom, and nothing but the gate may drive /cmd_vel.
topics=$(ros2 topic list 2>/dev/null || true)
for t in /scan /odom; do
  grep -qx "$t" <<< "$topics" || warn "no $t yet: is the robot bring-up (./robot.sh robot) running?"
done
cmd_vel_publishers=$(ros2 topic info /cmd_vel 2>/dev/null | awk '/Publisher count/ {print $3}' || true)
if [ "${cmd_vel_publishers:-0}" -gt 0 ]; then
  warn "/cmd_vel already has $cmd_vel_publishers publisher(s); only safety_gate may publish it."
  warn "Remap teleop / the web pad to /cmd_vel_in, or the gate is bypassed."
fi

cat <<EOF
Recording to $bag
Episodes logged to $csv

In other terminals (docker exec -it <container> bash, then source the workspace):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in
  ros2 run jbl_nodes episode --timeout 5 start teleop <spot_id> <k>
  ros2 run jbl_nodes episode --timeout 5 end abort <reason>      # a demo you would not imitate
dock_judge ends each episode itself (end success / end fail ...). Ctrl-C here when done.

EOF
exec ros2 launch jbl_nodes robot_record.launch.py bag:="$bag" trials_csv:="$csv" "$@"
