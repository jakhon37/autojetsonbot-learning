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
#
# Data goes to $DATA_DIR [<this checkout>/data/robot]: inside autoJetsonBot's container the
# checkout is /autonomous_ROS/autojetsonbot-learning, on the bind mount, so the bags survive the
# container and appear on the Jetson host (copy them off with rsync, README "Training"). $HOME
# (/root) inside the container is neither persistent nor visible on the host.
#
# Before launching it opens the camera_node device from the params file(s) and reads one frame
# (skipped with camera_source:=test_pattern or SKIP_CAMERA_CHECK=1).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO/data/robot}
BAG_DIR=${BAG_DIR:-$DATA_DIR/bags}
TRIALS_DIR=${TRIALS_DIR:-$DATA_DIR/trials}
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

# Camera preflight: the launch is exec'd, so check the device here, the way camera_node opens it.
camera_preflight() {
  local params=("$(ros2 pkg prefix jbl_nodes)/share/jbl_nodes/config/robot.yaml") a
  for a in "$@"; do
    case $a in
      camera_source:=test_pattern) return 0 ;;
      params_file:=*) params[0]=${a#params_file:=} ;;
      extra_params_file:=*) params+=("${a#extra_params_file:=}") ;;
    esac
  done
  python3 - "${params[@]}" <<'PY'
import sys

import yaml
from jbl_nodes.camera_node import OpenCvSource, parse_device

device, size = "0", [640, 480]
for path in sys.argv[1:]:
    if not path:
        continue
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    for section in ("/**", "camera_node"):
        values = (doc.get(section) or {}).get("ros__parameters") or {}
        device = values.get("device", device)
        size = [values.get("width", size[0]), values.get("height", size[1])]
source = OpenCvSource(parse_device(device), int(size[0]), int(size[1]), 30.0)
print("camera check: opening %s" % source.describe())
ok = source.open() and source.grab() and source.retrieve() is not None
source.close()
sys.exit(0 if ok else 1)
PY
}
if [ "${SKIP_CAMERA_CHECK:-0}" != 1 ] && ! camera_preflight "$@"; then
  die "cannot read a frame from camera_node's device. On the Nano's CSI camera the container needs
  --runtime nvidia -v /tmp/argus_socket:/tmp/argus_socket (docs/PROTOCOL.md section 4; check with
  gst-inspect-1.0 nvarguscamerasrc), and nothing else (camera_go.py) may hold the camera.
  docs/INTERFACES.md, Robot facts, has the capture details."
fi

cat <<EOF
Recording to $bag
Episodes logged to $csv

Teleop must publish /cmd_vel_in: the web dashboard with its Command Velocity topic set to
/cmd_vel_in (it publishes 10 Hz while a key or button is held), or in other terminals
(docker exec -it <container> bash, then source the workspace):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in
  ros2 run jbl_nodes episode --timeout 5 start teleop <spot_id> <k>
  ros2 run jbl_nodes episode --timeout 5 end abort <reason>      # a demo you would not imitate
dock_judge ends each episode itself (end success / end fail ...). Ctrl-C here when done.

EOF
exec ros2 launch jbl_nodes robot_record.launch.py bag:="$bag" trials_csv:="$csv" "$@"
