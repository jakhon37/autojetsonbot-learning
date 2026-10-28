#!/usr/bin/env bash
# The whole pipeline in simulation: servo baseline trials, scripted demonstrations, LeRobot
# dataset, BC training and ONNX export, BC trials with ONNX Runtime, report. ROS runs in a clean
# ros:<distro>-ros-base container with fake_robot standing in for the robot; conversion,
# training and analysis run on the host (Python with requirements-tools.txt).
#
#   bash scripts/sim_e2e.sh                                # all stages on Jazzy (~50 min on 4 CPU cores)
#   bash scripts/sim_e2e.sh servo                          # servo baseline trials only (~5 min)
#   bash scripts/sim_e2e.sh --distro foxy servo bc report  # reuse the trained model on Foxy
#
# Stages, in this order (default: all):
#   servo   10 test poses x --rounds, method "servo", recorded              (container)
#   demos   20 training spots (config/sim_demo_poses.yaml) x --demo-rounds,
#           servo_baseline as "servo_demo" with varied gains and fake_robot noise, recorded
#   train   bags -> dataset (bag_to_lerobot) -> BC on CPU from scratch (train_bc, servo_demo
#           successes only, so the 10 test poses stay held out) -> ONNX (export_onnx)   (host)
#   bc      10 test poses x --rounds with policy_runner + onnxruntime, recorded (container)
#   report  analyze_trials.py on servo + bc; copies the small outputs to --results      (host)
#
# Options
#   --distro D        foxy | humble | jazzy (default jazzy)
#   --work DIR        bags, dataset, model, logs (default <repo>/sim_work; large, gitignored)
#   --results DIR     small outputs (default <repo>/results/sim_e2e/<distro>)
#   --python PY       host Python (default $PYTHON, else python3)
#   --rounds N        rounds over the 10 test poses for servo and bc (default 1)
#   --demo-rounds N   demonstration rounds over the 20 training spots, 1-3 (default 2)
#   --epochs N        BC epochs (default 5, about 4-5 min each on 4 CPU cores)
#   --domain-id N     ROS_DOMAIN_ID inside the container (default 23)
# Environment: HTTPS_PROXY and CA_BUNDLE as in scripts/docker_test.sh.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
REGISTRY=${REGISTRY:-mirror.gcr.io/library}
CA_BUNDLE=${CA_BUNDLE:-/root/.ccr/ca-bundle.crt}
REPO_ID=local/jbl_sim
TEST_POSES="0 1 2 3 4 5 6 7 8 9"
DEMO_SPOTS=$(seq -s ' ' 10 29)
EPISODE_TIMEOUT_S=45  # dock_judge ends every episode by 30 s; the margin covers start-up

log() { printf '\n[sim_e2e %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[sim_e2e] error: %s\n' "$*" >&2; exit 1; }

# ============================================================== inside the container

csv_rows() { if [ -s "$1" ]; then echo $(($(wc -l < "$1") - 1)); else echo 0; fi; }

episode() { ros2 run jbl_nodes episode --timeout 10 --expect "$EXPECT" "$@"; }

# Start one episode, wait for its row in the CSV (dock_judge ends it, trial_logger writes it).
run_episode() {
  local method=$1 pose=$2 round=$3 rc=1 before deadline _
  for _ in 1 2 3; do
    before=$(csv_rows "$CSV")
    rc=0
    episode start "$method" "$pose" "$round" || rc=$?
    [ "$rc" -eq 0 ] && break
    if [ "$rc" -eq 3 ]; then  # some nodes started it: void it, then start again (an abort row)
      episode end abort "start not delivered to every node" || true
      sleep 1
    fi
  done
  if [ "$rc" -ne 0 ]; then
    echo "!! could not deliver 'start $method $pose $round'; slot skipped"
    return 0
  fi
  deadline=$((SECONDS + EPISODE_TIMEOUT_S))
  while [ "$(csv_rows "$CSV")" -le "$before" ]; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "!! no verdict for '$method $pose $round' within ${EPISODE_TIMEOUT_S} s; aborting it"
      episode end abort "no verdict within ${EPISODE_TIMEOUT_S} s" || true
      break
    fi
    sleep 0.5
  done
  echo "   $(tail -n 1 "$CSV")"
  sleep 1  # let the controller's final zero and the gate settle before the next teleport
}

wait_for_nodes() {
  local deadline=$((SECONDS + 60)) missing nodes
  while :; do
    missing=""
    # No daemon: with --network host a stale daemon of another container on the same domain
    # answers instead (and sees nothing). Not piped into grep -q: SIGPIPE + pipefail.
    nodes=$(ros2 node list --no-daemon --spin-time 3 2>/dev/null || true)
    for n in "$@"; do grep -qx "/$n" <<< "$nodes" || missing="$missing $n"; done
    [ -z "$missing" ] && return 0
    [ "$SECONDS" -ge "$deadline" ] && die "nodes did not come up:$missing (see $LAUNCH_LOG)"
    sleep 1
  done
}

# METHOD CONTROLLER ROUNDS POSES CSV BAG LAUNCH_LOG EXPECT [LAUNCH_ARGS] from the environment
# (docker exec -e on the host side).
# shellcheck disable=SC2153
inner_trials() {
  : "${METHOD:?}" "${CONTROLLER:?}" "${ROUNDS:?}" "${POSES:?}" "${CSV:?}" "${BAG:?}" "${LAUNCH_LOG:?}" "${EXPECT:?}"
  # shellcheck source=/dev/null
  . "/opt/ros/$ROS_DISTRO/setup.bash"
  # shellcheck source=/dev/null
  . /ws/install/setup.bash
  mkdir -p "$(dirname "$CSV")" "$(dirname "$LAUNCH_LOG")"
  # Job control gives the background launch its own process group and keeps SIGINT working
  # for it; without it bash starts background jobs with SIGINT ignored.
  set -m
  # shellcheck disable=SC2086  # LAUNCH_ARGS is a list of name:=value words
  ros2 launch jbl_nodes sim_dock.launch.py method:="$METHOD" controller:="$CONTROLLER" \
    trials_csv:="$CSV" bag:="$BAG" ${LAUNCH_ARGS:-} > "$LAUNCH_LOG" 2>&1 &
  local launch_pid=$!
  trap 'kill -INT '"$launch_pid"' 2>/dev/null || true; wait '"$launch_pid"' || true' EXIT
  local controller_node=servo_baseline
  [ "$CONTROLLER" = policy ] && controller_node=policy_runner
  wait_for_nodes fake_robot safety_gate dock_judge trial_logger "$controller_node"
  sleep 3  # the recorder subscribes to new topics on its next poll
  local round pose
  for round in $ROUNDS; do
    for pose in $POSES; do
      run_episode "$METHOD" "$pose" "$round"
    done
  done
  kill -INT "$launch_pid"
  wait "$launch_pid" || true
  trap - EXIT
  [ -f "$BAG/metadata.yaml" ] || die "the recorder did not finish $BAG"
}

inner_setup() {
  update-ca-certificates >/dev/null 2>&1 || true
  # apt over HTTPS only (plain-HTTP mirrors may be blocked by a proxy); the ROS apt repo is not needed
  sed -i "s|http://\(archive\|security\|ports\).ubuntu.com|https://\1.ubuntu.com|g" /etc/apt/sources.list 2>/dev/null || true
  for f in /etc/apt/sources.list.d/*; do
    [ -f "$f" ] && sed -i "s|http://\(archive\|security\|ports\).ubuntu.com|https://\1.ubuntu.com|g" "$f"
  done
  rm -f /etc/apt/sources.list.d/ros2*
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null
  apt-get install -y -qq python3-opencv python3-numpy python3-yaml python3-pip >/dev/null
  # shellcheck source=/dev/null
  . "/opt/ros/$ROS_DISTRO/setup.bash"
  mkdir -p /ws/src && cp -r /repo/ros2_ws/src/* /ws/src/
  cd /ws && colcon build --event-handlers console_cohesion- --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
}

# ONNX Runtime from PyPI next to the distro's own numpy (python3-opencv is built against it).
inner_onnxruntime() {
  if python3 -c "import onnxruntime" 2>/dev/null; then return 0; fi
  local numpy_version pip_args=()
  numpy_version=$(python3 -c "import numpy; print(numpy.__version__)")
  if grep -q -- --break-system-packages <<< "$(python3 -m pip install --help)"; then
    pip_args+=(--break-system-packages)  # Jazzy (PEP 668): the container is disposable
  fi
  # Foxy's pip 20.0 cannot read manylinux_2_27 wheel tags; a newer pip can.
  python3 -m pip install -q "${pip_args[@]}" --upgrade "pip<25" >/dev/null 2>&1 || true
  python3 -m pip install -q "${pip_args[@]}" onnxruntime "numpy==$numpy_version"
  python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"
}

if [ "${1:-}" = --inner ]; then
  shift
  set +u  # the ROS setup scripts read unset variables
  case $1 in
    setup) inner_setup ;;
    onnxruntime) inner_onnxruntime ;;
    trials) inner_trials ;;
    *) die "unknown inner command $1" ;;
  esac
  exit 0
fi

# ============================================================== on the host

DISTRO=jazzy
WORK=$REPO/sim_work
RESULTS=
PY=${PYTHON:-python3}
ROUNDS=1
DEMO_ROUNDS=2
EPOCHS=5
DOMAIN_ID=23
STAGES=()
while [ $# -gt 0 ]; do
  case $1 in
    --distro) DISTRO=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --results) RESULTS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --rounds) ROUNDS=$2; shift 2 ;;
    --demo-rounds) DEMO_ROUNDS=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    --domain-id) DOMAIN_ID=$2; shift 2 ;;
    servo|demos|train|bc|report) STAGES+=("$1"); shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ ${#STAGES[@]} -gt 0 ] || STAGES=(servo demos train bc report)
case $DISTRO in foxy|humble|jazzy) ;; *) die "--distro must be foxy, humble or jazzy" ;; esac
[ "$DEMO_ROUNDS" -ge 1 ] && [ "$DEMO_ROUNDS" -le 3 ] || die "--demo-rounds must be 1, 2 or 3"
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
RESULTS=${RESULTS:-$REPO/results/sim_e2e/$DISTRO}
BAGS=$WORK/bags/$DISTRO
TRIALS=$WORK/trials/$DISTRO
LOGS=$WORK/logs/$DISTRO
MODEL=$WORK/runs/bc/policy.onnx
mkdir -p "$BAGS" "$TRIALS" "$LOGS" "$WORK/params"
wants() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
rounds_list() { seq -s ' ' 1 "$1"; }

CONTAINER=
cleanup() { [ -n "$CONTAINER" ] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

container() {
  [ -n "$CONTAINER" ] && return 0
  local args=(-d --name "jbl_sim_${DISTRO}_$$" -e "ROS_DOMAIN_ID=$DOMAIN_ID" -e PYTHONUNBUFFERED=1
              -v "$REPO:/repo:ro" -v "$WORK:/work")
  if [ -n "${HTTPS_PROXY:-}" ]; then
    args+=(--network host -e "HTTPS_PROXY=$HTTPS_PROXY" -e "https_proxy=$HTTPS_PROXY")
  fi
  [ -f "$CA_BUNDLE" ] && args+=(-v "$CA_BUNDLE:/usr/local/share/ca-certificates/extra-ca.crt:ro")
  log "starting ros:$DISTRO-ros-base and building the workspace"
  CONTAINER=$(docker run "${args[@]}" "$REGISTRY/ros:$DISTRO-ros-base" sleep infinity)
  docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner setup
}

# trials <method> <controller> <rounds> <poses> <csv> <bag> [launch args...]  (paths under WORK)
trials() {
  local method=$1 controller=$2 rounds=$3 poses=$4 csv=$5 bag=$6
  shift 6
  container
  rm -rf "${BAGS:?}/$bag" "$BAGS/${bag}_qos.yaml"
  log "$method: poses [$poses] x rounds [$rounds] -> trials/$DISTRO/$csv, bags/$DISTRO/$bag"
  # Subscribers of /episode/event: fake_robot, dock_judge, trial_logger, the controller, the recorder.
  docker exec -e METHOD="$method" -e CONTROLLER="$controller" -e ROUNDS="$rounds" -e POSES="$poses" \
    -e CSV="/work/trials/$DISTRO/$csv" -e BAG="/work/bags/$DISTRO/$bag" -e EXPECT=5 \
    -e LAUNCH_LOG="/work/logs/$DISTRO/$bag.launch.log" -e LAUNCH_ARGS="$*" \
    "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner trials
}

# Servo gains per demonstration round: the defaults, then slower and faster variants, so the
# demonstrations are not one deterministic trajectory per spot (fake_robot adds wheel slip,
# start jitter and image noise on top).
write_demo_params() {
  local round=$1 path=$WORK/params/demo_r$1.yaml
  local gains
  case $round in
    1) gains="k_v: 1.0, k_w: 1.2, v_max: 0.15, w_max: 0.8, search_w: 0.5" ;;
    2) gains="k_v: 0.8, k_w: 1.5, v_max: 0.12, w_max: 0.7, search_w: 0.6" ;;
    3) gains="k_v: 1.2, k_w: 1.0, v_max: 0.18, w_max: 0.9, search_w: 0.4" ;;
  esac
  printf 'servo_baseline:\n  ros__parameters: {%s}\nfake_robot:\n  ros__parameters: {initial_pose_id: 10}\n' \
    "$gains" > "$path"
  echo "/work/params/demo_r$round.yaml"
}

if wants servo; then
  rm -f "$TRIALS/trials_servo.csv"
  trials servo servo "$(rounds_list "$ROUNDS")" "$TEST_POSES" trials_servo.csv servo
fi

if wants demos; then
  rm -f "$TRIALS/demos.csv"
  for r in $(rounds_list "$DEMO_ROUNDS"); do
    params=$(write_demo_params "$r")
    trials servo_demo servo "$r" "$DEMO_SPOTS" demos.csv "demos_r$r" \
      start_poses_file:=/ws/install/jbl_nodes/share/jbl_nodes/config/sim_demo_poses.yaml \
      randomize:=true seed:="$r" extra_params_file:="$params"
  done
fi

if wants train; then
  bags=()
  for b in "$BAGS"/servo "$BAGS"/demos_r*; do [ -f "$b/metadata.yaml" ] && bags+=("$b"); done
  [ ${#bags[@]} -gt 0 ] || die "no servo/demo bags in $BAGS; run the servo and demos stages first"
  export HF_HUB_OFFLINE=1
  log "bag_to_lerobot: ${bags[*]#"$WORK"/}"
  "$PY" "$REPO/tools/bag_to_lerobot.py" "${bags[@]}" --root "$WORK/dataset" --repo-id "$REPO_ID" \
    --overwrite 2>&1 | tee "$LOGS/bag_to_lerobot.log" | tail -n 15
  "$PY" "$REPO/tools/jbl_episodes.py" --root "$WORK/dataset" 2>&1 | tee "$LOGS/episodes.txt" | tail -n 12
  log "train_bc: $EPOCHS epochs on CPU, ResNet18 from scratch, servo_demo successes, validation on held-out spots"
  "$PY" "$REPO/tools/train_bc.py" --root "$WORK/dataset" --repo-id "$REPO_ID" --out "$WORK/runs/bc" \
    --overwrite --epochs "$EPOCHS" --batch-size 32 --device cpu --methods servo_demo --outcomes success \
    --split-by pose_id --val-fraction 0.2 --seed 0 2>&1 | tee "$LOGS/train_bc.log" | tail -n 15
  log "export_onnx"
  "$PY" "$REPO/tools/export_onnx.py" "$WORK/runs/bc/model.pt" --out "$MODEL" \
    --dataset-root "$WORK/dataset" --repo-id "$REPO_ID" 2>&1 | tee "$LOGS/export_onnx.log" | tail -n 8
fi

if wants bc; then
  [ -f "$MODEL" ] || die "no $MODEL; run the train stage first"
  container
  log "installing onnxruntime in the container"
  docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner onnxruntime
  rm -f "$TRIALS/trials_bc.csv"
  trials bc policy "$(rounds_list "$ROUNDS")" "$TEST_POSES" trials_bc.csv bc \
    model_path:=/work/runs/bc/policy.onnx backend:=onnxruntime
fi

if wants report; then
  mkdir -p "$RESULTS"
  combined=$RESULTS/trials.csv
  first=1
  : > "$combined"
  for part in "$TRIALS/trials_servo.csv" "$TRIALS/trials_bc.csv"; do
    [ -s "$part" ] || continue
    if [ $first -eq 1 ]; then cat "$part" >> "$combined"; first=0; else tail -n +2 "$part" >> "$combined"; fi
  done
  [ $first -eq 0 ] || die "no trial CSVs in $TRIALS"
  # Copies for the results folder, with paths relative to the repository.
  keep() { sed "s|$REPO/||g" "$1" > "$2"; }
  if [ -s "$TRIALS/demos.csv" ]; then cp "$TRIALS/demos.csv" "$RESULTS/demos.csv"; fi
  extra=()
  if [ -f "$BAGS/bc/metadata.yaml" ]; then extra+=(--latency "sim-cpu-onnxruntime=$BAGS/bc"); fi
  for b in servo bc; do
    if [ -f "$BAGS/$b/metadata.yaml" ]; then extra+=(--gate "$b=$BAGS/$b"); fi
  done
  log "analyze_trials -> ${RESULTS#"$REPO"/}"
  "$PY" "$REPO/tools/analyze_trials.py" "$combined" --out-dir "$RESULTS" \
    --setting "SIMULATION (fake_robot, ROS 2 $DISTRO)" "${extra[@]}"
  if [ -s "$TRIALS/trials_bc.csv" ]; then  # the model these BC trials ran
    for f in policy_meta.json metrics.jsonl split.json; do
      if [ -f "$WORK/runs/bc/$f" ]; then keep "$WORK/runs/bc/$f" "$RESULTS/bc_$f"; fi
    done
  fi
  for f in episodes.txt train_bc.log export_onnx.log; do
    if [ -f "$LOGS/$f" ]; then keep "$LOGS/$f" "$RESULTS/$f"; fi
  done
  log "done: report in ${RESULTS#"$REPO"/}/report.md"
fi
