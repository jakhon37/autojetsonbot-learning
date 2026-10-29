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
#   --hfov RAD        camera HFOV of the recorded stages (servo, demos, bc), given to fake_robot
#                     (camera_hfov_rad) and to dock_judge and servo_baseline (hfov_rad); default
#                     1.396, the simulator's 80 deg lens; 1.0856 is the robot's Pi Camera v2
# Environment: HTTPS_PROXY and CA_BUNDLE as in scripts/docker_test.sh; SIM_CPUSET (e.g. 0-2)
# pins the simulation container to those cores (docker run --cpuset-cpus; empty = not pinned).
# Each stage's $WORK/logs/<distro>/<stage>.provenance records the commit, the paths that differ
# from it (outside results/, and tracked under results/), the lens, SIM_CPUSET (the container
# stages), /proc/loadavg at its start and end, and the stop rules and lens the running nodes report.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
REGISTRY=${REGISTRY:-mirror.gcr.io/library}
CA_BUNDLE=${CA_BUNDLE:-}
REPO_ID=local/jbl_sim
TEST_POSES="0 1 2 3 4 5 6 7 8 9"
DEMO_SPOTS=$(seq -s ' ' 10 29)
EPISODE_TIMEOUT_S=45  # dock_judge ends every episode by 30 s; the margin covers start-up
SIM_HFOV=1.396  # fake_robot's lens (config/sim.yaml); the robot's Pi Camera v2 is 1.0856

log() { printf '\n[sim_e2e %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[sim_e2e] error: %s\n' "$*" >&2; exit 1; }
# Paths that differ from the commit, for the provenance: outside results/ (code, configs, docs;
# untracked files included) and tracked files under results/. A run's own new result files are
# untracked under results/ and not counted, so both stay 0 while the code is the commit's.
dirty_line() {
  echo "dirty: $(git -C "$REPO" status --porcelain -- . ':(exclude)results' 2>/dev/null | wc -l) paths outside" \
    "results/, $(git -C "$REPO" status --porcelain --untracked-files=no -- results 2>/dev/null | wc -l) tracked under results/"
}

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
    if [ "$rc" -eq 3 ]; then  # some nodes started it
      # PROTOCOL section 10: a crash of the method's own node is its fail, not an abort.
      # controller_node is inner_trials' local (bash scoping is dynamic). A here-string, not a
      # pipe into grep -q: SIGPIPE + pipefail.
      if ! grep -qx "/$controller_node" <<< "$(ros2 node list --no-daemon --spin-time 3 2>/dev/null || true)"; then
        EXPECT=$((EXPECT - 1)) episode end fail controller_down || true  # the live nodes get it
        echo "   $(tail -n 1 "$CSV")"
        return 0
      fi
      episode end abort "start not delivered to every node" || true  # void it, start again
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

# node.param ... -> "node.param: value" lines, as the running nodes report them (provenance)
node_params() {
  local spec pids=() dir
  dir=$(mktemp -d)
  for spec in "$@"; do
    { ros2 param get --no-daemon --spin-time 2 --hide-type "/${spec%%.*}" "${spec#*.}" > "$dir/$spec" 2>&1 || true; } &
    pids+=($!)
  done
  wait "${pids[@]}" || true
  for spec in "$@"; do echo "$spec: $(tail -n 1 "$dir/$spec")"; done
  rm -rf "$dir"
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

# METHOD CONTROLLER ROUNDS POSES CSV BAG LAUNCH_LOG PARAMS_LOG EXPECT [LAUNCH_ARGS] from the
# environment (docker exec -e on the host side).
# shellcheck disable=SC2153
inner_trials() {
  : "${METHOD:?}" "${CONTROLLER:?}" "${ROUNDS:?}" "${POSES:?}" "${CSV:?}" "${BAG:?}" "${LAUNCH_LOG:?}" "${EXPECT:?}"
  : "${PARAMS_LOG:?}"
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
  local reported=(dock_judge.request_still_v dock_judge.hfov_rad fake_robot.camera_hfov_rad)
  [ "$CONTROLLER" = servo ] && reported+=(servo_baseline.stop_hysteresis servo_baseline.hfov_rad)
  node_params "${reported[@]}" > "$PARAMS_LOG"
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

# ONNX Runtime from PyPI next to the distro's own numpy (python3-opencv is built against it),
# pinned to the versions the published runs used; the installed version is written to
# /work/logs/<distro>/onnxruntime.txt for the report's provenance.
inner_onnxruntime() {
  local ort
  case $ROS_DISTRO in
    foxy) ort=onnxruntime==1.10.0 ;;  # the last release that accepts Ubuntu 20.04's numpy 1.17
    jazzy) ort=onnxruntime==1.31.1 ;;
    *) ort=onnxruntime ;;  # not run for the published results
  esac
  if ! python3 -c "import onnxruntime" 2>/dev/null; then
    local numpy_version pip_args=()
    numpy_version=$(python3 -c "import numpy; print(numpy.__version__)")
    if grep -q -- --break-system-packages <<< "$(python3 -m pip install --help)"; then
      pip_args+=(--break-system-packages)  # Jazzy (PEP 668): the container is disposable
    fi
    # Foxy's pip 20.0 cannot read manylinux_2_27 wheel tags; a newer pip can.
    python3 -m pip install -q "${pip_args[@]}" --upgrade "pip<25" >/dev/null 2>&1 || true
    python3 -m pip install -q "${pip_args[@]}" "$ort" "numpy==$numpy_version"
  fi
  mkdir -p "/work/logs/$ROS_DISTRO"
  python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())" \
    | tee "/work/logs/$ROS_DISTRO/onnxruntime.txt"
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
HFOV=$SIM_HFOV
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
    --hfov) HFOV=$2; shift 2 ;;
    servo|demos|train|bc|report) STAGES+=("$1"); shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ ${#STAGES[@]} -gt 0 ] || STAGES=(servo demos train bc report)
case $DISTRO in foxy|humble|jazzy) ;; *) die "--distro must be foxy, humble or jazzy" ;; esac
[ "$DEMO_ROUNDS" -ge 1 ] && [ "$DEMO_ROUNDS" -le 3 ] || die "--demo-rounds must be 1, 2 or 3"
# a decimal point, so the parameter files carry a double (Humble and newer reject an int)
[[ $HFOV =~ ^[0-9]+\.[0-9]+$ ]] || die "--hfov must be radians with a decimal point, e.g. 1.0856"
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
RESULTS=${RESULTS:-$REPO/results/sim_e2e/$DISTRO}
BAGS=$WORK/bags/$DISTRO
TRIALS=$WORK/trials/$DISTRO
LOGS=$WORK/logs/$DISTRO
MODEL=$WORK/runs/bc/policy.onnx
mkdir -p "$BAGS" "$TRIALS" "$LOGS" "$WORK/params"
wants() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
# What a stage ran: the commit and the paths that differ from it (dirty_line), the lens, the CPU
# pinning and the load at its start; trials() and stage_end() add the rest. The train stage runs
# on the host: no lens, no pinning.
PROV=
provenance() {
  PROV=$LOGS/$1.provenance
  { echo "stage: $1 ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
    dirty_line
    if [ "$1" = train ]; then
      echo "SIM_CPUSET: n/a (a host stage, not pinned)"
    else
      echo "camera hfov_rad: $HFOV (fake_robot camera_hfov_rad, dock_judge and servo_baseline hfov_rad)"
      echo "SIM_CPUSET: ${SIM_CPUSET:-(not set: container not pinned)}"
    fi
    echo "loadavg at start: $(cat /proc/loadavg)"
  } > "$PROV"
}
stage_end() { echo "loadavg at end: $(cat /proc/loadavg) ($(date -u +%Y-%m-%dT%H:%M:%SZ))" >> "$PROV"; }
rounds_list() { seq -s ' ' 1 "$1"; }

CONTAINER=
cleanup() { [ -n "$CONTAINER" ] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

container() {
  [ -n "$CONTAINER" ] && return 0
  local args=(-d --name "jbl_sim_${DISTRO}_$$" -e "ROS_DOMAIN_ID=$DOMAIN_ID" -e PYTHONUNBUFFERED=1
              -v "$REPO:/repo:ro" -v "$WORK:/work")
  [ -z "${SIM_CPUSET:-}" ] || args+=(--cpuset-cpus "$SIM_CPUSET")
  if [ -n "${HTTPS_PROXY:-}" ]; then
    args+=(--network host -e "HTTPS_PROXY=$HTTPS_PROXY" -e "https_proxy=$HTTPS_PROXY")
  fi
  [ -f "$CA_BUNDLE" ] && args+=(-v "$CA_BUNDLE:/usr/local/share/ca-certificates/extra-ca.crt:ro")
  log "starting ros:$DISTRO-ros-base and building the workspace"
  CONTAINER=$(docker run "${args[@]}" "$REGISTRY/ros:$DISTRO-ros-base" sleep infinity)
  docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner setup
}

# trials <method> <controller> <rounds> <poses> <csv> <bag> [launch args...]  (paths under WORK)
# The camera HFOV (--hfov) goes to every recorded launch: fake_robot's lens through the params
# file, the judge's and the servo's through hfov_rad.
trials() {
  local method=$1 controller=$2 rounds=$3 poses=$4 csv=$5 bag=$6
  shift 6
  container
  rm -rf "${BAGS:?}/$bag" "$BAGS/${bag}_qos.yaml" "$LOGS/$bag.params.txt"
  log "$method: poses [$poses] x rounds [$rounds] -> trials/$DISTRO/$csv, bags/$DISTRO/$bag (camera HFOV $HFOV rad)"
  # Subscribers of /episode/event: fake_robot, dock_judge, trial_logger, the controller, the recorder.
  docker exec -e METHOD="$method" -e CONTROLLER="$controller" -e ROUNDS="$rounds" -e POSES="$poses" \
    -e CSV="/work/trials/$DISTRO/$csv" -e BAG="/work/bags/$DISTRO/$bag" -e EXPECT=5 \
    -e LAUNCH_LOG="/work/logs/$DISTRO/$bag.launch.log" -e PARAMS_LOG="/work/logs/$DISTRO/$bag.params.txt" \
    -e LAUNCH_ARGS="hfov_rad:=$HFOV $*" "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner trials
  { echo "launch $bag, as the nodes report it:"
    sed 's/^/  /' "$LOGS/$bag.params.txt" 2>/dev/null || echo "  (no $bag.params.txt)"
    echo "loadavg after $bag: $(cat /proc/loadavg)"
  } >> "$PROV"
}

# fake_robot's lens for the servo and bc stages (the demos carry it in demo_r<k>.yaml)
write_lens_params() {
  printf 'fake_robot:\n  ros__parameters: {camera_hfov_rad: %s}\n' "$HFOV" > "$WORK/params/lens.yaml"
  echo /work/params/lens.yaml
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
  printf 'servo_baseline:\n  ros__parameters: {%s}\nfake_robot:\n  ros__parameters: {initial_pose_id: 10, camera_hfov_rad: %s}\n' \
    "$gains" "$HFOV" > "$path"
  echo "/work/params/demo_r$round.yaml"
}

if wants servo; then
  provenance servo
  rm -f "$TRIALS/trials_servo.csv"
  trials servo servo "$(rounds_list "$ROUNDS")" "$TEST_POSES" trials_servo.csv servo \
    extra_params_file:="$(write_lens_params)"
  stage_end
fi

if wants demos; then
  provenance demos
  rm -f "$TRIALS/demos.csv"
  for r in $(rounds_list "$DEMO_ROUNDS"); do
    params=$(write_demo_params "$r")
    trials servo_demo servo "$r" "$DEMO_SPOTS" demos.csv "demos_r$r" \
      start_poses_file:=/ws/install/jbl_nodes/share/jbl_nodes/config/sim_demo_poses.yaml \
      randomize:=true seed:="$r" extra_params_file:="$params"
  done
  stage_end
fi

if wants train; then
  provenance train
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
  stage_end
fi

if wants bc; then
  [ -f "$MODEL" ] || die "no $MODEL; run the train stage first"
  provenance bc
  container
  log "installing onnxruntime in the container"
  docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner onnxruntime
  rm -f "$TRIALS/trials_bc.csv"
  trials bc policy "$(rounds_list "$ROUNDS")" "$TEST_POSES" trials_bc.csv bc \
    model_path:=/work/runs/bc/policy.onnx backend:=onnxruntime extra_params_file:="$(write_lens_params)"
  stage_end
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
  # Every slot each method was run on, so the report shows skipped slots too (a slot whose
  # start could not be delivered at all leaves no row).
  schedule=$WORK/schedule_$DISTRO.csv
  { echo method,pose_id,round
    for m in servo bc; do
      [ -s "$TRIALS/trials_$m.csv" ] || continue
      for r in $(rounds_list "$ROUNDS"); do for p in $TEST_POSES; do echo "$m,$p,$r"; done; done
    done
  } > "$schedule"
  log "analyze_trials -> ${RESULTS#"$REPO"/}"
  "$PY" "$REPO/tools/analyze_trials.py" "$combined" --out-dir "$RESULTS" --schedule "$schedule" \
    --setting "SIMULATION (fake_robot, ROS 2 $DISTRO)" "${extra[@]}"
  if [ -s "$TRIALS/trials_bc.csv" ]; then  # the model these BC trials ran
    for f in policy_meta.json metrics.jsonl split.json; do
      if [ -f "$WORK/runs/bc/$f" ]; then keep "$WORK/runs/bc/$f" "$RESULTS/bc_$f"; fi
    done
  fi
  for f in episodes.txt train_bc.log export_onnx.log; do
    if [ -f "$LOGS/$f" ]; then keep "$LOGS/$f" "$RESULTS/$f"; fi
  done
  # Provenance: what produced these numbers (each stage's commit, the image, the model).
  { echo "report: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
    echo "dirty: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) paths"
    echo "image: $(docker image inspect --format '{{index .RepoDigests 0}}' "$REGISTRY/ros:$DISTRO-ros-base" 2>/dev/null || echo unknown)"
    if [ -f "$MODEL" ]; then echo "policy.onnx sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"; fi
    if [ -f "$LOGS/onnxruntime.txt" ]; then cat "$LOGS/onnxruntime.txt"; fi
    for st in servo demos train bc; do
      if [ -f "$LOGS/$st.provenance" ]; then echo; cat "$LOGS/$st.provenance"; fi
    done
  } > "$RESULTS/provenance.txt"
  log "done: report in ${RESULTS#"$REPO"/}/report.md"
fi
