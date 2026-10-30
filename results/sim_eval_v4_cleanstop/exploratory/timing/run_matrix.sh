#!/usr/bin/env bash
# Benchmark matrix for the v4 timing investigation (exploratory, post hoc).
#
# Two containers from the same image (ros:jazzy-ros-base + the workspace built by
# scripts/sim_e2e.sh --inner setup + onnxruntime 1.31.1 by --inner onnxruntime): "free" (no
# cpuset, as v3 ran) and "pin" (--cpuset-cpus 0-2, as v4 ran). In each, harness.py runs the
# repo's decode + preprocess + ONNX Runtime backend on recorded frames at 10 Hz, per
# configuration N measured ticks after WARM warm-up ticks, with the background idle, the
# simulator's own nodes (sim_dock.launch.py method:=none, recording a bag, as in a session) or
# a calibrated busy loop (2 processes x 80 % = 1.6 cores) running in the same container.
# Before every run: nothing may run outside the two containers (other_cores < 0.25 over 3 s,
# else wait), and the load is logged.
#
# Paths (the run used scratch paths; this copy takes them from the environment):
#   IMG      the image (TIMING.md, "Files and how to reproduce", builds it); required
#   WORKDIR  host folder mounted as /bench: frames_v4_noise.npz, frames_v3_noise.npz
#            (extract_frames.py) and the outputs; default <repo>/sim_work/timing
#   PYTHON   host Python for hostload.py (default python3)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
HB=${WORKDIR:-$REPO/sim_work/timing}
mkdir -p "$HB" && HB=$(cd "$HB" && pwd)
PY=${PYTHON:-python3}
IMG=${IMG:?set IMG to the image built as TIMING.md describes}
cp "$HERE/harness.py" "$HERE/busy.py" "$HB/"  # what the containers run, as /bench/*.py
N=${N:-400}
WARM=${WARM:-20}
OUTNAME=${OUTNAME:-results}
LOG=$HB/$OUTNAME.log
L=logs_$OUTNAME
mkdir -p "$HB/$L"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
cid() { docker inspect -f '{{.Id}}' "$1"; }
ns() { cat "/sys/fs/cgroup/cpuacct/docker/$1/cpuacct.usage"; }

start_container() {  # name domain_id [cpuset]
  local args=(-d --name "$1" -e "ROS_DOMAIN_ID=$2" -e PYTHONUNBUFFERED=1 -v "$REPO:/repo:ro" -v "$HB:/bench"
              --network host)
  local m
  for m in v4 v2; do
    args+=(-v "$REPO/sim_work/$m/runs/bc/policy.onnx:/models/bc_$m/policy.onnx:ro"
           -v "$REPO/sim_work/$m/runs/bc/policy_meta.json:/models/bc_$m/policy_meta.json:ro")
  done
  [ -z "${3:-}" ] || args+=(--cpuset-cpus "$3")
  docker run "${args[@]}" "$IMG" sleep infinity > /dev/null
  log "container $1: cpuset '$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$1")', ROS_DOMAIN_ID $2"
}

quiet_check() {
  local ids j other attempt
  for attempt in $(seq 1 40); do
    ids=$(docker ps -q --no-trunc | tr '\n' ' ')
    # shellcheck disable=SC2086
    j=$($PY -I "$HERE/hostload.py" 3 $ids)
    other=$($PY -c 'import json,sys; print(json.loads(sys.argv[1])["other_cores"])' "$j")
    if $PY -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 0.25 else 1)' "$other"; then
      log "  host: $j; docker ps: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
      return 0
    fi
    log "  host busy outside the benchmark, waiting: $j"
    log "  top: $(top -bn1 | sed -n '8,12p' | tr -s ' ' | tr '\n' '|')"
    sleep 20
  done
  log "  host never went quiet; running anyway"
}

run_cfg() {  # container label model frames threads [spin]
  local c=$1 label=$2 model=$3 frames=$4 threads=$5 spin=${6:-default} id u0 u1 t0 t1
  quiet_check
  id=$(cid "$c")
  log "RUN $label ($c, cpuset '$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$c")', $model, threads $threads, spin $spin)"
  u0=$(ns "$id"); t0=$(date +%s.%N)
  docker exec "$c" bash -c ". /opt/ros/jazzy/setup.bash; . /ws/install/setup.bash; cd /bench && \
    python3 harness.py --model $model --frames $frames --threads $threads --spin $spin --n $N --warmup $WARM \
    --label '$label' --out /bench/$OUTNAME.jsonl" > "$HB/$L/$label.out" 2>&1
  u1=$(ns "$id"); t1=$(date +%s.%N)
  log "  done $label: container $($PY -c "print(round(($u1-$u0)/1e9/($t1-$t0),3))") cores over the run; $(grep -o '"infer_ms": {[^}]*}' "$HB/$L/$label.out")"
  grep -h 'E:onnxruntime' "$HB/$L/$label.out" | sed 's/^/  ORT: /' | cut -c1-260 | tee -a "$LOG" > /dev/null
}

bg_measure() {  # container label: the background alone, 15 s
  local j
  j=$($PY -I "$HERE/hostload.py" 15 "$(cid "$1")")
  log "  background $2 alone in $1: $j"
}

start_sim() {  # container
  docker exec -d "$1" bash -c ". /opt/ros/jazzy/setup.bash; . /ws/install/setup.bash; \
    exec ros2 launch jbl_nodes sim_dock.launch.py method:=none bag:=/bench/bg_$1 > /bench/$L/bg_$1.launch.log 2>&1"
  sleep 25
  log "  sim nodes in $1: $(docker exec "$1" bash -c '. /opt/ros/jazzy/setup.bash; ros2 node list --no-daemon --spin-time 3 2>/dev/null' | tr '\n' ' ')"
  bg_measure "$1" sim
}

stop_sim() {
  docker exec "$1" pkill -INT -f 'ros2 launch' || true
  local i
  for i in $(seq 1 30); do
    docker exec "$1" pgrep -f 'ros2|fake_robot|safety_gate|dock_judge|trial_logger' > /dev/null || break
    sleep 1
  done
  docker exec "$1" pkill -KILL -f 'ros2|fake_robot|safety_gate|dock_judge|trial_logger' 2>/dev/null || true
  du -sh "$HB/bg_$1" 2>/dev/null | sed 's/^/  bag: /' | tee -a "$LOG" > /dev/null
  rm -rf "$HB/bg_$1" "$HB/bg_$1_qos.yaml"
}

start_busy() {
  docker exec -d "$1" bash -c "exec python3 /bench/busy.py --procs 2 --duty 0.8 --seconds 7200 > /bench/$L/busy_$1.log 2>&1"
  sleep 3
  bg_measure "$1" busy
}

stop_busy() {
  docker exec "$1" pkill -TERM -f busy.py || true
  sleep 2
  log "  $(tr '\n' ' ' < "$HB/$L/busy_$1.log")"
}

M4=/models/bc_v4/policy.onnx
M2=/models/bc_v2/policy.onnx
F4=frames_v4_noise.npz
F3=frames_v3_noise.npz

log "=== matrix start: N=$N WARM=$WARM, image $IMG ($(docker image inspect -f '{{.Id}}' "$IMG" | cut -c1-19))"
start_container jbl_timing_free 87
start_container jbl_timing_pin 88 0-2

PHASES=${PHASES:-"idle sim busy repeat"}
for phase in $PHASES; do
  case $phase in
    idle)
      for c in free pin; do
        for t in default 1 2 3 4; do run_cfg "jbl_timing_$c" "idle_${c}_t$t" $M4 $F4 "$t"; done
      done ;;
    sim)
      for c in free pin; do
        start_sim "jbl_timing_$c"
        for t in default 1 2 3 4; do run_cfg "jbl_timing_$c" "sim_${c}_t$t" $M4 $F4 "$t"; done
        run_cfg "jbl_timing_$c" "sim_${c}_tdefault_bcv2" $M2 $F3 default
        [ "$c" = pin ] && run_cfg "jbl_timing_$c" "sim_${c}_tdefault_nospin" $M4 $F4 default 0
        bg_measure "jbl_timing_$c" sim-after
        stop_sim "jbl_timing_$c"
      done ;;
    busy)
      for c in free pin; do
        start_busy "jbl_timing_$c"
        for t in default 1 2 3; do run_cfg "jbl_timing_$c" "busy_${c}_t$t" $M4 $F4 "$t"; done
        stop_busy "jbl_timing_$c"
      done ;;
    repeat)
      for c in free pin; do
        start_sim "jbl_timing_$c"
        run_cfg "jbl_timing_$c" "rep_sim_${c}_tdefault" $M4 $F4 default
        run_cfg "jbl_timing_$c" "rep_sim_${c}_t2" $M4 $F4 2
        stop_sim "jbl_timing_$c"
      done ;;
    threads)  # added after the review (PHASES=threads): 3 against 2 threads again, pinned, under
      # the simulator's nodes, after one default run that checks the new image against the first
      start_sim jbl_timing_pin
      run_cfg jbl_timing_pin rep_sim_pin_tdefault_r2 $M4 $F4 default
      for r in 2 3; do
        run_cfg jbl_timing_pin "rep_sim_pin_t3_r$r" $M4 $F4 3
        run_cfg jbl_timing_pin "rep_sim_pin_t2_r$r" $M4 $F4 2
      done
      stop_sim jbl_timing_pin ;;
  esac
done
docker rm -f jbl_timing_free jbl_timing_pin > /dev/null
log "=== matrix end"
echo "MATRIX EXIT 0"
