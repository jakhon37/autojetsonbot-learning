#!/usr/bin/env bash
# End-to-end check with the real policy_runner in the simulator (exploratory, post hoc).
#
# Per configuration: a fresh container from the benchmark image, the workspace rebuilt from the
# working tree (which has the new intra_op_threads parameter), then scripts/sim_eval.sh's own
# inner trial loop (--inner run) on the noise condition (seed 201, randomisation on, test poses
# 0-9, round 1) with bc_v4 alone, as session 2 ran a policy arm alone. intra_op_threads comes
# from an extra params file (0 = ONNX Runtime's default, what v4 ran).
#
#   bash run_e2e.sh NAME CPUSET THREADS      (CPUSET "" = not pinned)
#
# Paths as in run_matrix.sh (IMG required; WORKDIR default <repo>/sim_work/timing; PYTHON).
# e2e.log gets, per configuration: the load before the run, the container's CPU and the load at
# the end, policy_runner's start-up and warm-up lines and ONNX Runtime's pin errors from the
# launch log, the parameters the running nodes reported, and every trial's outcome.
# (Run 1, 21:50-22:04 UTC, used an earlier start-up grep that could not match a line with
# ", N intra-op threads", and logged neither the warm-up line, the parameters nor the trials.)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
HB=${WORKDIR:-$REPO/sim_work/timing}
E2E=$HB/e2e
PY=${PYTHON:-python3}
IMG=${IMG:?set IMG to the image built as TIMING.md describes}
NAME=$1 CPUSET=$2 THREADS=$3
C=jbl_e2e_$NAME
D=$E2E/$NAME
mkdir -p "$D"
LOG=$E2E/e2e.log
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

args=(-d --name "$C" -e ROS_DOMAIN_ID=89 -e PYTHONUNBUFFERED=1 -v "$REPO:/repo:ro" -v "$E2E:/work" --network host
      -v "$REPO/sim_work/v4/runs/bc/policy.onnx:/models/bc_v4/policy.onnx:ro"
      -v "$REPO/sim_work/v4/runs/bc/policy_meta.json:/models/bc_v4/policy_meta.json:ro")
[ -z "$CPUSET" ] || args+=(--cpuset-cpus "$CPUSET")
docker run "${args[@]}" "$IMG" sleep infinity > /dev/null
trap 'docker rm -f "$C" > /dev/null 2>&1' EXIT
log "e2e $NAME: container cpuset '$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$C")', intra_op_threads $THREADS"
docker exec "$C" bash -c ". /opt/ros/jazzy/setup.bash && rm -rf /ws/src/* && cp -r /repo/ros2_ws/src/* /ws/src/ \
  && cd /ws && colcon build --packages-select jbl_nodes --event-handlers console_cohesion- 2>&1 | tail -1" | tee -a "$LOG"

printf 'fake_robot:\n  ros__parameters: {camera_hfov_rad: 1.396}\npolicy_runner:\n  ros__parameters: {intra_op_threads: %d}\n' \
  "$THREADS" > "$D/params.yaml"
{ echo "trial,round,block,method,pose_id,event"
  for p in 0 1 2 3 4 5 6 7 8 9; do echo "$((p + 1)),1,$((p + 1)),bc_v4,$p,start bc_v4 $p 1"; done; } > "$D/schedule.csv"
rm -rf "$D/bag" "$D/bag_qos.yaml" "$D/trials.csv"

# Quiet host before the start: nothing outside this container.
for attempt in $(seq 1 40); do
  j=$($PY -I "$HERE/hostload.py" 3 "$(docker inspect -f '{{.Id}}' "$C")")
  other=$($PY -c 'import json,sys; print(json.loads(sys.argv[1])["other_cores"])' "$j")
  $PY -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 0.25 else 1)' "$other" && break
  log "  host busy outside the benchmark, waiting: $j"; sleep 20
done
log "  host before: $j; docker ps: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"

id=$(docker inspect -f '{{.Id}}' "$C")
u0=$(cat "/sys/fs/cgroup/cpuacct/docker/$id/cpuacct.usage"); t0=$(date +%s.%N)
docker exec -e SCHEDULE=/work/$NAME/schedule.csv -e CSV=/work/$NAME/trials.csv -e BAG=/work/$NAME/bag \
  -e LAUNCH_LOG=/work/$NAME/launch.log -e SERVO_LOG=/work/$NAME/servo.log -e PARAMS_LOG=/work/$NAME/params.txt \
  -e SEED=201 -e RANDOMIZE=true -e POSES_FILE=start_poses.yaml -e HFOV=1.396 -e EXTRA_PARAMS=/work/$NAME/params.yaml \
  -e WITH_SERVO=0 -e POLICY_METHOD=bc_v4 -e MODEL=/models/bc_v4/policy.onnx \
  "$C" bash /repo/scripts/sim_eval.sh --inner run > "$D/run.out" 2>&1
rc=$?
u1=$(cat "/sys/fs/cgroup/cpuacct/docker/$id/cpuacct.usage"); t1=$(date +%s.%N)
log "  run exit $rc; container $($PY -c "print(round(($u1-$u0)/1e9/($t1-$t0),3))") cores over $($PY -c "print(round($t1-$t0))") s; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
log "  startup: $(grep -o 'onnxruntime 1[^ ]* [A-Za-z]*Provider.* on /camera[^ ]*' "$D/launch.log" | head -1)"
log "  warm-up: $(grep -o 'warm-up inference [0-9.]* ms' "$D/launch.log" | head -1)"
grep -o 'pthread_setaffinity_np failed[^.]*' "$D/launch.log" | head -2 | sed 's/^/  ORT: /' | tee -a "$LOG"
log "  nodes report: $(tr '\n' ';' < "$D/params.txt")"
log "  rows: $(($(wc -l < "$D/trials.csv") - 1)); outcomes: $(cut -d, -f4 "$D/trials.csv" | tail -n +2 | sort | uniq -c | tr '\n' ' ')"
tail -n +2 "$D/trials.csv" | cut -d, -f1-5 | sed 's/^/  trial (method,pose_id,round,outcome,duration_s): /' | tee -a "$LOG"
echo "E2E EXIT $rc"
