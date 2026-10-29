#!/usr/bin/env bash
# Harder SIMULATION evaluation of servo vs BC (results/sim_eval_v2): randomised conditions,
# poses beyond the training distribution and a camera-HFOV mismatch, 10 poses x 3 rounds per
# condition, servo and bc interleaved in one launch per a tools/make_schedule.py schedule.
#
#   bash scripts/sim_eval.sh                      # all conditions on Jazzy (~35 min each)
#   bash scripts/sim_eval.sh noise hfov62         # only these
#   bash scripts/sim_eval.sh --report-only        # re-analyse what is recorded
#   bash scripts/sim_eval.sh --work sim_work/v2 --results results/sim_eval_v3_bcfix \
#       --methods bc_v2 --pair-with results/sim_eval_v2 clean_ref noise wide hfov62
#                                                 # one new arm, paired with the v2 arms
#
# Conditions (fake_robot randomize:=true in all but clean_ref, so rounds are not replicas):
#   clean_ref  test poses 0-9, randomize:=false, sim camera HFOV 80 deg and always 1 round
#            (without randomisation a second round replays the first): the setup of the
#            scripts/sim_e2e.sh trials. Only run when named.
#   noise    test poses 0-9 (config/start_poses.yaml), sim camera HFOV 80 deg; the default
#            randomisation: per-episode brightness gain 0.7-1.3 and offset +-20, pixel noise
#            sigma 3, wheel bias sigma 2 % and slip sigma 3 %, lidar noise sigma 5 mm, start
#            jitter +-3 cm / +-3 deg (sim_world.NoiseConfig). The BC demonstrations were
#            recorded with the same randomisation, so the images are in distribution.
#   wide     config/sim_eval_wide_poses.yaml: 10 start poses outside the demonstrations'
#            distance / lateral / heading box (far, wide, steep, near), otherwise as noise
#   hfov62   test poses 0-9 with the robot's Pi Camera v2 lens (HFOV 62.2 deg = 1.0856 rad) in
#            fake_robot; dock_judge and servo_baseline are told the true HFOV (hfov_rad, as
#            config/robot.yaml does on the robot); BC was trained on 80 deg images and has no
#            such parameter. Otherwise as noise.
#
# Arms (--methods): servo is servo_baseline; any other name is policy_runner with the model,
# answering to "start <name> ...", so the name is what trials.csv and the bags record. At most
# one policy arm per session; without servo no servo node runs.
#
# Resumable: slots with a scored row in trials.csv are skipped; each launch records a new
# bag segment. Work files: --work/eval_v2/<distro>/<condition> (this session's trials.csv,
# schedule.csv, episodes_gt.csv, provenance.txt and bags); small outputs:
# --results/<condition> and --results/SUMMARY_table.md (scripts/sim_eval_summary.py), which
# also lists --results/clean_ref if present (in v2: the clean scripts/sim_e2e.sh trials with
# their ground truth, made by hand as SUMMARY.md says).
#
# --pair-with DIR (repeatable) puts the rows of DIR/<condition> (trials.csv, episodes_gt.csv,
# schedule.csv; an earlier --results folder or another session's work folder) in front of
# this session's in --results/<condition>, so the arms are paired on pose_id x round. The
# sessions share each condition's seed, poses and schedule rule but not its noise draws
# (one random stream per launch, consumed in timing-dependent order).
#
# Options
#   --distro D        foxy | humble | jazzy (default jazzy)
#   --work DIR        default <repo>/sim_work (the model is DIR/runs/bc/policy.onnx)
#   --results DIR     default <repo>/results/sim_eval_v2
#   --python PY       host Python with requirements-tools.txt (default $PYTHON, else python3)
#   --methods LIST    comma-separated arms (default servo,bc)
#   --pair-with DIR   repeatable; see above
#   --rounds N        rounds over the 10 poses (default 3; clean_ref always 1)
#   --schedule-seed N make_schedule.py seed (default 2026)
#   --domain-id N     ROS_DOMAIN_ID inside the container (default 24)
#   --report-only     skip the trials, only analyse
# Environment: HTTPS_PROXY and CA_BUNDLE as in scripts/docker_test.sh.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
REGISTRY=${REGISTRY:-mirror.gcr.io/library}
CA_BUNDLE=${CA_BUNDLE:-}
ALL_CONDITIONS="noise wide hfov62"
EPISODE_TIMEOUT_S=45  # dock_judge ends every episode by 30 s; the margin covers start-up
SIM_HFOV=1.396
ROBOT_HFOV=1.0856

log() { printf '\n[sim_eval %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[sim_eval] error: %s\n' "$*" >&2; exit 1; }

# condition -> fake_robot seed and randomize, start poses file (in the package share), camera
# HFOV, rounds
condition_seed() { case $1 in clean_ref) echo 0 ;; noise) echo 201 ;; wide) echo 202 ;; hfov62) echo 203 ;; *) return 1 ;; esac; }
condition_randomize() { if [ "$1" = clean_ref ]; then echo false; else echo true; fi; }
condition_poses() { case $1 in wide) echo sim_eval_wide_poses.yaml ;; *) echo start_poses.yaml ;; esac; }
condition_hfov() { case $1 in hfov62) echo "$ROBOT_HFOV" ;; *) echo "$SIM_HFOV" ;; esac; }
condition_rounds() { if [ "$1" = clean_ref ]; then echo 1; else echo "$ROUNDS"; fi; }

# 0 when the slot (method, pose_id, round) has a scored row in the CSV
scored() {
  [ -s "$1" ] && awk -F, -v m="$2" -v p="$3" -v r="$4" \
    'NR > 1 && $1 == m && $2 == p && $3 == r && ($4 == "success" || $4 == "fail") { f = 1 } END { exit !f }' "$1"
}

# ============================================================== inside the container

csv_rows() { if [ -s "$1" ]; then echo $(($(wc -l < "$1") - 1)); else echo 0; fi; }

episode() { ros2 run jbl_nodes episode --timeout 10 --expect "$EXPECT" "$@"; }

controller_of() { if [ "$1" = servo ]; then echo servo_baseline; else echo policy_runner; fi; }

# Same as scripts/sim_e2e.sh run_episode: start, wait for the judge's row, abort on no verdict.
run_episode() {
  local method=$1 pose=$2 round=$3 rc=1 before deadline _ controller_node
  controller_node=$(controller_of "$method")
  for _ in 1 2 3; do
    before=$(csv_rows "$CSV")
    rc=0
    episode start "$method" "$pose" "$round" || rc=$?
    [ "$rc" -eq 0 ] && break
    if [ "$rc" -eq 3 ]; then
      if ! grep -qx "/$controller_node" <<< "$(ros2 node list --no-daemon --spin-time 3 2>/dev/null || true)"; then
        EXPECT=$((EXPECT - 1)) episode end fail controller_down || true
        echo "   $(tail -n 1 "$CSV")"
        return 0
      fi
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
  sleep 1
}

wait_for_nodes() {
  local deadline=$((SECONDS + 60)) missing nodes
  while :; do
    missing=""
    nodes=$(ros2 node list --no-daemon --spin-time 3 2>/dev/null || true)
    for n in "$@"; do grep -qx "/$n" <<< "$nodes" || missing="$missing $n"; done
    [ -z "$missing" ] && return 0
    [ "$SECONDS" -ge "$deadline" ] && die "nodes did not come up:$missing (see $LAUNCH_LOG)"
    sleep 1
  done
}

# SCHEDULE CSV BAG LAUNCH_LOG SERVO_LOG SEED RANDOMIZE POSES_FILE HFOV EXTRA_PARAMS WITH_SERVO
# from the environment, plus POLICY_METHOD and MODEL for a policy arm (POLICY_METHOD empty: none).
# shellcheck disable=SC2153
inner_run() {
  : "${SCHEDULE:?}" "${CSV:?}" "${BAG:?}" "${LAUNCH_LOG:?}" "${SERVO_LOG:?}" "${SEED:?}" "${POSES_FILE:?}"
  : "${HFOV:?}" "${EXTRA_PARAMS:?}" "${RANDOMIZE:?}" "${WITH_SERVO:?}"
  # shellcheck source=/dev/null
  . "/opt/ros/$ROS_DISTRO/setup.bash"
  # shellcheck source=/dev/null
  . /ws/install/setup.bash
  local share=/ws/install/jbl_nodes/share/jbl_nodes
  mkdir -p "$(dirname "$CSV")" "$(dirname "$LAUNCH_LOG")" "$(dirname "$BAG")"
  set -m
  local launch_args=(trials_csv:="$CSV" bag:="$BAG" randomize:="$RANDOMIZE" seed:="$SEED"
                     start_poses_file:="$share/config/$POSES_FILE" hfov_rad:="$HFOV"
                     extra_params_file:="$EXTRA_PARAMS")
  local nodes=(fake_robot safety_gate dock_judge trial_logger) servo_pid=
  if [ -n "${POLICY_METHOD:-}" ]; then
    : "${MODEL:?}"
    # policy_runner answers to "start $POLICY_METHOD ..."; servo_baseline (below) to "start servo ...".
    ros2 launch jbl_nodes sim_dock.launch.py method:="$POLICY_METHOD" controller:=policy \
      model_path:="$MODEL" backend:=onnxruntime "${launch_args[@]}" > "$LAUNCH_LOG" 2>&1 &
    nodes+=(policy_runner)
  else
    ros2 launch jbl_nodes sim_dock.launch.py method:=servo controller:=servo \
      "${launch_args[@]}" > "$LAUNCH_LOG" 2>&1 &
    nodes+=(servo_baseline)
  fi
  local launch_pid=$!
  if [ -n "${POLICY_METHOD:-}" ] && [ "$WITH_SERVO" = 1 ]; then
    ros2 run jbl_nodes servo_baseline --ros-args --params-file "$share/config/sim.yaml" \
      --params-file "$EXTRA_PARAMS" -p method_name:=servo -p hfov_rad:="$HFOV" > "$SERVO_LOG" 2>&1 &
    servo_pid=$!
    nodes+=(servo_baseline)
  fi
  # ros2 run only stops when its whole group gets SIGINT
  local stop="kill -INT $launch_pid 2>/dev/null || true; wait $launch_pid || true"
  [ -z "$servo_pid" ] || stop="kill -INT -- -$servo_pid 2>/dev/null || true; $stop; wait $servo_pid || true"
  # shellcheck disable=SC2064  # the pids are fixed now
  trap "$stop" EXIT
  wait_for_nodes "${nodes[@]}"
  sleep 3
  # Subscribers of /episode/event: these nodes but safety_gate, plus the recorder.
  EXPECT=${#nodes[@]}
  local trial round block method pose _event
  # fd 3: the ros2 commands in the loop inherit stdin and must not eat the schedule
  while IFS=, read -r -u 3 trial round block method pose _event; do
    [ "$trial" = trial ] && continue
    if scored "$CSV" "$method" "$pose" "$round"; then continue; fi
    echo "-- trial $trial (round $round, block $block): $method pose $pose"
    run_episode "$method" "$pose" "$round" < /dev/null
  done 3< "$SCHEDULE"
  eval "$stop"
  trap - EXIT
  [ -f "$BAG/metadata.yaml" ] || die "the recorder did not finish $BAG"
}

if [ "${1:-}" = --inner ]; then
  shift
  set +u
  case $1 in
    run) inner_run ;;
    *) die "unknown inner command $1" ;;
  esac
  exit 0
fi

# ============================================================== on the host

DISTRO=jazzy
WORK=$REPO/sim_work
RESULTS=$REPO/results/sim_eval_v2
PY=${PYTHON:-python3}
ROUNDS=3
SCHEDULE_SEED=2026
DOMAIN_ID=24
REPORT_ONLY=0
METHODS=servo,bc
PAIR_WITH=()
CONDITIONS=()
while [ $# -gt 0 ]; do
  case $1 in
    --distro) DISTRO=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --results) RESULTS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --methods) METHODS=$2; shift 2 ;;
    --pair-with) PAIR_WITH+=("$2"); shift 2 ;;
    --rounds) ROUNDS=$2; shift 2 ;;
    --schedule-seed) SCHEDULE_SEED=$2; shift 2 ;;
    --domain-id) DOMAIN_ID=$2; shift 2 ;;
    --report-only) REPORT_ONLY=1; shift ;;
    clean_ref|noise|wide|hfov62) CONDITIONS+=("$1"); shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
# shellcheck disable=SC2206
[ ${#CONDITIONS[@]} -gt 0 ] || CONDITIONS=($ALL_CONDITIONS)
case $DISTRO in foxy|humble|jazzy) ;; *) die "--distro must be foxy, humble or jazzy" ;; esac
WITH_SERVO=0
POLICY_METHOD=
IFS=, read -r -a method_list <<< "$METHODS"
[ ${#method_list[@]} -gt 0 ] || die "--methods needs at least one method"
for m in "${method_list[@]}"; do
  if [ "$m" = servo ]; then
    WITH_SERVO=1
  elif [ -z "$POLICY_METHOD" ]; then
    POLICY_METHOD=$m
  else
    die "--methods: one policy arm per session, got $POLICY_METHOD and $m"
  fi
done
for p in "${PAIR_WITH[@]}"; do [ -d "$p" ] || die "--pair-with $p is not a directory"; done
mkdir -p "$WORK" "$RESULTS"
WORK=$(cd "$WORK" && pwd)
MODEL=$WORK/runs/bc/policy.onnx
[ -z "$POLICY_METHOD" ] || [ -f "$MODEL" ] || die "no $MODEL; train it with scripts/sim_e2e.sh first"

CONTAINER=
cleanup() { [ -n "$CONTAINER" ] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

container() {
  [ -n "$CONTAINER" ] && return 0
  local args=(-d --name "jbl_eval_${DISTRO}_$$" -e "ROS_DOMAIN_ID=$DOMAIN_ID" -e PYTHONUNBUFFERED=1
              -v "$REPO:/repo:ro" -v "$WORK:/work")
  if [ -n "${HTTPS_PROXY:-}" ]; then
    args+=(--network host -e "HTTPS_PROXY=$HTTPS_PROXY" -e "https_proxy=$HTTPS_PROXY")
  fi
  [ -f "$CA_BUNDLE" ] && args+=(-v "$CA_BUNDLE:/usr/local/share/ca-certificates/extra-ca.crt:ro")
  log "starting ros:$DISTRO-ros-base, building the workspace, installing onnxruntime"
  CONTAINER=$(docker run "${args[@]}" "$REGISTRY/ros:$DISTRO-ros-base" sleep infinity)
  docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner setup
  [ -z "$POLICY_METHOD" ] || docker exec "$CONTAINER" bash /repo/scripts/sim_e2e.sh --inner onnxruntime
}

all_scored() {  # schedule csv
  local trial round block method pose _event
  while IFS=, read -r -u 3 trial round block method pose _event; do
    [ "$trial" = trial ] && continue
    scored "$2" "$method" "$pose" "$round" || return 1
  done 3< "$1"
}

# Trials for every condition first, then the reports (a report error must not cost trials).
for cond in "${CONDITIONS[@]}"; do
  dir=$WORK/eval_v2/$DISTRO/$cond
  mkdir -p "$dir/bags" "$dir/logs"
  seed=$(condition_seed "$cond")
  poses=$(condition_poses "$cond")
  hfov=$(condition_hfov "$cond")
  "$PY" "$REPO/tools/make_schedule.py" --methods "$METHODS" --poses 10 --rounds "$(condition_rounds "$cond")" \
    --seed "$SCHEDULE_SEED" --out "$dir/schedule.csv"
  printf 'fake_robot:\n  ros__parameters: {camera_hfov_rad: %s}\n' "$hfov" > "$dir/params.yaml"
  [ "$REPORT_ONLY" -eq 0 ] || continue
  if all_scored "$dir/schedule.csv" "$dir/trials.csv"; then
    log "$cond: every scheduled slot is scored; nothing to run"
    continue
  fi
  container
  seg=seg_$(date -u +%Y%m%dT%H%M%SZ)
  { echo "segment: $seg ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
    echo "dirty: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) paths"
  } >> "$dir/segments.txt"
  log "$cond: $METHODS; fake_robot randomize $(condition_randomize "$cond") seed $seed, poses $poses, camera HFOV $hfov rad -> eval_v2/$DISTRO/$cond ($seg)"
  docker exec -e SCHEDULE="/work/eval_v2/$DISTRO/$cond/schedule.csv" \
    -e CSV="/work/eval_v2/$DISTRO/$cond/trials.csv" -e BAG="/work/eval_v2/$DISTRO/$cond/bags/$seg" \
    -e LAUNCH_LOG="/work/eval_v2/$DISTRO/$cond/logs/$seg.launch.log" \
    -e SERVO_LOG="/work/eval_v2/$DISTRO/$cond/logs/$seg.servo.log" \
    -e SEED="$seed" -e RANDOMIZE="$(condition_randomize "$cond")" -e POSES_FILE="$poses" -e HFOV="$hfov" \
    -e EXTRA_PARAMS="/work/eval_v2/$DISTRO/$cond/params.yaml" -e MODEL=/work/runs/bc/policy.onnx \
    -e POLICY_METHOD="$POLICY_METHOD" -e WITH_SERVO="$WITH_SERVO" \
    "$CONTAINER" bash /repo/scripts/sim_eval.sh --inner run
done
cleanup
CONTAINER=

report() {
  local cond=$1 dir=$WORK/eval_v2/$DISTRO/$1 out=$RESULTS/$1 seed poses hfov p k=0 parts=() sources=()
  seed=$(condition_seed "$cond")
  poses=$(condition_poses "$cond")
  hfov=$(condition_hfov "$cond")
  [ -s "$dir/trials.csv" ] || { log "$cond: no trials yet; no report"; return 0; }
  log "$cond: ground truth, then analyze_trials -> ${out#"$REPO"/}"
  "$PY" "$REPO/scripts/sim_eval_summary.py" episodes --bags "$dir/bags" \
    --poses "$REPO/ros2_ws/src/jbl_nodes/config/$poses" --out "$dir/episodes_gt.csv"
  { echo "condition: $cond"
    echo "arms: $METHODS"
    echo "fake_robot: randomize $(condition_randomize "$cond"), seed $seed, start poses config/$poses, camera_hfov_rad $hfov"
    echo "dock_judge, servo_baseline: hfov_rad $hfov"
    echo "schedule: tools/make_schedule.py --methods $METHODS --poses 10 --rounds $(condition_rounds "$cond") --seed $SCHEDULE_SEED (interleaved)"
    echo "report: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "image: $(docker image inspect --format '{{index .RepoDigests 0}}' "$REGISTRY/ros:$DISTRO-ros-base" 2>/dev/null || echo unknown)"
    if [ -n "$POLICY_METHOD" ]; then
      echo "policy.onnx sha256: $(sha256sum "$MODEL" | cut -d' ' -f1) (${MODEL#"$REPO"/})"
      if [ -f "$WORK/logs/$DISTRO/onnxruntime.txt" ]; then cat "$WORK/logs/$DISTRO/onnxruntime.txt"; fi
    fi
    echo
    cat "$dir/segments.txt" 2>/dev/null || true
  } > "$dir/provenance.txt"
  sources=(--gate "session=$dir/bags")
  [ -z "$POLICY_METHOD" ] || sources+=(--latency "sim-cpu-onnxruntime=$dir/bags")
  for p in "${PAIR_WITH[@]}"; do
    [ -s "$p/$cond/trials.csv" ] || { log "$cond: no $p/$cond/trials.csv to pair with"; return 1; }
    parts+=("$p/$cond")
    k=$((k + 1))
    if [ -d "$p/$cond/bags" ]; then sources+=(--latency "earlier$k=$p/$cond/bags" --gate "earlier$k=$p/$cond/bags"); fi
  done
  # trials.csv, schedule.csv and episodes_gt.csv of the earlier sessions, then this one's
  "$PY" "$REPO/scripts/sim_eval_summary.py" merge --out "$out" "${parts[@]}" "$dir"
  "$PY" "$REPO/tools/analyze_trials.py" "$out/trials.csv" --out-dir "$out" --schedule "$out/schedule.csv" \
    --setting "SIMULATION (fake_robot, ROS 2 $DISTRO), condition $cond" "${sources[@]}"
  { cat "$dir/provenance.txt"
    k=0
    for p in "${parts[@]}"; do
      k=$((k + 1))
      echo
      echo "== earlier$k: rows of ${p#"$REPO"/} (an earlier session; its own noise draws) =="
      cat "$p/provenance.txt" 2>/dev/null || echo "(no provenance.txt)"
    done
  } > "$out/provenance.txt"
}

for cond in "${CONDITIONS[@]}"; do
  report "$cond" || log "$cond: report failed (rerun with --report-only)"
done
"$PY" "$REPO/scripts/sim_eval_summary.py" table --results "$RESULTS" --conditions clean_ref $ALL_CONDITIONS \
  --vs bc --out "$RESULTS/SUMMARY_table.md" || log "summary table failed (rerun with --report-only)"
log "done: ${RESULTS#"$REPO"/}"
