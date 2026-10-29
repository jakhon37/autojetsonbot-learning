#!/usr/bin/env bash
# Train ACT (LeRobot 0.6.1, lerobot-train) on the docking dataset written by bag_to_lerobot.py.
#
#   DATASET_ROOT=data/jbl_dock REPO_ID=jakhon37/jbl_dock OUT=runs/act bash tools/train_act.sh
#   DATASET_ROOT=... bash tools/train_act.sh --policy.kl_weight=5      # extra args go to lerobot-train
#
# By default it trains on the same episodes as train_bc's recommended run (README): successful
# teleop demos, split by pose_id with validation fraction 0.2 and seed 0 (the validation part is
# left out). Failed demos and policy or baseline rollouts (trial bags, possibly on the held-out
# test poses) are never trained on unless asked for. To use a BC run's exact split:
#   EPISODES=$(python -c 'import json; print(json.load(open("runs/bc/split.json"))["train"])') \
#   DATASET_ROOT=... bash tools/train_act.sh
#
# Settings (environment variables, defaults in brackets):
#   DATASET_ROOT    local LeRobot dataset (required)
#   REPO_ID         dataset id [local/<basename of DATASET_ROOT>]
#   OUT             output directory, must not exist [runs/act]
#   DEVICE          cuda | cpu | mps [cuda when nvidia-smi sees a GPU, else cpu]
#   STEPS [20000]  BATCH [8]  WORKERS [4]  SEED [1000]  SAVE_FREQ [STEPS/4]  LOG_FREQ [100]
#   CHUNK           actions predicted per inference [10 = 1 s at 10 fps]
#   N_ACTION_STEPS  actions executed per inference at rollout [1]: policy_runner asks for one
#                   command per 0.1 s tick, i.e. receding horizon on the first action of each chunk
#   EPISODES        JSON list of training episode indices, or "all" for every episode in the
#                   dataset [the train part of the split below]
#   METHODS [teleop]  OUTCOMES [success]  VAL_FRACTION [0.2]  SPLIT_SEED [0]
#                   the default episode selection: jbl_episodes.py --methods --outcomes --split
#                   --split-by pose_id (comma lists; "none" matches episodes without a value)
#   PYTHON          interpreter for jbl_episodes.py [python3]
#   BACKBONE        imagenet | none [imagenet]: ImageNet ResNet18 weights are downloaded once from
#                   download.pytorch.org into $TORCH_HOME; "none" trains from scratch, offline
#   AMP             1 to use mixed precision on CUDA [0]
#   LEROBOT_TRAIN   lerobot-train executable [lerobot-train on PATH]
#
# Hardware: CUDA is the intended target; budget roughly an hour for 20k steps at batch 8 on a
# recent consumer GPU. CPU works but is only practical for smoke tests (STEPS=2) and tiny runs.
#
# Deployment caveat: ACT on this dataset also takes observation.state (odometry v, w) as input
# and predicts CHUNK x 2 actions in dataset units (m/s, rad/s, de-normalised by LeRobot's
# processors), whereas policy_runner's ONNX contract is image -> [1, 2] normalised action
# (docs/INTERFACES.md). Running ACT on the robot therefore needs a wrapper that feeds state and
# returns the first action divided by (0.25, 1.5); tools/export_onnx.py exports BC checkpoints only.
set -euo pipefail

: "${DATASET_ROOT:?set DATASET_ROOT to the dataset directory written by bag_to_lerobot.py}"
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "train_act.sh: $DATASET_ROOT is not a LeRobot dataset (no meta/info.json)" >&2
  exit 2
fi
REPO_ID=${REPO_ID:-local/$(basename "$DATASET_ROOT")}
OUT=${OUT:-runs/act}
if [ -z "${DEVICE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then DEVICE=cuda; else DEVICE=cpu; fi
fi
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-4}
SEED=${SEED:-1000}
SAVE_FREQ=${SAVE_FREQ:-$(( STEPS >= 4 ? STEPS / 4 : 1 ))}
LOG_FREQ=${LOG_FREQ:-100}
CHUNK=${CHUNK:-10}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
BACKBONE=${BACKBONE:-imagenet}
AMP=${AMP:-0}
LEROBOT_TRAIN=${LEROBOT_TRAIN:-lerobot-train}

extra=()
case "$BACKBONE" in
  imagenet) ;;  # lerobot's ACT default: ResNet18_Weights.IMAGENET1K_V1
  none) extra+=(--policy.pretrained_backbone_weights=null) ;;
  *) echo "train_act.sh: BACKBONE must be imagenet or none, got '$BACKBONE'" >&2; exit 2 ;;
esac
if [ -z "${EPISODES:-}" ]; then
  if ! EPISODES=$("${PYTHON:-python3}" "$(dirname "$0")/jbl_episodes.py" --root "$DATASET_ROOT" \
      --methods "${METHODS:-teleop}" --outcomes "${OUTCOMES:-success}" --split --split-by pose_id \
      --val-fraction "${VAL_FRACTION:-0.2}" --seed "${SPLIT_SEED:-0}" \
      | "${PYTHON:-python3}" -c 'import json, sys; t = json.load(sys.stdin)["train"]; print(t) if t else sys.exit(1)'); then
    echo "train_act.sh: no training episodes match METHODS=${METHODS:-teleop} OUTCOMES=${OUTCOMES:-success}" \
      "in $DATASET_ROOT (set EPISODES=all to train on every episode)" >&2
    exit 2
  fi
  echo "train_act.sh: training episodes $EPISODES (methods ${METHODS:-teleop}, outcomes ${OUTCOMES:-success})" >&2
fi
[ "$EPISODES" != all ] && extra+=("--dataset.episodes=$EPISODES")
[ "$AMP" = 1 ] && extra+=(--policy.use_amp=true)

set -x
exec "$LEROBOT_TRAIN" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.chunk_size="$CHUNK" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=act_jbl_dock \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --num_workers="$WORKERS" \
  --seed="$SEED" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --wandb.enable=false \
  "${extra[@]}" \
  "$@"
