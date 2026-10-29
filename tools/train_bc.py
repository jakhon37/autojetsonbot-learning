#!/usr/bin/env python3
"""Behaviour-cloning baseline: ResNet18 on the front camera -> normalised (v, w) command.

    python tools/train_bc.py --root data/jbl_dock --repo-id jakhon37/jbl_dock --out runs/bc --pretrained \
        --methods teleop --outcomes success --split-by pose_id
    python tools/export_onnx.py runs/bc/model.pt --dataset-root data/jbl_dock

Use the filters: without --methods / --outcomes, failed demos and policy or baseline rollouts
(including any on the held-out test poses) are trained on; without --split-by pose_id,
validation holds repeats of training start spots.

Inputs are prepared exactly as on the robot: the 240x320 RGB dataset frame goes through
jbl_nodes.preprocess.to_model_input (squash to 224x224, ImageNet mean/std). Targets are
preprocess.twist_to_action(v, w), i.e. (v / 0.25 m/s, w / 1.5 rad/s) clipped to [-1, 1].
Augmentation works on the uint8 frame before that step: colour jitter, a small random
translation and, with --hflip, a mirror image whose turn rate is negated.

The split is by episode (jbl_episodes.split_episodes), never by frame. Validation MAE is
reported in m/s and rad/s next to the MAE of always predicting the mean training command, so
a model that learned nothing is easy to spot.

Outputs in --out:
    model.pt          best checkpoint (lowest validation MAE; the last epoch without validation)
    last.pt           final epoch
    policy_meta.json  preprocessing, action scale, dataset identity, split and metrics
    metrics.jsonl     one line per epoch
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torchvision
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "jbl_nodes"))
from jbl_nodes import preprocess  # noqa: E402

import jbl_episodes  # noqa: E402

CHECKPOINT_FORMAT = "jbl-bc-1"
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_SCALE = np.array([preprocess.V_MAX, preprocess.W_MAX], dtype=np.float32)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Model


class BCPolicy(nn.Module):
    """ResNet18 features (+ optional normalised odometry) -> MLP -> normalised (v, w)."""

    def __init__(self, pretrained: bool = False, use_state: bool = False, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # download failure surfaces as URLError, OSError, RuntimeError...
            raise RuntimeError(
                "could not load ImageNet ResNet18 weights (%s). Put resnet18-f37072fd.pth in "
                "$TORCH_HOME/hub/checkpoints (default ~/.cache/torch) or train without --pretrained." % exc
            ) from exc
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_state = use_state
        self.head = nn.Sequential(
            nn.Linear(512 + (2 if use_state else 0), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(image)
        if self.use_state:
            if state is None:
                raise ValueError("this policy was trained with observation.state; pass it")
            features = torch.cat([features, state], dim=1)
        return self.head(features)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[BCPolicy, dict[str, Any]]:
    """(model in eval mode, checkpoint dict) from a train_bc checkpoint; no weight download."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("%s is not a %s checkpoint" % (path, CHECKPOINT_FORMAT))
    model = BCPolicy(pretrained=False, **ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------------------------
# Data


@dataclass
class Augment:
    """Training-time augmentation of a uint8 HxWx3 RGB dataset frame."""

    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.03  # fraction of the hue circle
    max_shift_px: int = 8  # at 320 px width, about 2 degrees of bearing
    hflip: bool = False

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        img = rgb.astype(np.float32)
        img *= rng.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        mean = float((img @ _LUMA).mean())
        img = (img - mean) * rng.uniform(1.0 - self.contrast, 1.0 + self.contrast) + mean
        gray = (img @ _LUMA)[..., None]
        img = gray + (img - gray) * rng.uniform(1.0 - self.saturation, 1.0 + self.saturation)
        out = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
        if self.hue > 0.0:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)  # uint8 hue spans 0..179
            shift = int(round(rng.uniform(-self.hue, self.hue) * 180.0))
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if self.max_shift_px > 0:
            dx, dy = rng.integers(-self.max_shift_px, self.max_shift_px + 1, size=2)
            m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            out = cv2.warpAffine(out, m, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        flipped = bool(self.hflip and rng.random() < 0.5)
        if flipped:
            out = out[:, ::-1]
        return np.ascontiguousarray(out), flipped


def frame_rgb_uint8(img: torch.Tensor) -> np.ndarray:
    """LeRobot CHW frame -> HxWx3 uint8 RGB, whatever the dataset's frame storage.

    Video frames arrive as uint8 (return_uint8=True); image-stored frames (bag_to_lerobot
    --video off) arrive as float32 in [0, 1] whatever return_uint8 says. Both must reach
    preprocess.to_model_input as the uint8 frame the robot sees.
    """
    if img.dtype != torch.uint8:
        img = (img.float() * 255.0).round().clamp(0, 255).to(torch.uint8)
    return img.permute(1, 2, 0).numpy()


class BCFrames(torch.utils.data.Dataset):
    """LeRobotDataset (return_uint8=True) -> (model input, normalised state, normalised action)."""

    def __init__(self, frames: Any, augment: Augment | None = None, seed: int = 0) -> None:
        self.frames = frames
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def _generator(self) -> np.random.Generator:
        # Worker seeds change every epoch (DataLoader draws a new base seed), so augmentations
        # do too; with num_workers=0 one generator runs through all epochs.
        info = torch.utils.data.get_worker_info()
        seed = self.seed if info is None else info.seed
        if self._rng is None or self._rng_seed != seed:
            self._rng, self._rng_seed = np.random.default_rng(seed), seed
        return self._rng

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.frames[index]
        rgb = frame_rgb_uint8(item[IMAGE_KEY])
        v, w = (float(x) for x in item[ACTION_KEY])
        sv, sw = (float(x) for x in item[STATE_KEY])
        if self.augment is not None:
            rgb, flipped = self.augment(rgb, self._generator())
            if flipped:
                w, sw = -w, -sw
        x = preprocess.to_model_input(rgb)
        return (
            torch.from_numpy(x),
            torch.from_numpy(preprocess.twist_to_action(sv, sw)),
            torch.from_numpy(preprocess.twist_to_action(v, w)),
        )


def open_frames(repo_id: str, root: Path, episodes: Sequence[int], revision: str | None = None) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, episodes=list(episodes), revision=revision, return_uint8=True)


def episode_actions(root: Path, episodes: Sequence[int]) -> np.ndarray:
    """Normalised actions of the given episodes, read from parquet without decoding video."""
    import pandas as pd

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", ACTION_KEY]) for f in files])
    raw = np.stack(df[df["episode_index"].isin(list(episodes))][ACTION_KEY].to_numpy())
    return np.clip(raw.astype(np.float32) / ACTION_SCALE, -1.0, 1.0)


# --------------------------------------------------------------------------------------------
# Provenance


def git_info(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(["git", "-C", str(path)] + list(args), capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def dataset_fingerprint(root: Path) -> str:
    """sha256 over meta/ and data/ (parquet + json; videos are left out for speed)."""
    digest = hashlib.sha256()
    for sub in ("meta", "data"):
        for path in sorted((root / sub).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _stats_summary(stats: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = stats.get(key)
    if not entry:
        return None
    keys = [k for k in ("mean", "std", "min", "max") if k in entry]
    return {k: np.asarray(entry[k], dtype=np.float64).round(6).tolist() for k in keys}


def build_policy_meta(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    train_eps: Sequence[int],
    val_eps: Sequence[int],
    augment: Augment | None,
) -> dict[str, Any]:
    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    import lerobot

    return {
        "policy": "bc_resnet18",
        "input_name": "image",
        "output_name": "action",
        "input_shape": [1, 3] + list(preprocess.MODEL_HW),
        "output_shape": [1, 2],
        "image": {
            "color": "RGB",
            "dataset_hw": list(preprocess.DATASET_HW),
            "resize": "squash 240x320 -> 224x224 with cv2.INTER_AREA, no crop (jbl_nodes.preprocess)",
            "scale": "1/255",
        },
        "normalization": {
            "type": "imagenet",
            "mean": [round(float(x), 6) for x in preprocess.IMAGENET_MEAN],
            "std": [round(float(x), 6) for x in preprocess.IMAGENET_STD],
        },
        "action": {
            "names": ["v", "w"],
            "units": ["m/s", "rad/s"],
            "normalised": True,
            "to_twist": "clip to [-1, 1], then v = a0 * v_max, w = a1 * w_max",
        },
        "action_scale": [preprocess.V_MAX, preprocess.W_MAX],
        "v_max": preprocess.V_MAX,
        "w_max": preprocess.W_MAX,
        "uses_state": bool(model_config["use_state"]),
        "dataset": {
            "repo_id": args.repo_id,
            "root": str(root.resolve()),
            "revision": args.revision,
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fingerprint_sha256": dataset_fingerprint(root),
            "stats": {k: _stats_summary(stats, k) for k in (ACTION_KEY, STATE_KEY)},
        },
        "train_episodes": list(train_eps),
        "val_episodes": list(val_eps),
        "split": {
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "group_by": args.split_by,
            "outcomes": args.outcomes,
            "methods": args.methods,
        },
        "training": {
            "model": model_config,
            "pretrained": args.pretrained,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "huber_delta": args.huber_delta,
            "schedule": "cosine per step to lr * 0.01",
            "augment": asdict(augment) if augment is not None else None,
        },
        "git": git_info(Path(__file__).resolve().parent),
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "lerobot": str(lerobot.__version__)},
        "created_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------------------------
# Training


@torch.no_grad()
def evaluate(model: BCPolicy, loader: Any, device: torch.device, loss_fn: nn.Module,
             const_action: np.ndarray) -> dict[str, float]:
    """Validation loss and MAE of the clipped prediction, in normalised and physical units."""
    model.eval()
    loss_sum, n = 0.0, 0
    abs_err = torch.zeros(2, dtype=torch.float64)
    const_err = torch.zeros(2, dtype=torch.float64)
    const = torch.from_numpy(const_action).to(torch.float64)
    for x, state, target in loader:
        x, state, target = x.to(device), state.to(device), target.to(device)
        pred = model(x, state)
        loss_sum += float(loss_fn(pred, target)) * len(x)
        n += len(x)
        abs_err += (pred.clamp(-1.0, 1.0) - target).abs().sum(0).double().cpu()
        const_err += (const - target.double().cpu()).abs().sum(0)
    mae, mae_const = (abs_err / n).numpy(), (const_err / n).numpy()
    return {
        "val_loss": loss_sum / n,
        "val_mae_norm": float(mae.mean()),
        "val_mae_v": float(mae[0] * preprocess.V_MAX),
        "val_mae_w": float(mae[1] * preprocess.W_MAX),
        "val_mae_v_const": float(mae_const[0] * preprocess.V_MAX),
        "val_mae_w_const": float(mae_const[1] * preprocess.W_MAX),
        "val_frames": n,
    }


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _csv(text: str | None) -> list[str] | None:
    return None if text is None else [x.strip() for x in text.split(",") if x.strip()]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("--root", required=True, type=Path, help="local LeRobot dataset (bag_to_lerobot.py output)")
    parser.add_argument("--repo-id", default=None, help="dataset id (default: local/<root name>)")
    parser.add_argument("--revision", default=None, help="dataset revision, recorded in policy_meta.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="allow an --out that already has model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.1, help="in normalised action units")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet ResNet18 weights (downloaded once)")
    parser.add_argument("--use-state", action="store_true",
                        help="also feed normalised odometry (v, w); not exportable to the robot's image-only "
                             "ONNX contract, and BC on own velocity tends to copy it instead of steering")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--hflip", action="store_true",
                        help="mirror half the frames and negate w (off by default: a mirrored ArUco marker "
                             "is not the marker the robot will see)")
    parser.add_argument("--max-shift-px", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-by", default=None, help="keep groups on one side, e.g. pose_id")
    parser.add_argument("--outcomes", default=None, help="train on these outcomes only, e.g. success or success,none")
    parser.add_argument("--methods", default=None,
                        help="train on these episode methods only, e.g. teleop,none (keeps policy rollouts "
                             "from trial bags out of the demonstrations; none = bags without events)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps")
    args = parser.parse_args(argv)
    args.repo_id = args.repo_id or "local/%s" % args.root.resolve().name
    args.outcomes = _csv(args.outcomes)
    args.methods = _csv(args.methods)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "model.pt").exists() and not args.overwrite:
        raise SystemExit("%s already holds a model.pt; pass --overwrite or choose another --out" % args.out)

    info = json.loads((args.root / "meta" / "info.json").read_text())
    records = jbl_episodes.load_episode_table(args.root, total_episodes=info["total_episodes"])
    records = jbl_episodes.filter_episodes(records, outcomes=args.outcomes, methods=args.methods)
    methods = sorted({str(r.get("method")) for r in records})
    if args.methods is None and len(methods) > 1:
        print("train_bc: warning: no --methods filter; training on every method in the dataset (%s), "
              "including any policy or baseline rollouts" % ", ".join(methods), file=sys.stderr)
    if args.split_by is None and any(r.get("pose_id") is not None for r in records):
        print("train_bc: warning: no --split-by; validation may hold repeats of training start spots "
              "(use --split-by pose_id)", file=sys.stderr)
    train_eps, val_eps = jbl_episodes.split_episodes(records, args.val_fraction, args.seed, args.split_by)
    if not train_eps:
        raise SystemExit("no training episodes (check --outcomes / --methods)")
    (args.out / "split.json").write_text(json.dumps({"train": train_eps, "val": val_eps}) + "\n")

    augment = None if args.no_augment else Augment(max_shift_px=args.max_shift_px, hflip=args.hflip)
    train_set = BCFrames(open_frames(args.repo_id, args.root, train_eps, args.revision), augment, args.seed)
    if len(train_set) < 2:
        raise SystemExit("need at least 2 training frames, got %d" % len(train_set))
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0)
    # A last batch of one frame would break BatchNorm in training mode.
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=len(train_set) % args.batch_size == 1,
                                               **loader_kwargs)
    val_loader = None
    if val_eps:
        val_set = BCFrames(open_frames(args.repo_id, args.root, val_eps, args.revision))
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kwargs)
    const_action = episode_actions(args.root, train_eps).mean(axis=0)

    model_config = {"use_state": args.use_state, "hidden": args.hidden, "dropout": args.dropout}
    model = BCPolicy(pretrained=args.pretrained, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)
    meta = build_policy_meta(args, model_config, train_eps, val_eps, augment)
    print("train: %d episodes / %d frames, val: %d episodes / %d frames, device %s" % (
        len(train_eps), len(train_set), len(val_eps), len(val_loader.dataset) if val_loader else 0, device))

    best_score, best_metrics = math.inf, None
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.write_text("")
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        model.train()
        loss_sum, n = 0.0, 0
        for x, state, target in train_loader:
            x, state, target = x.to(device), state.to(device), target.to(device)
            loss = loss_fn(model(x, state), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
        metrics: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "lr": scheduler.get_last_lr()[0]}
        if val_loader is not None:
            metrics.update(evaluate(model, val_loader, device, loss_fn, const_action))
        metrics["seconds"] = round(time.monotonic() - t0, 1)
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        line = "epoch %3d/%d  train_loss %.4f" % (epoch, args.epochs, metrics["train_loss"])
        if val_loader is not None:
            line += "  val_loss %.4f  val MAE v %.3f m/s  w %.3f rad/s  (mean-command baseline v %.3f  w %.3f)" % (
                metrics["val_loss"], metrics["val_mae_v"], metrics["val_mae_w"],
                metrics["val_mae_v_const"], metrics["val_mae_w_const"])
        print(line + "  %.1f s" % metrics["seconds"], flush=True)

        checkpoint = {"format": CHECKPOINT_FORMAT, "model_config": model_config, "epoch": epoch, "metrics": metrics,
                      "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        score = metrics.get("val_mae_norm", -epoch)  # without validation the newest epoch wins
        if score < best_score:
            best_score, best_metrics = score, metrics
            meta["metrics"] = dict(metrics, best_epoch=epoch)
            torch.save(dict(checkpoint, policy_meta=meta), args.out / "model.pt")
        torch.save(dict(checkpoint, policy_meta=dict(meta, metrics=metrics)), args.out / "last.pt")

    meta_path = args.out / "policy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("best epoch %d -> %s (+ %s)" % (best_metrics["epoch"], args.out / "model.pt", meta_path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
