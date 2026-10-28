"""Image preprocessing shared by training (tools/) and on-robot inference (policy_runner).

One function, used on both sides, so the policy sees the same pixels at train and run time.
Python 3.8 compatible; numpy + OpenCV only.

Pipeline (docs/INTERFACES.md, "Policy I/O"):
    BGR or RGB uint8 frame (any size) -> RGB -> resize to dataset size 240x320 (H x W)
    -> resize (squash) to 224x224 -> float32 in [0, 1] -> ImageNet mean/std -> CHW
"""

from typing import Tuple

import cv2
import numpy as np

DATASET_HW = (240, 320)
MODEL_HW = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
V_MAX = 0.25  # m/s for normalised action 1.0
W_MAX = 1.5  # rad/s for normalised action 1.0


def to_dataset_frame(img: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """Any HxWx3 uint8 frame -> 240x320x3 uint8 RGB (what the dataset stores)."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected HxWx3 image, got shape %s" % (img.shape,))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_bgr else img
    h, w = DATASET_HW
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def to_model_input(rgb_dataset_frame: np.ndarray) -> np.ndarray:
    """240x320x3 uint8 RGB -> float32 [3, 224, 224] normalised (no batch dim)."""
    h, w = MODEL_HW
    x = cv2.resize(rgb_dataset_frame, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess(img: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """Camera frame -> float32 [1, 3, 224, 224] ready for the policy."""
    return to_model_input(to_dataset_frame(img, is_bgr))[None]


def action_to_twist(a: np.ndarray) -> Tuple[float, float]:
    """Normalised action [a_v, a_w] -> clipped (v m/s, w rad/s)."""
    a = np.clip(np.asarray(a, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
    return float(a[0] * V_MAX), float(a[1] * W_MAX)


def twist_to_action(v: float, w: float) -> np.ndarray:
    """(v, w) -> normalised action, clipped to [-1, 1]."""
    return np.clip(np.array([v / V_MAX, w / W_MAX], dtype=np.float32), -1.0, 1.0)
