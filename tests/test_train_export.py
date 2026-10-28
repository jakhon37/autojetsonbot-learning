"""train_bc.py, export_onnx.py, train_act.sh and make_marker.py on the synthetic dataset (CPU).

The exported policy is also loaded through the robot's own loader (jbl_nodes.policy_backends),
so a mismatch between what the tools write and what policy_runner accepts fails here.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import export_onnx
import jbl_episodes
import make_marker
import train_bc
from conftest import REPO_ID, TOOLS
from jbl_nodes import policy_backends, preprocess

IMAGE_KEY = train_bc.IMAGE_KEY


def _dataset_frame(root, index=0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root, return_uint8=True)
    return ds[index]


# --------------------------------------------------------------------------------------------
# Training


def test_bc_run_outputs(bc_run, dataset_root):
    names = {p.name for p in bc_run.iterdir()}
    assert {"model.pt", "last.pt", "policy_meta.json", "metrics.jsonl", "split.json"} <= names
    epochs = [json.loads(line) for line in (bc_run / "metrics.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epochs] == [1, 2]
    for e in epochs:
        assert np.isfinite(e["train_loss"]) and np.isfinite(e["val_loss"])
        for key in ("val_mae_v", "val_mae_w", "val_mae_v_const", "val_mae_w_const"):
            assert 0.0 <= e[key] < 3.0, key

    meta = json.loads((bc_run / "policy_meta.json").read_text())
    split = json.loads((bc_run / "split.json").read_text())
    all_eps = [r["episode_index"] for r in jbl_episodes.read_episode_meta(dataset_root)]
    assert meta["train_episodes"] == split["train"] and meta["val_episodes"] == split["val"]
    assert split["val"] and not set(split["train"]) & set(split["val"])
    assert sorted(split["train"] + split["val"]) == all_eps
    assert meta["input_shape"] == [1, 3, 224, 224] and meta["action_scale"] == [0.25, 1.5]
    assert (meta["v_max"], meta["w_max"]) == (preprocess.V_MAX, preprocess.W_MAX)
    assert meta["normalization"]["type"] == "imagenet"
    assert meta["dataset"]["repo_id"] == REPO_ID and meta["dataset"]["codebase_version"] == "v3.0"
    assert re.fullmatch(r"[0-9a-f]{64}", meta["dataset"]["fingerprint_sha256"])
    assert meta["dataset"]["stats"]["action"]["mean"] is not None
    assert set(meta["git"]) == {"commit", "dirty"}
    assert meta["uses_state"] is False and meta["metrics"]["best_epoch"] in (1, 2)
    best = min(epochs, key=lambda e: e["val_mae_norm"])
    assert meta["metrics"]["epoch"] == best["epoch"]

    model, ckpt = train_bc.load_checkpoint(bc_run / "model.pt")  # weights_only load, no download
    assert not model.training and ckpt["epoch"] == best["epoch"]
    assert ckpt["policy_meta"] == meta


def test_policy_meta_is_accepted_by_policy_runner(bc_run):
    meta = json.loads((bc_run / "policy_meta.json").read_text())
    checked = policy_backends.check_policy_meta(meta)
    assert {"input_shape", "normalization.mean", "normalization.std", "action_scale"} <= set(checked)


def test_training_inputs_match_robot_preprocessing(dataset_root):
    item = _dataset_frame(dataset_root, 5)
    frames = train_bc.BCFrames([item], augment=None)
    x, state, action = frames[0]
    rgb = item[IMAGE_KEY].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(x.numpy(), preprocess.to_model_input(rgb))
    np.testing.assert_array_equal(action.numpy(), preprocess.twist_to_action(*item["action"].tolist()))
    np.testing.assert_array_equal(state.numpy(), preprocess.twist_to_action(*item["observation.state"].tolist()))


def test_hflip_negates_turn_rate():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :100] = (200, 40, 40)  # asymmetric: red on the left
    item = {
        IMAGE_KEY: torch.from_numpy(rgb).permute(2, 0, 1),
        "action": torch.tensor([0.1, 0.6]),
        "observation.state": torch.tensor([0.05, -0.3]),
    }
    flip_only = train_bc.Augment(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, max_shift_px=0, hflip=True)
    frames = train_bc.BCFrames([item], augment=flip_only, seed=1)
    plain = preprocess.to_model_input(rgb)
    mirrored = preprocess.to_model_input(np.ascontiguousarray(rgb[:, ::-1]))
    seen = set()
    for _ in range(40):
        x, state, action = frames[0]
        flipped = bool(action[1] < 0)
        seen.add(flipped)
        np.testing.assert_allclose(action.numpy(), preprocess.twist_to_action(0.1, -0.6 if flipped else 0.6))
        np.testing.assert_allclose(state.numpy(), preprocess.twist_to_action(0.05, 0.3 if flipped else -0.3))
        np.testing.assert_array_equal(x.numpy(), mirrored if flipped else plain)
    assert seen == {True, False}


def test_augment_changes_pixels_not_geometry_much():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    out, flipped = train_bc.Augment()(rgb, np.random.default_rng(1))
    assert out.shape == rgb.shape and out.dtype == np.uint8 and not flipped
    assert not np.array_equal(out, rgb)
    shift_only = train_bc.Augment(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, max_shift_px=8)
    shifted, _ = shift_only(rgb, np.random.default_rng(2))
    best = min(
        (np.abs(shifted[20:-20, 20:-20].astype(int) - np.roll(rgb, (dy, dx), (0, 1))[20:-20, 20:-20]).mean(), dx, dy)
        for dx in range(-8, 9) for dy in range(-8, 9)
    )
    assert best[0] == 0.0  # a pure integer translation within +-8 px


def test_state_policy_trains_without_validation_and_is_not_exportable(dataset_root, tmp_path):
    out = tmp_path / "bc_state"
    rc = train_bc.main(["--root", str(dataset_root), "--repo-id", REPO_ID, "--out", str(out), "--epochs", "1",
                        "--batch-size", "16", "--num-workers", "0", "--device", "cpu", "--use-state",
                        "--val-fraction", "0", "--no-augment", "--outcomes", "success,none",
                        "--methods", "teleop,none"])
    assert rc == 0
    meta = json.loads((out / "policy_meta.json").read_text())
    assert meta["uses_state"] is True and meta["val_episodes"] == []
    teleop_successes = [r["episode_index"] for r in jbl_episodes.read_episode_meta(dataset_root)
                        if r["outcome"] in ("success", None) and r["method"] in ("teleop", None)]
    assert meta["train_episodes"] == teleop_successes and len(teleop_successes) == 4
    with pytest.raises(SystemExit, match="observation.state"):
        export_onnx.main([str(out / "model.pt"), "--out", str(tmp_path / "x.onnx")])
    assert not (tmp_path / "x.onnx").exists()
    with pytest.raises(SystemExit, match="model.pt"):
        train_bc.main(["--root", str(dataset_root), "--out", str(out), "--epochs", "1"])


# --------------------------------------------------------------------------------------------
# Export


@pytest.fixture(scope="module")
def exported(bc_run, dataset_root, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("deploy")
    rc = export_onnx.main([str(bc_run / "model.pt"), "--out", str(out_dir / "policy.onnx"),
                           "--dataset-root", str(dataset_root), "--repo-id", REPO_ID,
                           "--torchscript", str(out_dir / "policy.ts")])
    assert rc == 0
    return out_dir


def test_onnx_contract_and_parity(exported, bc_run):
    import onnx

    model = onnx.load(str(exported / "policy.onnx"))
    onnx.checker.check_model(model, full_check=True)
    meta = json.loads((exported / "policy_meta.json").read_text())
    info = meta.pop("export")
    assert meta == json.loads((bc_run / "policy_meta.json").read_text())
    assert info["exporter"] == "torchscript" and info["opset"] == 13 and info["tensorrt8_compatible_opset"]
    assert info["parity"]["max_abs_diff"] < 1e-4
    assert info["parity"]["dataset_frames"] > 0 and info["parity"]["random_inputs"] == 8
    assert info["torchscript"]["max_abs_diff"] < 1e-4
    assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"])


def test_robot_backends_reproduce_torch(exported, bc_run, dataset_root):
    """policy_runner's onnxruntime and torch backends give the training model's output."""
    model, _ = train_bc.load_checkpoint(bc_run / "model.pt")
    rgb = _dataset_frame(dataset_root, 10)[IMAGE_KEY].permute(1, 2, 0).numpy()
    x = preprocess.to_model_input(rgb)[None]
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy()
    for name, path in (("onnxruntime", exported / "policy.onnx"), ("torch", exported / "policy.ts")):
        backend = policy_backends.create_backend(name, str(path))
        try:
            assert backend.meta is not None and "action_scale" in backend.meta_checked
            np.testing.assert_allclose(backend.infer(x), reference, atol=1e-4)
        finally:
            backend.close()


def test_dynamo_exporter_fallback_path(bc_run, tmp_path):
    """The torch.export-based exporter also meets the contract (opset 18: onnxruntime, not TRT 8.0-8.2)."""
    out = tmp_path / "policy.onnx"
    assert export_onnx.main([str(bc_run / "model.pt"), "--out", str(out), "--exporter", "dynamo"]) == 0
    info = json.loads((tmp_path / "policy_meta.json").read_text())["export"]
    assert info["exporter"] == "dynamo" and info["opset"] == 18 and not info["tensorrt8_compatible_opset"]
    assert info["parity"]["max_abs_diff"] < 1e-4
    backend = policy_backends.create_backend("onnxruntime", str(out))
    try:
        assert backend.infer(np.zeros((1, 3, 224, 224), dtype=np.float32)).shape == (1, 2)
    finally:
        backend.close()


def test_failed_parity_writes_nothing(bc_run, tmp_path):
    with pytest.raises(SystemExit, match="parity check failed"):
        export_onnx.main([str(bc_run / "model.pt"), "--out", str(tmp_path / "p.onnx"), "--tolerance", "-1"])
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------------------
# ACT and the marker


def test_train_act_two_steps_on_cpu(dataset_root, tmp_path):
    out = tmp_path / "act"
    env = dict(os.environ, DATASET_ROOT=str(dataset_root), REPO_ID=REPO_ID, OUT=str(out), DEVICE="cpu",
               STEPS="2", BATCH="4", WORKERS="0", LOG_FREQ="1", BACKBONE="none", EPISODES="[0,2]",
               LEROBOT_TRAIN=str(Path(sys.executable).with_name("lerobot-train")))
    proc = subprocess.run(["bash", str(TOOLS / "train_act.sh")], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    model_dir = out / "checkpoints" / "last" / "pretrained_model"
    config = json.loads((model_dir / "config.json").read_text())
    assert (config["type"], config["chunk_size"], config["n_action_steps"]) == ("act", 10, 1)
    assert config["pretrained_backbone_weights"] is None
    assert set(config["input_features"]) == {IMAGE_KEY, "observation.state"}
    train_config = json.loads((model_dir / "train_config.json").read_text())
    assert train_config["dataset"]["episodes"] == [0, 2] and train_config["steps"] == 2
    assert (model_dir / "model.safetensors").is_file()


def test_make_marker(tmp_path):
    from PIL import Image

    assert make_marker.main(["--out-dir", str(tmp_path)]) == 0
    png, pdf = tmp_path / "aruco_4x4_50_id7_100mm.png", tmp_path / "aruco_4x4_50_id7_100mm.pdf"
    with Image.open(str(png)) as im:
        assert im.info["dpi"] == pytest.approx((300, 300), abs=0.01)
        gray = np.asarray(im.convert("L"))
    px_per_mm = 300 / 25.4
    w_mm, h_mm = make_marker.page_size_mm(100.0, 10.0, 300)
    assert gray.shape == (round(h_mm * px_per_mm), round(w_mm * px_per_mm))
    # Black extent of the marker: 100 mm to within one pixel, crisp (pure black and white).
    top = int(round(make_marker.MARGIN_MM * px_per_mm)) - 2
    bottom = int(round((make_marker.MARGIN_MM + 120.0) * px_per_mm)) + 2
    ys, xs = np.nonzero(gray[top:bottom] < 128)
    assert abs((xs.max() - xs.min() + 1) / px_per_mm - 100.0) <= 1.0 / px_per_mm
    assert abs((ys.max() - ys.min() + 1) / px_per_mm - 100.0) <= 1.0 / px_per_mm
    marker = gray[top + ys.min(): top + ys.max() + 1, xs.min(): xs.max() + 1]
    assert set(np.unique(marker)) == {0, 255}
    # 10 mm quiet zone: white all around the black square (the dashed cut line is at 10 mm).
    q = int(10 * px_per_mm) - 4
    ring = gray[top + ys.min() - q: top + ys.max() + 1 + q, xs.min() - q: xs.max() + 1 + q].copy()
    ring[q:-q, q:-q] = 255
    assert ring.min() == 255
    # The cell pattern is DICT_4X4_50 id 7.
    with pytest.raises(ValueError, match="ids 0..49"):
        make_marker.marker_cells(50)
    cells = make_marker.marker_cells(7)
    n = cells.shape[0]
    centres = ((np.arange(n) + 0.5) * marker.shape[0] / n).astype(int)
    np.testing.assert_array_equal(marker[np.ix_(centres, centres)] < 128, cells)
    # PDF page in points matches the PNG page; vector marker is 100 mm by construction.
    media = re.search(rb"/MediaBox \[ *0 0 ([0-9.]+) ([0-9.]+) *\]", pdf.read_bytes())
    assert media and float(media.group(1)) == pytest.approx(w_mm / 25.4 * 72, abs=0.01)
    assert float(media.group(2)) == pytest.approx(h_mm / 25.4 * 72, abs=0.01)
    fig = make_marker.build_figure(7, "DICT_4X4_50", 100.0, 10.0, 72)
    corners = fig.axes[0].transData.transform([(15.0, 15.0), (115.0, 115.0)])
    assert np.allclose(np.abs(corners[1] - corners[0]), 100.0 / 25.4 * 72)
