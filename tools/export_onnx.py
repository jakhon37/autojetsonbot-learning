#!/usr/bin/env python3
"""train_bc checkpoint -> ONNX policy for policy_runner (and TensorRT on the Jetson).

    python tools/export_onnx.py runs/bc/model.pt                              # -> runs/bc/policy.onnx
    python tools/export_onnx.py runs/bc/model.pt --out deploy/bc/policy.onnx \
        --dataset-root data/jbl_dock --repo-id jakhon37/jbl_dock --torchscript deploy/bc/policy.ts

The graph has one input "image" float32 [1,3,224,224] and one output "action" float32 [1,2]
(normalised (v, w), docs/INTERFACES.md "Policy I/O"), with static shapes. The TorchScript-based
exporter is tried first because it writes opset 13 / IR 7, which TensorRT 8.0-8.2 (JetPack 4.6)
can parse; the torch.export-based exporter is the fallback and produces opset 18, which needs
TensorRT >= 8.6 (fine for onnxruntime).

After export: onnx.checker (full check), then ONNX Runtime against PyTorch on random inputs and,
with --dataset-root, on dataset frames preprocessed exactly as on the robot. Any output differing
by more than --tolerance fails the export and no file is left at --out. policy_meta.json is
written beside the model: the checkpoint's metadata (for model.pt, the training run's
policy_meta.json) plus an "export" section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "jbl_nodes"))
from jbl_nodes import preprocess  # noqa: E402

from train_bc import IMAGE_KEY, frame_rgb_uint8, load_checkpoint  # noqa: E402

INPUT_NAME = "image"
OUTPUT_NAME = "action"
INPUT_SHAPE = (1, 3) + tuple(preprocess.MODEL_HW)
OUTPUT_SHAPE = (1, 2)
META_FILENAME = "policy_meta.json"
TRT8_MAX_OPSET = 13


class ImagePolicy(nn.Module):
    """The robot's contract: image in, normalised action out."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.policy(image)


def export_onnx(module: nn.Module, path: Path, opset: int = TRT8_MAX_OPSET, exporter: str = "auto") -> str:
    """Write ``module`` to ``path``; returns the exporter that succeeded."""
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    common = dict(input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], do_constant_folding=True)
    errors = []
    if exporter in ("auto", "torchscript"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)  # "legacy exporter": chosen on purpose
                torch.onnx.export(module, (example,), str(path), opset_version=opset, dynamo=False, **common)
            return "torchscript"
        except Exception as exc:  # the fallback below is the remedy for any exporter failure
            if exporter == "torchscript":
                raise
            errors.append("torchscript exporter: %s" % exc)
    dynamo_opset = max(opset, 18)  # its lowest; version conversion below 18 fails for ResNet ops
    if dynamo_opset > TRT8_MAX_OPSET:
        print("warning: %sexporting with opset %d; TensorRT 8.0-8.2 (JetPack 4.6) parses opset <= %d"
              % ("%s; " % errors[0] if errors else "", dynamo_opset, TRT8_MAX_OPSET), file=sys.stderr)
    torch.onnx.export(module, (example,), str(path), opset_version=dynamo_opset, dynamo=True,
                      external_data=False, **common)
    return "dynamo"


def check_graph(path: Path) -> Any:
    """onnx.checker plus the I/O contract; returns the loaded model."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)

    def dims(value: Any) -> tuple[Any, ...]:
        return tuple(d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim)

    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if [i.name for i in inputs] != [INPUT_NAME] or dims(inputs[0]) != INPUT_SHAPE:
        raise ValueError("ONNX inputs %s, expected %s %s"
                         % ([(i.name, dims(i)) for i in inputs], INPUT_NAME, INPUT_SHAPE))
    if [o.name for o in graph.output] != [OUTPUT_NAME] or dims(graph.output[0]) != OUTPUT_SHAPE:
        raise ValueError("ONNX outputs %s, expected %s %s" % (
            [(o.name, dims(o)) for o in graph.output], OUTPUT_NAME, OUTPUT_SHAPE))
    float32 = onnx.TensorProto.FLOAT
    if inputs[0].type.tensor_type.elem_type != float32 or graph.output[0].type.tensor_type.elem_type != float32:
        raise ValueError("ONNX input and output must be float32")
    return model


def dataset_inputs(root: Path, repo_id: str | None, count: int) -> list[np.ndarray]:
    """``count`` frames spread over the dataset, preprocessed exactly as on the robot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id or "local/%s" % root.resolve().name, root=root, return_uint8=True)
    indices = np.unique(np.linspace(0, len(ds) - 1, num=min(count, len(ds))).round().astype(int))
    return [preprocess.to_model_input(frame_rgb_uint8(ds[int(i)][IMAGE_KEY]))[None] for i in indices]


def parity(module: nn.Module, run: Any, inputs: Sequence[np.ndarray]) -> float:
    """Largest |reference - candidate| over all inputs."""
    worst = 0.0
    with torch.no_grad():
        for x in inputs:
            ref = module(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()
            got = np.asarray(run(x))
            if got.shape != ref.shape:
                raise ValueError("output shape %s, expected %s" % (got.shape, ref.shape))
            worst = max(worst, float(np.abs(got - ref).max()))
    return worst


def ort_latency_ms(session: Any, runs: int = 50) -> float:
    x = np.zeros(INPUT_SHAPE, dtype=np.float32)
    session.run([OUTPUT_NAME], {INPUT_NAME: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: x})
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("checkpoint", type=Path, help="train_bc.py checkpoint (model.pt)")
    parser.add_argument("--out", type=Path, default=None, help="ONNX file (default: policy.onnx beside the checkpoint)")
    parser.add_argument("--opset", type=int, default=TRT8_MAX_OPSET)
    parser.add_argument("--exporter", choices=("auto", "torchscript", "dynamo"), default="auto")
    parser.add_argument("--dataset-root", type=Path, default=None, help="dataset for the real-frame parity check")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--dataset-frames", type=int, default=32)
    parser.add_argument("--random-inputs", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max |onnxruntime - torch| allowed")
    parser.add_argument("--torchscript", type=Path, default=None,
                        help="also write a TorchScript file for policy_runner's torch backend")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import onnxruntime as ort

    model, ckpt = load_checkpoint(args.checkpoint)
    if ckpt["model_config"].get("use_state"):
        raise SystemExit("this policy also takes observation.state, but policy_runner feeds the image only "
                         "(docs/INTERFACES.md, Policy I/O); retrain without --use-state to deploy it")
    module = ImagePolicy(model).eval()
    out = args.out or args.checkpoint.with_name("policy.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    inputs = [rng.standard_normal(INPUT_SHAPE).astype(np.float32) for _ in range(args.random_inputs)]
    n_dataset = 0
    if args.dataset_root is not None:
        frames = dataset_inputs(args.dataset_root, args.repo_id, args.dataset_frames)
        inputs += frames
        n_dataset = len(frames)
    if not inputs:
        raise SystemExit("no parity inputs: use --random-inputs > 0 or --dataset-root")

    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp_onnx = Path(tmp) / out.name
        exporter = export_onnx(module, tmp_onnx, args.opset, args.exporter)
        graph = check_graph(tmp_onnx)
        session = ort.InferenceSession(str(tmp_onnx), providers=["CPUExecutionProvider"])
        max_diff = parity(module, lambda x: session.run([OUTPUT_NAME], {INPUT_NAME: x})[0], inputs)
        print("onnxruntime vs torch: max |diff| %.3g over %d random + %d dataset inputs (tolerance %.0e)"
              % (max_diff, args.random_inputs, n_dataset, args.tolerance))
        if not max_diff <= args.tolerance:
            raise SystemExit("parity check failed; nothing written")
        latency = ort_latency_ms(session)
        tmp_onnx.replace(out)

    opset = next(o.version for o in graph.opset_import if o.domain in ("", "ai.onnx"))
    export_info: dict[str, Any] = {
        "file": out.name,
        "sha256": sha256(out),
        "exporter": exporter,
        "opset": opset,
        "ir_version": graph.ir_version,
        "tensorrt8_compatible_opset": opset <= TRT8_MAX_OPSET,
        "parity": {"max_abs_diff": max_diff, "tolerance": args.tolerance, "random_inputs": args.random_inputs,
                   "dataset_frames": n_dataset},
        "onnxruntime": ort.__version__,
        "ort_cpu_latency_ms_p50": round(latency, 2),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
    }

    if args.torchscript is not None:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            traced = torch.jit.trace(module, torch.zeros(INPUT_SHAPE))
        args.torchscript.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(args.torchscript))
        reloaded = torch.jit.load(str(args.torchscript))
        ts_diff = parity(module, lambda x: reloaded(torch.from_numpy(x)).detach().numpy(), inputs)
        if not ts_diff <= args.tolerance:
            args.torchscript.unlink()
            raise SystemExit("TorchScript parity check failed (max |diff| %.3g)" % ts_diff)
        export_info["torchscript"] = {"file": args.torchscript.name, "max_abs_diff": ts_diff}
        print("torchscript vs torch: max |diff| %.3g -> %s" % (ts_diff, args.torchscript))

    meta = dict(ckpt["policy_meta"], export=export_info)
    write_meta(out.with_name(META_FILENAME), meta)
    if args.torchscript is not None and args.torchscript.parent.resolve() != out.parent.resolve():
        write_meta(args.torchscript.with_name(META_FILENAME), meta)
    print("wrote %s (%s exporter, opset %d, onnxruntime CPU p50 %.1f ms) + %s"
          % (out, exporter, opset, latency, META_FILENAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
