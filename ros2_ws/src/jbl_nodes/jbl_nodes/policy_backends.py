"""Inference backends for policy_runner behind one interface.

    backend = create_backend("onnxruntime", "/models/bc/policy.onnx")
    action = backend.infer(x)      # x: float32 [1, 3, 224, 224] -> float32 [1, 2]

Backends: ``onnxruntime`` (CUDA provider when present, else CPU; ``intra_op_threads=N`` sizes
its CPU thread pool, 0 = ONNX Runtime's default), ``tensorrt`` (TensorRT 8.x engine on the
Jetson), ``torch`` (TorchScript file) and ``constant`` (fixed action, for tests).
Each heavy dependency is imported only by the backend that needs it, so this module imports
with numpy + OpenCV alone.

``policy_meta.json`` beside the model file is checked before the model is loaded. The fields
looked for, all optional (an absent field is reported, not guessed):

    input_shape | input_size | image_size  [1,3,224,224], [3,224,224] or [224,224]
    mean, std            ImageNet values; top level or inside "normalization"/"normalisation"
    normalization        or the string "imagenet"
    action_scale         [0.25, 1.5] or {"v": 0.25, "w": 1.5}; or top-level v_max / w_max
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from jbl_nodes.preprocess import IMAGENET_MEAN, IMAGENET_STD, MODEL_HW, V_MAX, W_MAX

INPUT_SHAPE = (1, 3) + tuple(MODEL_HW)
OUTPUT_SHAPE = (1, 2)
INPUT_NAME = "image"
OUTPUT_NAME = "action"
META_FILENAME = "policy_meta.json"
BACKEND_NAMES = ("onnxruntime", "tensorrt", "torch", "constant")


class PolicyMetaError(ValueError):
    """policy_meta.json disagrees with the preprocessing or action mapping used on the robot."""


class Backend(object):
    """Common interface: ``infer`` validates shapes around the backend-specific ``_infer``."""

    name = "base"

    def __init__(self) -> None:
        self.meta: Optional[Dict[str, Any]] = None
        self.meta_checked: List[str] = []
        self.description = self.name

    def infer(self, x: np.ndarray) -> np.ndarray:
        if x.shape != INPUT_SHAPE:
            raise ValueError("policy input must have shape %s, got %s" % (INPUT_SHAPE, x.shape))
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.asarray(self._infer(x), dtype=np.float32)
        if y.size != 2:
            raise ValueError(
                "%s backend returned %d values (shape %s), expected 2" % (self.name, y.size, y.shape)
            )
        return y.reshape(OUTPUT_SHAPE)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        """Release native resources; the backend must not be used afterwards."""


class ConstantBackend(Backend):
    """Returns the same normalised action for every input."""

    name = "constant"

    def __init__(self, action: Sequence[float] = (0.0, 0.0)) -> None:
        super().__init__()
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 2 or not np.all(np.isfinite(a)):
            raise ValueError("constant action must be two finite numbers, got %r" % (action,))
        self._action = a.reshape(OUTPUT_SHAPE)
        self.description = "constant %s" % [round(float(v), 4) for v in a]

    def _infer(self, x: np.ndarray) -> np.ndarray:
        return self._action.copy()


class OnnxRuntimeBackend(Backend):
    """ONNX Runtime session; uses the CUDA provider when the installed build has it.

    ``intra_op_threads`` sizes ONNX Runtime's intra-op thread pool. 0 creates the session
    exactly as before the option existed, with ONNX Runtime's default: on Linux (1.31) one
    thread per physical core of the host, the workers pinned one per core and spinning after
    each run. A container limited to fewer cores (docker --cpuset-cpus) does not change that
    count, so the pool oversubscribes the cores it may use, and the pin to a core outside the
    set fails ("pthread_setaffinity_np failed ... Specify the number of threads explicitly").
    N > 0 makes a pool of N threads (the caller's included), which ONNX Runtime does not pin.
    """

    name = "onnxruntime"
    DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def __init__(
        self, model_path: str, providers: Optional[Sequence[str]] = None, intra_op_threads: int = 0
    ) -> None:
        super().__init__()
        self.intra_op_threads = _intra_op_threads(intra_op_threads)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "the onnxruntime backend needs the onnxruntime package "
                "(pip install onnxruntime, or the onnxruntime-gpu wheel built for JetPack)"
            ) from exc
        available = ort.get_available_providers()
        wanted = list(providers) if providers else list(self.DEFAULT_PROVIDERS)
        use = [p for p in wanted if p in available]
        if not use:
            raise RuntimeError("none of the providers %s is available (have %s)" % (wanted, available))
        if self.intra_op_threads:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.intra_op_threads
            self._session = ort.InferenceSession(model_path, sess_options=options, providers=use)
        else:  # ONNX Runtime's own defaults, as every run up to v4 used them
            self._session = ort.InferenceSession(model_path, providers=use)
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        self._input = _pick_io(inputs, INPUT_NAME, "input")
        self._output = _pick_io(outputs, OUTPUT_NAME, "output").name
        _check_static_dims(self._input.shape, INPUT_SHAPE, "ONNX input %r" % self._input.name)
        self.providers = self._session.get_providers()
        self.description = "onnxruntime %s %s" % (ort.__version__, ",".join(self.providers))
        if self.intra_op_threads:
            self.description += ", %d intra-op threads" % self.intra_op_threads

    def _infer(self, x: np.ndarray) -> np.ndarray:
        return self._session.run([self._output], {self._input.name: x})[0]

    def close(self) -> None:
        self._session = None


class TorchBackend(Backend):
    """TorchScript module (torch.jit.save output) taking the image tensor, returning [1, 2]."""

    name = "torch"

    def __init__(self, model_path: str, device: str = "auto") -> None:
        super().__init__()
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "the torch backend needs PyTorch (on the Jetson: the NVIDIA JetPack wheel)"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device)
        self._module = torch.jit.load(model_path, map_location=self._device)
        self._module.eval()
        self.description = "torch %s %s" % (torch.__version__, self._device)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self._module(torch.from_numpy(x).to(self._device))
        if isinstance(out, dict):
            out = out[OUTPUT_NAME]
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().float().cpu().numpy()

    def close(self) -> None:
        self._module = None


class TensorRTBackend(Backend):
    """Serialized TensorRT 8.x engine, executed with ``execute_async_v2``.

    NOT TESTED ON HARDWARE: written against the TensorRT 8.x binding API (JetPack 4.6 / 5.x)
    and exercised here only with stand-in modules. Device memory comes from pycuda when it is
    installed, else from cuda-python; with neither, construction raises ImportError.
    """

    name = "tensorrt"

    def __init__(self, engine_path: str) -> None:
        super().__init__()
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "the tensorrt backend needs the TensorRT Python bindings (JetPack's python3-libnvinfer); "
                "use the onnxruntime backend where TensorRT is not installed"
            ) from exc
        if not hasattr(trt.ICudaEngine, "num_bindings"):
            raise RuntimeError(
                "TensorRT %s has no binding API; this backend supports TensorRT 8.x"
                % getattr(trt, "__version__", "?")
            )
        self._cuda = _make_cuda_memory()
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            blob = f.read()
        with self._cuda.activated():
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(blob)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize %s (engine built for another GPU or version?)"
                    % engine_path
                )
            self._context = self._engine.create_execution_context()
            self._setup_bindings(trt)
        self.description = "tensorrt %s (%s)" % (getattr(trt, "__version__", "?"), self._cuda.name)

    def _setup_bindings(self, trt: Any) -> None:
        engine, context = self._engine, self._context
        inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "expected an engine with 1 input and 1 output, got %d and %d" % (len(inputs), len(outputs))
            )
        self._in_index, self._out_index = inputs[0], outputs[0]
        engine_shape = tuple(engine.get_binding_shape(self._in_index))
        _check_static_dims(
            engine_shape, INPUT_SHAPE, "TensorRT input %r" % engine.get_binding_name(self._in_index)
        )
        if -1 in engine_shape:  # dynamic batch: pin it to the one shape we use
            context.set_binding_shape(self._in_index, INPUT_SHAPE)
        self._host: Dict[int, np.ndarray] = {}
        self._device: Dict[int, int] = {}
        for index in (self._in_index, self._out_index):
            shape = tuple(context.get_binding_shape(index))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(index)))
            self._host[index] = self._cuda.host_empty(shape, dtype)
            self._device[index] = self._cuda.device_alloc(self._host[index].nbytes)
        bindings = [0] * engine.num_bindings
        for index, ptr in self._device.items():
            bindings[index] = int(ptr)
        self._bindings = bindings

    def _infer(self, x: np.ndarray) -> np.ndarray:
        host_in, host_out = self._host[self._in_index], self._host[self._out_index]
        np.copyto(host_in, x.reshape(host_in.shape), casting="same_kind")
        with self._cuda.activated():
            self._cuda.copy_to_device(self._device[self._in_index], host_in)
            if not self._context.execute_async_v2(
                bindings=self._bindings, stream_handle=self._cuda.stream_handle
            ):
                raise RuntimeError("TensorRT execute_async_v2 failed")
            self._cuda.copy_to_host(host_out, self._device[self._out_index])
            self._cuda.synchronize()
        return host_out.astype(np.float32)

    def close(self) -> None:
        if self._cuda is None:
            return
        with self._cuda.activated():
            self._context = None
            self._engine = None
            self._runtime = None
            self._cuda.free_all()
        self._cuda = None


class _PyCudaMemory(object):
    """Device buffers and a stream through pycuda, on the device's primary context.

    The context is pushed around every call instead of using pycuda.autoinit, whose context is
    bound to the importing thread and would break if a callback ran on another thread.
    """

    name = "pycuda"

    def __init__(self) -> None:
        import pycuda.driver as cuda

        cuda.init()
        self._cuda = cuda
        self._ctx = cuda.Device(0).retain_primary_context()
        self._allocations: List[Any] = []
        with self.activated():
            self._stream = cuda.Stream()

    def activated(self) -> "_ContextGuard":
        return _ContextGuard(self._ctx.push, self._cuda.Context.pop)

    @property
    def stream_handle(self) -> int:
        return self._stream.handle

    def host_empty(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return self._cuda.pagelocked_empty(shape, dtype)

    def device_alloc(self, nbytes: int) -> int:
        allocation = self._cuda.mem_alloc(nbytes)
        self._allocations.append(allocation)
        return int(allocation)

    def copy_to_device(self, device_ptr: int, host: np.ndarray) -> None:
        self._cuda.memcpy_htod_async(device_ptr, host, self._stream)

    def copy_to_host(self, host: np.ndarray, device_ptr: int) -> None:
        self._cuda.memcpy_dtoh_async(host, device_ptr, self._stream)

    def synchronize(self) -> None:
        self._stream.synchronize()

    def free_all(self) -> None:
        for allocation in self._allocations:
            allocation.free()
        self._allocations = []


class _CudaPythonMemory(object):
    """Same operations through NVIDIA cuda-python (runtime API, implicit primary context)."""

    name = "cuda-python"

    def __init__(self) -> None:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart
        self._cudart = cudart
        self._allocations: List[int] = []
        self._stream = self._check(cudart.cudaStreamCreate())

    def _check(self, result: Any) -> Any:
        err, rest = result[0], result[1:]
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("CUDA runtime error %s" % (err,))
        return rest[0] if len(rest) == 1 else (rest or None)

    def activated(self) -> "_ContextGuard":
        return _ContextGuard(lambda: None, lambda: None)

    @property
    def stream_handle(self) -> int:
        stream = self._stream
        return int(stream.getPtr()) if hasattr(stream, "getPtr") else int(stream)

    def host_empty(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    def device_alloc(self, nbytes: int) -> int:
        ptr = int(self._check(self._cudart.cudaMalloc(nbytes)))
        self._allocations.append(ptr)
        return ptr

    def copy_to_device(self, device_ptr: int, host: np.ndarray) -> None:
        kind = self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self._check(
            self._cudart.cudaMemcpyAsync(device_ptr, host.ctypes.data, host.nbytes, kind, self._stream)
        )

    def copy_to_host(self, host: np.ndarray, device_ptr: int) -> None:
        kind = self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        self._check(
            self._cudart.cudaMemcpyAsync(host.ctypes.data, device_ptr, host.nbytes, kind, self._stream)
        )

    def synchronize(self) -> None:
        self._check(self._cudart.cudaStreamSynchronize(self._stream))

    def free_all(self) -> None:
        for ptr in self._allocations:
            self._check(self._cudart.cudaFree(ptr))
        self._allocations = []
        self._check(self._cudart.cudaStreamDestroy(self._stream))


class _ContextGuard(object):
    def __init__(self, enter: Any, leave: Any) -> None:
        self._enter, self._leave = enter, leave

    def __enter__(self) -> None:
        self._enter()

    def __exit__(self, *exc: Any) -> None:
        self._leave()


def _make_cuda_memory() -> Any:
    errors = []
    for cls in (_PyCudaMemory, _CudaPythonMemory):
        try:
            return cls()
        except ImportError as exc:
            errors.append("%s: %s" % (cls.name, exc))
    raise ImportError(
        "the tensorrt backend needs pycuda or cuda-python for device memory (%s)" % "; ".join(errors)
    )


def _intra_op_threads(value: Any) -> int:
    """ONNX Runtime intra-op thread count: an int >= 0 (0 = ONNX Runtime's default)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            "intra_op_threads must be an integer >= 0 (0 = ONNX Runtime's default), got %r" % (value,)
        )
    return int(value)


def _pick_io(entries: Sequence[Any], preferred: str, kind: str) -> Any:
    for entry in entries:
        if entry.name == preferred:
            return entry
    if len(entries) == 1:
        return entries[0]
    raise ValueError("model has %d %ss and none is named %r" % (len(entries), kind, preferred))


def _check_static_dims(shape: Sequence[Any], expected: Sequence[int], what: str) -> None:
    """Compare a model's declared shape with ``expected``, ignoring dynamic (named/-1/None) dims."""
    if len(shape) != len(expected):
        raise ValueError("%s has shape %s, expected %s" % (what, list(shape), list(expected)))
    for got, want in zip(shape, expected):
        if isinstance(got, int) and got >= 0 and got != want:
            raise ValueError("%s has shape %s, expected %s" % (what, list(shape), list(expected)))


def meta_path_for(model_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), META_FILENAME)


def load_policy_meta(model_path: str) -> Optional[Dict[str, Any]]:
    """policy_meta.json beside ``model_path`` as a dict, or None when there is none."""
    path = meta_path_for(model_path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise PolicyMetaError("%s must hold a JSON object" % path)
    return meta


def _lookup(
    meta: Dict[str, Any], keys: Sequence[str], sections: Sequence[str] = ()
) -> Tuple[Optional[str], Any]:
    for section in (None,) + tuple(sections):
        scope = meta if section is None else meta.get(section)
        if not isinstance(scope, dict):
            continue
        for key in keys:
            if key in scope:
                return (key if section is None else "%s.%s" % (section, key)), scope[key]
    return None, None


def check_policy_meta(meta: Dict[str, Any]) -> List[str]:
    """Raise PolicyMetaError on any mismatch; return the names of the fields that were checked."""
    checked = []
    norm_sections = ("normalization", "normalisation", "preprocess", "input")

    key, size = _lookup(
        meta, ("input_shape", "input_size", "image_size", "input_hw"), ("input", "preprocess")
    )
    if key is not None:
        dims = [int(v) for v in size] if isinstance(size, (list, tuple)) else [int(size), int(size)]
        if len(dims) not in (2, 3, 4) or tuple(dims[-2:]) != tuple(MODEL_HW):
            raise PolicyMetaError("%s = %s, but the robot feeds %s images" % (key, size, list(INPUT_SHAPE)))
        if len(dims) >= 3 and dims[-3] != 3:
            raise PolicyMetaError("%s = %s, but the robot feeds 3-channel images" % (key, size))
        checked.append(key)

    key, norm = _lookup(meta, ("normalization", "normalisation"))
    if isinstance(norm, str):
        if norm.strip().lower() != "imagenet":
            raise PolicyMetaError("%s = %r, but the robot uses ImageNet mean/std" % (key, norm))
        checked.append(key)
    for names, reference in ((("mean", "image_mean"), IMAGENET_MEAN), (("std", "image_std"), IMAGENET_STD)):
        key, value = _lookup(meta, names, norm_sections)
        if key is None:
            continue
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.shape != reference.shape or not np.allclose(value, reference, atol=1e-3):
            raise PolicyMetaError(
                "%s = %s, but the robot normalises with %s" % (key, value.tolist(), reference.tolist())
            )
        checked.append(key)

    key, scale = _lookup(meta, ("action_scale",))
    if key is None:
        v_key, v_max = _lookup(meta, ("v_max",))
        w_key, w_max = _lookup(meta, ("w_max",))
        if v_key is not None and w_key is not None:
            key, scale = "v_max/w_max", [v_max, w_max]
    if key is not None:
        if isinstance(scale, dict):
            scale = [scale.get("v", scale.get("v_max")), scale.get("w", scale.get("w_max"))]
        if (
            len(scale) != 2
            or None in scale
            or not np.allclose([float(s) for s in scale], [V_MAX, W_MAX], atol=1e-6)
        ):
            raise PolicyMetaError(
                "%s = %s, but the robot maps actions with v_max=%s, w_max=%s" % (key, scale, V_MAX, W_MAX)
            )
        checked.append(key)
    return checked


def create_backend(
    name: str,
    model_path: str = "",
    constant_action: Sequence[float] = (0.0, 0.0),
    providers: Optional[Sequence[str]] = None,
    torch_device: str = "auto",
    intra_op_threads: int = 0,
) -> Backend:
    """Backend by name (see BACKEND_NAMES).

    Model-backed backends check policy_meta.json first; ``backend.meta`` is None when the file
    is absent (the caller decides whether to warn) and ``backend.meta_checked`` lists the
    fields that were verified. ``intra_op_threads`` sizes the onnxruntime backend's intra-op
    thread pool (0 = ONNX Runtime's default; see OnnxRuntimeBackend); the other backends
    ignore it, but a negative value is refused for every backend.
    """
    key = name.strip().lower()
    if key not in BACKEND_NAMES:
        raise ValueError("unknown backend %r (choose from %s)" % (name, ", ".join(BACKEND_NAMES)))
    intra_op_threads = _intra_op_threads(intra_op_threads)
    if key == "constant":
        return ConstantBackend(constant_action)
    if not model_path:
        raise ValueError("backend %r needs a model_path" % key)
    if not os.path.isfile(model_path):
        raise FileNotFoundError("model file not found: %s" % model_path)
    meta = load_policy_meta(model_path)
    checked = check_policy_meta(meta) if meta is not None else []
    if key == "onnxruntime":
        backend: Backend = OnnxRuntimeBackend(model_path, providers, intra_op_threads)
    elif key == "torch":
        backend = TorchBackend(model_path, torch_device)
    else:
        backend = TensorRTBackend(model_path)
    backend.meta = meta
    backend.meta_checked = checked
    return backend
