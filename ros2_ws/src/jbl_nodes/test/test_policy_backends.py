"""policy_backends: interface, factory, policy_meta.json checks and each backend.

ONNX Runtime and TorchScript run for real (host venv); TensorRT has no GPU here, so its
control flow runs against stand-in tensorrt / pycuda / cuda-python modules. How the ONNX
Runtime session is configured (intra_op_threads) is also checked against a stand-in
onnxruntime module, so that part runs in the ROS containers too, which have no ONNX Runtime.
"""

import ctypes
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from jbl_nodes import policy_backends as pb
from jbl_nodes.preprocess import IMAGENET_MEAN, IMAGENET_STD, action_to_twist, preprocess

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None
needs_torch_ort = pytest.mark.skipif(not (HAVE_TORCH and HAVE_ORT), reason="needs torch and onnxruntime")
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")

GOOD_META = {
    "input_shape": [1, 3, 224, 224],
    "normalization": {"mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist()},
    "action_scale": {"v": 0.25, "w": 1.5},
    "dataset_stats": {"action": {"mean": [0.1, 0.0], "std": [0.05, 0.4]}},
}


def _input(value=0.0):
    return np.full(pb.INPUT_SHAPE, value, dtype=np.float32)


# ---------------------------------------------------------------- interface and factory


def test_constant_backend_returns_its_action():
    backend = pb.create_backend("constant", constant_action=[0.4, -0.2])
    assert isinstance(backend, pb.ConstantBackend) and backend.name == "constant"
    out = backend.infer(_input())
    assert out.shape == (1, 2) and out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.4, -0.2]], rtol=1e-6)
    out[0, 0] = 9.0  # callers may not corrupt the backend's state
    np.testing.assert_allclose(backend.infer(_input()), [[0.4, -0.2]], rtol=1e-6)


@pytest.mark.parametrize("action", [[1.0], [1.0, 2.0, 3.0], [float("nan"), 0.0]])
def test_constant_backend_rejects_bad_actions(action):
    with pytest.raises(ValueError):
        pb.ConstantBackend(action)


@pytest.mark.parametrize("shape", [(3, 224, 224), (1, 3, 240, 320), (2, 3, 224, 224)])
def test_infer_rejects_wrong_input_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        pb.ConstantBackend().infer(np.zeros(shape, np.float32))


def test_infer_rejects_wrong_output_size():
    class Broken(pb.Backend):
        name = "broken"

        def _infer(self, x):
            return np.zeros((1, 3), np.float32)

    with pytest.raises(ValueError, match="3 values"):
        Broken().infer(_input())


def test_factory_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        pb.create_backend("tflite", "x")
    with pytest.raises(ValueError, match="needs a model_path"):
        pb.create_backend("onnxruntime", "")
    with pytest.raises(FileNotFoundError):
        pb.create_backend("onnxruntime", str(tmp_path / "missing.onnx"))


def test_factory_is_case_and_space_tolerant():
    assert pb.create_backend("  Constant ").name == "constant"


# ---------------------------------------------------------------- policy_meta.json


def test_good_meta_passes_and_reports_checked_fields():
    checked = pb.check_policy_meta(GOOD_META)
    assert checked == ["input_shape", "normalization.mean", "normalization.std", "action_scale"]


@pytest.mark.parametrize(
    "meta",
    [
        {"input_size": [224, 224], "normalisation": "imagenet", "v_max": 0.25, "w_max": 1.5},
        {
            "image_size": 224,
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
            "action_scale": [0.25, 1.5],
        },
        {"input": {"input_shape": [3, 224, 224]}, "preprocess": {"mean": [0.485, 0.456, 0.406]}},
        {"dataset_stats": {"observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}},  # nothing to check
    ],
)
def test_meta_aliases_are_understood(meta):
    pb.check_policy_meta(meta)


@pytest.mark.parametrize(
    "meta,match",
    [
        ({"input_shape": [1, 3, 240, 320]}, "input_shape"),
        ({"input_shape": [1, 1, 224, 224]}, "3-channel"),
        ({"image_size": 96}, "image_size"),
        ({"normalization": "none"}, "ImageNet"),
        ({"normalization": {"mean": [0.5, 0.5, 0.5], "std": IMAGENET_STD.tolist()}}, "normalization.mean"),
        ({"std": [1.0, 1.0, 1.0]}, "std"),
        ({"action_scale": [0.5, 1.5]}, "action_scale"),
        ({"v_max": 0.25, "w_max": 2.0}, "v_max/w_max"),
    ],
)
def test_meta_mismatches_are_rejected(meta, match):
    with pytest.raises(pb.PolicyMetaError, match=match):
        pb.check_policy_meta(meta)


def test_meta_is_loaded_from_beside_the_model(tmp_path):
    model = tmp_path / "policy.onnx"
    assert pb.load_policy_meta(str(model)) is None
    (tmp_path / pb.META_FILENAME).write_text(json.dumps(GOOD_META))
    assert pb.load_policy_meta(str(model)) == GOOD_META
    (tmp_path / pb.META_FILENAME).write_text("[1, 2]")
    with pytest.raises(pb.PolicyMetaError, match="JSON object"):
        pb.load_policy_meta(str(model))


# ---------------------------------------------------------------- ONNX Runtime and TorchScript


def _tiny_policy():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=8, stride=8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
        nn.Tanh(),
    ).eval()


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    if not (HAVE_TORCH and HAVE_ORT):
        pytest.skip("needs torch and onnxruntime")
    import torch

    model = _tiny_policy()
    path = tmp_path_factory.mktemp("onnx") / "policy.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(pb.INPUT_SHAPE),),
        str(path),
        input_names=["image"],
        output_names=["action"],
        external_data=False,
    )
    return model, path


def _reference(model, x):
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


@needs_torch_ort
def test_onnxruntime_matches_torch(onnx_model):
    model, path = onnx_model
    backend = pb.create_backend("onnxruntime", str(path), providers=["CPUExecutionProvider"])
    assert backend.name == "onnxruntime" and backend.providers == ["CPUExecutionProvider"]
    assert backend.meta is None
    x = np.random.default_rng(0).standard_normal(pb.INPUT_SHAPE).astype(np.float32)
    np.testing.assert_allclose(backend.infer(x), _reference(model, x), atol=1e-5)
    backend.close()


@needs_torch_ort
def test_onnxruntime_default_providers_fall_back_to_cpu(onnx_model):
    _, path = onnx_model
    backend = pb.OnnxRuntimeBackend(str(path))
    assert backend.providers[-1] == "CPUExecutionProvider"
    with pytest.raises(RuntimeError, match="none of the providers"):
        pb.OnnxRuntimeBackend(str(path), providers=["NoSuchExecutionProvider"])


@needs_torch_ort
def test_onnxruntime_on_a_preprocessed_camera_frame(onnx_model):
    model, path = onnx_model
    backend = pb.create_backend("onnxruntime", str(path))
    frame = np.random.default_rng(3).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    x = preprocess(frame, is_bgr=True)
    out = backend.infer(x)
    np.testing.assert_allclose(out, _reference(model, x), atol=1e-5)
    v, w = action_to_twist(out)
    assert abs(v) <= 0.25 and abs(w) <= 1.5


@needs_torch_ort
def test_onnxruntime_checks_meta_before_loading(onnx_model, tmp_path):
    _, path = onnx_model
    model = tmp_path / "policy.onnx"
    model.write_bytes(path.read_bytes())
    (tmp_path / pb.META_FILENAME).write_text(json.dumps(GOOD_META))
    backend = pb.create_backend("onnxruntime", str(model))
    assert backend.meta == GOOD_META and "action_scale" in backend.meta_checked

    (tmp_path / pb.META_FILENAME).write_text(json.dumps(dict(GOOD_META, input_shape=[1, 3, 96, 96])))
    with pytest.raises(pb.PolicyMetaError):
        pb.create_backend("onnxruntime", str(model))


@pytest.mark.skipif(not HAVE_ORT, reason="needs onnxruntime")
def test_onnxruntime_rejects_a_model_with_another_input_size(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image"], ["pooled"], axes=[2, 3], keepdims=0),
            helper.make_node("Slice", ["pooled", "starts", "ends", "axes"], ["action"]),
        ],
        "wrong_size",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 96, 96])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 2])],
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("ends", TensorProto.INT64, [1], [2]),
            helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
        ],
    )
    path = tmp_path / "wrong.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))
    with pytest.raises(ValueError, match="96"):
        pb.create_backend("onnxruntime", str(path))


# ---------------------------------------------------------------- ONNX Runtime's thread pool


def _fake_onnxruntime(calls):
    """Stand-in onnxruntime that records how sessions are created (no ONNX Runtime needed)."""
    ort = types.ModuleType("onnxruntime")
    ort.__version__ = "0.0-stand-in"

    class SessionOptions(object):
        def __init__(self):
            self.intra_op_num_threads = 0

    class _IO(object):
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class InferenceSession(object):
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            self._providers = list(kwargs["providers"])

        def get_inputs(self):
            return [_IO("image", list(pb.INPUT_SHAPE))]

        def get_outputs(self):
            return [_IO("action", [1, 2])]

        def get_providers(self):
            return self._providers

    ort.SessionOptions = SessionOptions
    ort.InferenceSession = InferenceSession
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    return ort


def test_onnxruntime_intra_op_threads_reach_the_session(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_onnxruntime(calls))
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"")

    default = pb.create_backend("onnxruntime", str(model))
    # 0 keeps the session exactly as every run before the option made it: no session options.
    assert calls[-1] == (str(model), {"providers": ["CPUExecutionProvider"]})
    assert default.intra_op_threads == 0 and "intra-op" not in default.description

    sized = pb.create_backend("onnxruntime", str(model), intra_op_threads=2)
    assert calls[-1][1]["sess_options"].intra_op_num_threads == 2
    assert sized.intra_op_threads == 2 and sized.description.endswith(", 2 intra-op threads")
    assert pb.OnnxRuntimeBackend(str(model), intra_op_threads=np.int64(3)).intra_op_threads == 3


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True, None])
def test_bad_intra_op_threads_are_refused_before_a_session_exists(monkeypatch, tmp_path, bad):
    calls = []
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_onnxruntime(calls))
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"")
    with pytest.raises(ValueError, match="intra_op_threads"):
        pb.create_backend("onnxruntime", str(model), intra_op_threads=bad)
    with pytest.raises(ValueError, match="intra_op_threads"):
        pb.OnnxRuntimeBackend(str(model), intra_op_threads=bad)
    with pytest.raises(ValueError, match="intra_op_threads"):
        pb.create_backend("constant", intra_op_threads=bad)
    assert calls == []


def _thread_ids():
    return {int(t) for t in os.listdir("/proc/self/task")}


@needs_torch_ort
@pytest.mark.skipif(
    not (os.path.isdir("/proc/self/task") and hasattr(os, "sched_getaffinity")), reason="needs Linux /proc"
)
@pytest.mark.parametrize("threads", [1, 2])
def test_onnxruntime_intra_op_threads_size_the_real_pool(onnx_model, threads):
    model, path = onnx_model
    default = pb.create_backend("onnxruntime", str(path), providers=["CPUExecutionProvider"])
    assert default._session.get_session_options().intra_op_num_threads == 0
    default.close()

    before = _thread_ids()
    backend = pb.create_backend(
        "onnxruntime", str(path), providers=["CPUExecutionProvider"], intra_op_threads=threads
    )
    created = _thread_ids() - before
    assert backend._session.get_session_options().intra_op_num_threads == threads
    assert len(created) == threads - 1, "the pool counts the calling thread"
    for tid in created:  # an explicit size is not pinned: each worker may use every allowed core
        assert os.sched_getaffinity(tid) == os.sched_getaffinity(0)
    x = np.random.default_rng(2).standard_normal(pb.INPUT_SHAPE).astype(np.float32)
    np.testing.assert_allclose(backend.infer(x), _reference(model, x), atol=1e-5)
    backend.close()


@needs_torch
def test_torchscript_backend_matches_torch(tmp_path):
    import torch

    model = _tiny_policy()
    path = tmp_path / "policy.pt"
    torch.jit.trace(model, torch.zeros(pb.INPUT_SHAPE)).save(str(path))
    backend = pb.create_backend("torch", str(path))
    assert backend.name == "torch" and "cpu" in backend.description
    x = np.random.default_rng(1).standard_normal(pb.INPUT_SHAPE).astype(np.float32)
    np.testing.assert_allclose(backend.infer(x), _reference(model, x), atol=1e-5)


def test_missing_runtime_gives_clear_import_errors(monkeypatch, tmp_path):
    model = tmp_path / "policy.bin"
    model.write_bytes(b"")
    for module, backend in (("onnxruntime", "onnxruntime"), ("torch", "torch"), ("tensorrt", "tensorrt")):
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(ImportError, match=module):
            pb.create_backend(backend, str(model))


# ---------------------------------------------------------------- TensorRT with stand-in modules


class _DeviceHeap(object):
    """Fake device memory: integer 'pointers' mapping to byte arrays."""

    def __init__(self):
        self.blocks = {}
        self._next = 0x1000

    def alloc(self, nbytes):
        ptr = self._next
        self._next += 0x1000 + nbytes
        self.blocks[ptr] = np.zeros(nbytes, np.uint8)
        return ptr


def _fake_tensorrt(heap, with_binding_api=True, dynamic_batch=True):
    trt = types.ModuleType("tensorrt")
    trt.__version__ = "8.2.1.8"

    class Logger(object):
        WARNING = 2

        def __init__(self, level):
            self.level = level

    class Context(object):
        def __init__(self):
            self.input_shape = (-1, 3, 224, 224) if dynamic_batch else pb.INPUT_SHAPE

        def set_binding_shape(self, index, shape):
            assert index == 0
            self.input_shape = tuple(shape)

        def get_binding_shape(self, index):
            return self.input_shape if index == 0 else (1, 2)

        def execute_async_v2(self, bindings, stream_handle):
            assert stream_handle and -1 not in self.input_shape
            x = heap.blocks[bindings[0]].view(np.float32)
            heap.blocks[bindings[1]][:] = np.array([x.mean(), -x.mean()], np.float32).view(np.uint8)
            return True

    class ICudaEngine(object):
        if with_binding_api:
            num_bindings = 2

        def binding_is_input(self, index):
            return index == 0

        def get_binding_name(self, index):
            return ["image", "action"][index]

        def get_binding_shape(self, index):
            return (
                (-1, 3, 224, 224)
                if (index == 0 and dynamic_batch)
                else ((1, 3, 224, 224) if index == 0 else (1, 2))
            )

        def get_binding_dtype(self, index):
            return np.float32

        def create_execution_context(self):
            return Context()

    class Runtime(object):
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, blob):
            return ICudaEngine() if blob == b"fake-engine" else None

    trt.Logger, trt.Runtime, trt.ICudaEngine = Logger, Runtime, ICudaEngine
    trt.nptype = lambda dtype: dtype
    return trt


def _fake_pycuda(heap, context_stack):
    driver = types.ModuleType("pycuda.driver")

    class Allocation(object):
        def __init__(self, nbytes):
            self.ptr = heap.alloc(nbytes)

        def __int__(self):
            return self.ptr

        def free(self):
            del heap.blocks[self.ptr]

    class PrimaryContext(object):
        def push(self):
            context_stack.append(self)

    class Context(object):
        @staticmethod
        def pop():
            context_stack.pop()

    class Device(object):
        def __init__(self, ordinal):
            assert ordinal == 0

        def retain_primary_context(self):
            return PrimaryContext()

    class Stream(object):
        handle = 0xBEEF

        def __init__(self):
            assert context_stack, "pycuda calls need a current context"

        def synchronize(self):
            assert context_stack

    def memcpy_htod_async(ptr, host, stream):
        assert context_stack
        heap.blocks[int(ptr)][:] = np.ascontiguousarray(host).view(np.uint8).reshape(-1)

    def memcpy_dtoh_async(host, ptr, stream):
        assert context_stack
        host[...] = heap.blocks[int(ptr)].view(host.dtype).reshape(host.shape)

    driver.init = lambda: None
    driver.Device, driver.Context, driver.Stream = Device, Context, Stream
    driver.mem_alloc = Allocation
    driver.pagelocked_empty = lambda shape, dtype: np.empty(shape, dtype)
    driver.memcpy_htod_async, driver.memcpy_dtoh_async = memcpy_htod_async, memcpy_dtoh_async
    package = types.ModuleType("pycuda")
    package.driver = driver
    return {"pycuda": package, "pycuda.driver": driver}


def _fake_cuda_python(heap):
    runtime = types.ModuleType("cuda.bindings.runtime")

    class cudaError_t(object):
        cudaSuccess = 0

    class cudaMemcpyKind(object):
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    class Stream(object):
        def getPtr(self):
            return 0xCAFE

    ok = cudaError_t.cudaSuccess

    def cudaMemcpyAsync(dst, src, nbytes, kind, stream):
        if kind == cudaMemcpyKind.cudaMemcpyHostToDevice:
            ctypes.memmove(heap.blocks[dst].ctypes.data, src, nbytes)
        else:
            ctypes.memmove(dst, heap.blocks[src].ctypes.data, nbytes)
        return (ok,)

    def cudaFree(ptr):
        del heap.blocks[ptr]
        return (ok,)

    runtime.cudaError_t, runtime.cudaMemcpyKind = cudaError_t, cudaMemcpyKind
    runtime.cudaStreamCreate = lambda: (ok, Stream())
    runtime.cudaStreamSynchronize = lambda stream: (ok,)
    runtime.cudaStreamDestroy = lambda stream: (ok,)
    runtime.cudaMalloc = lambda nbytes: (ok, heap.alloc(nbytes))
    runtime.cudaMemcpyAsync, runtime.cudaFree = cudaMemcpyAsync, cudaFree
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = runtime
    package = types.ModuleType("cuda")
    package.bindings = bindings
    return {"cuda": package, "cuda.bindings": bindings, "cuda.bindings.runtime": runtime}


def _block_modules(monkeypatch, names):
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


PYCUDA_MODULES = ("pycuda", "pycuda.driver")
CUDA_PYTHON_MODULES = ("cuda", "cuda.bindings", "cuda.bindings.runtime", "cuda.cudart")


@pytest.fixture
def engine_file(tmp_path):
    path = tmp_path / "policy.engine"
    path.write_bytes(b"fake-engine")
    return path


@pytest.mark.parametrize("memory", ["pycuda", "cuda-python"])
@pytest.mark.parametrize("dynamic_batch", [True, False])
def test_tensorrt_flow_with_stand_in_modules(monkeypatch, engine_file, memory, dynamic_batch):
    heap, context_stack = _DeviceHeap(), []
    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(heap, dynamic_batch=dynamic_batch))
    if memory == "pycuda":
        for name, module in _fake_pycuda(heap, context_stack).items():
            monkeypatch.setitem(sys.modules, name, module)
    else:
        _block_modules(monkeypatch, PYCUDA_MODULES)
        for name, module in _fake_cuda_python(heap).items():
            monkeypatch.setitem(sys.modules, name, module)

    backend = pb.create_backend("tensorrt", str(engine_file))
    assert backend.name == "tensorrt" and memory in backend.description
    np.testing.assert_allclose(backend.infer(_input(0.25)), [[0.25, -0.25]], rtol=1e-6)
    np.testing.assert_allclose(backend.infer(_input(-0.5)), [[-0.5, 0.5]], rtol=1e-6)
    assert context_stack == []  # every push was popped
    assert len(heap.blocks) == 2
    backend.close()
    assert heap.blocks == {}


def test_tensorrt_without_cuda_memory_module(monkeypatch, engine_file):
    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(_DeviceHeap()))
    _block_modules(monkeypatch, PYCUDA_MODULES + CUDA_PYTHON_MODULES)
    with pytest.raises(ImportError, match="pycuda or cuda-python"):
        pb.create_backend("tensorrt", str(engine_file))


def test_tensorrt_10_is_rejected(monkeypatch, engine_file):
    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(_DeviceHeap(), with_binding_api=False))
    with pytest.raises(RuntimeError, match="8.x"):
        pb.create_backend("tensorrt", str(engine_file))


def test_tensorrt_bad_engine(monkeypatch, tmp_path):
    heap = _DeviceHeap()
    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(heap))
    for name, module in _fake_pycuda(heap, []).items():
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "corrupt.engine"
    path.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="deserialize"):
        pb.create_backend("tensorrt", str(path))
