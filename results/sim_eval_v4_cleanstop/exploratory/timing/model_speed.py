"""Do the BC models differ in cost? Subnormal weights, and interleaved single-thread timing.

    python model_speed.py FRAMES.npz MODEL.onnx [MODEL.onnx ...]

Per model: the share of subnormal float32 values among its weights, then 300 inferences each,
interleaved round-robin on the same recorded frames (decode + jbl_nodes preprocess outside the
timed part), with one intra-op thread so the pool plays no part. Prints p50 / p90 / p99 in ms.
Run with jbl_nodes on PYTHONPATH (ros2_ws/src/jbl_nodes).
"""

import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def subnormal_share(path):
    model = onnx.load(path)
    total = sub = 0
    tiny = np.finfo(np.float32).tiny
    for init in model.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype != np.float32:
            continue
        nz = a[a != 0]
        total += a.size
        sub += int(np.sum(np.abs(nz) < tiny))
    return sub, total


def main(frames_path, models):
    import cv2

    from jbl_nodes.preprocess import preprocess

    z = np.load(frames_path)
    data, offsets = z["data"], z["offsets"]
    xs = []
    for i in range(0, len(offsets) - 1, 2):
        img = cv2.imdecode(data[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR)
        xs.append(preprocess(img, is_bgr=True))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sessions = []
    for m in models:
        sub, total = subnormal_share(m)
        print("%s: %d subnormal of %d float32 weights" % (m, sub, total))
        s = ort.InferenceSession(m, sess_options=so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        for _ in range(5):
            s.run(None, {name: xs[0]})
        sessions.append((m, s, name, []))
    for k in range(300):
        x = xs[k % len(xs)]
        for m, s, name, times in sessions:
            t0 = time.perf_counter()
            s.run(None, {name: x})
            times.append((time.perf_counter() - t0) * 1e3)
    for m, s, name, times in sessions:
        p50, p90, p99 = np.percentile(times, [50, 90, 99])
        print("%s: 1 thread, %d runs: p50 %.2f  p90 %.2f  p99 %.2f ms" % (m, len(times), p50, p90, p99))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
