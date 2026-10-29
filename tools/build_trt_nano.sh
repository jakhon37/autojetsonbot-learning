#!/usr/bin/env bash
# Build and time a TensorRT FP16 engine of the exported policy on the Jetson Nano
# (JetPack 4.6.x: TensorRT 8.0.1 on 4.6, 8.2.1 on 4.6.1+).
#
# UNTESTED ON HARDWARE: no Jetson was available when this was written. The trtexec options are
# those of TensorRT 8.0 / 8.2 (--workspace in MiB; 8.4+ replaced it with --memPoolSize).
#
#   bash tools/build_trt_nano.sh deploy/bc/policy.onnx
#   # -> deploy/bc/policy_fp16.engine, policy_fp16.timing.txt, policy_fp32.timing.txt, build log
#
# Run it on the Nano itself: an engine only runs on the GPU model and TensorRT version that built
# it. Export the ONNX file with tools/export_onnx.py (opset 13 / IR 7, which TensorRT 8.0-8.2
# parses; the input "image" [1,3,224,224] is static, so no shape flags are needed). With --fp16
# the layers run in half precision but the "image" and "action" bindings stay float32, which is
# what policy_runner's tensorrt backend feeds and reads.
#
# Steps:
#   1. FP16 engine build (log kept beside the engine)
#   2. FP16 latency over 1000 iterations, plus an FP32 run of the same ONNX for reference
#   3. policy_meta.json copied beside the engine; policy_runner checks it before loading
#
# Before timing, pin the clocks, otherwise DVFS dominates the numbers:
#   sudo nvpmodel -m 0 && sudo jetson_clocks
# Then on the robot:
#   ros2 run jbl_nodes policy_runner --ros-args -p backend:=tensorrt -p model_path:=<engine>
#
# Settings (environment variables): TRTEXEC [/usr/src/tensorrt/bin/trtexec], WORKSPACE_MB [1024],
# ITERATIONS [1000], SKIP_FP32 [0].
set -euo pipefail

ONNX=${1:?usage: build_trt_nano.sh <policy.onnx> [engine path]}
ENGINE=${2:-${ONNX%.onnx}_fp16.engine}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
WORKSPACE_MB=${WORKSPACE_MB:-1024}
ITERATIONS=${ITERATIONS:-1000}
SKIP_FP32=${SKIP_FP32:-0}
STEM=${ENGINE%.engine}
META="$(dirname "$ONNX")/policy_meta.json"

[ -f "$ONNX" ] || { echo "build_trt_nano.sh: no such file: $ONNX" >&2; exit 2; }
[ -x "$TRTEXEC" ] || { echo "build_trt_nano.sh: trtexec not found at $TRTEXEC (set TRTEXEC)" >&2; exit 2; }
if [ -r /etc/nv_tegra_release ]; then
  echo "L4T: $(head -n 1 /etc/nv_tegra_release)"
else
  echo "warning: /etc/nv_tegra_release not found; this does not look like a Jetson" >&2
fi
dpkg-query -W -f='TensorRT: ${Version}\n' libnvinfer8 2>/dev/null || true
# policy_runner's tensorrt backend needs the TensorRT Python bindings and pycuda in the Python
# it runs with (JetPack installs neither pycuda nor, for Python 3.8, tensorrt): README, "Deployment".
if ! python3 -c 'import tensorrt, pycuda.driver' >/dev/null 2>&1; then
  echo "warning: python3 cannot import tensorrt and pycuda.driver here; policy_runner backend:=tensorrt" \
    "will not start in this Python (README, Deployment: pip3 install 'pycuda==2022.1')" >&2
fi

echo "== 1. FP16 engine: $ONNX -> $ENGINE"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --workspace="$WORKSPACE_MB" \
  --buildOnly \
  --verbose > "$STEM.build.log" 2>&1 || { tail -n 40 "$STEM.build.log" >&2; exit 1; }
echo "engine written ($(du -h "$ENGINE" | cut -f1)); build log: $STEM.build.log"

echo "== 2. FP16 timing ($ITERATIONS iterations, batch 1)"
"$TRTEXEC" \
  --loadEngine="$ENGINE" \
  --iterations="$ITERATIONS" \
  --warmUp=2000 \
  --duration=0 \
  --useSpinWait \
  --avgRuns=100 \
  --percentile=99 > "$STEM.timing.txt" 2>&1
grep -E "Latency:|GPU Compute Time:|Throughput:" "$STEM.timing.txt" || tail -n 20 "$STEM.timing.txt"

if [ "$SKIP_FP32" != 1 ]; then
  FP32_TIMING="$(dirname "$ENGINE")/$(basename "${ONNX%.onnx}")_fp32.timing.txt"
  echo "== 2b. FP32 reference timing (engine not saved)"
  "$TRTEXEC" \
    --onnx="$ONNX" \
    --workspace="$WORKSPACE_MB" \
    --iterations="$ITERATIONS" \
    --warmUp=2000 \
    --duration=0 \
    --useSpinWait \
    --avgRuns=100 \
    --percentile=99 > "$FP32_TIMING" 2>&1
  grep -E "Latency:|GPU Compute Time:|Throughput:" "$FP32_TIMING" || tail -n 20 "$FP32_TIMING"
fi

echo "== 3. policy_meta.json"
if [ -f "$META" ]; then
  if [ "$(cd "$(dirname "$META")" && pwd)" != "$(cd "$(dirname "$ENGINE")" && pwd)" ]; then
    cp "$META" "$(dirname "$ENGINE")/policy_meta.json"
  fi
  echo "policy_meta.json beside $ENGINE"
else
  echo "warning: no $META; policy_runner will load the engine unchecked" >&2
fi
echo "done. Compare FP16 outputs with onnxruntime on recorded frames before trusting the engine."
