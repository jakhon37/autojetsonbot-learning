#!/usr/bin/env python3
"""policy_runner's per-tick work on recorded frames, paced like its ROS timer.

    python3 harness.py --model M.onnx --frames F.npz --threads default|N [--spin default|0|1]
                       [--n 400] [--warmup 20] [--rate 10] [--label L] [--out results.jsonl]

Each tick does what PolicyRunner._tick does, with the repo's own code: decode the recorded
CompressedImage bytes (jbl_nodes.imgconv.compressed_to_bgr), preprocess
(jbl_nodes.preprocess.preprocess), infer through the repo's ONNX Runtime backend
(jbl_nodes.policy_backends.create_backend), map the action (action_to_twist). The backend is
warmed up with two zero inputs as the node does, then --warmup ticks are discarded and --n
ticks measured. Ticks follow an rcl timer at --rate Hz: a tick that overruns its period is
followed at once by the next one, and missed periods are skipped (rcl_timer_call); --rate 0
runs back to back.

--threads default leaves the session options as policy_runner creates them; N sets
SessionOptions.intra_op_num_threads = N, either through the backend's own argument when it has
one (--via param) or by wrapping onnxruntime.InferenceSession (--via patch).

One JSON line per run: the CPU view of the process, the threads the session created and their
CPU affinity, preprocess / infer / total p50, p90, p99, the share of ticks over the 100 ms
period, and the CPU time the process used (cores = CPU seconds / wall seconds).
"""

import argparse
import inspect
import json
import os
import resource
import sys
import time

import numpy as np


def cpu_view():
    view = {"os_cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0))}
    for path in (
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
        "/sys/fs/cgroup/cpuset.cpus.effective",
    ):
        try:
            with open(path) as f:
                view[path] = f.read().strip()
        except OSError:
            pass
    with open("/proc/cpuinfo") as f:
        view["cpuinfo_processors"] = sum(1 for line in f if line.startswith("processor"))
    return view


def tids():
    return set(int(t) for t in os.listdir("/proc/self/task"))


def thread_cpu_s(tid):
    try:
        with open("/proc/self/task/%d/stat" % tid) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except OSError:
        return None


def thread_info(tid):
    try:
        affinity = sorted(os.sched_getaffinity(tid))
    except OSError:
        affinity = None
    try:
        with open("/proc/self/task/%d/comm" % tid) as f:
            name = f.read().strip()
    except OSError:
        name = "?"
    return {"tid": tid, "name": name, "affinity": affinity}


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(a, [50.0, 90.0, 99.0])
    return {"p50": round(float(p50), 2), "p90": round(float(p90), 2), "p99": round(float(p99), 2),
            "mean": round(float(a.mean()), 2), "max": round(float(a.max()), 2)}


class Frame(object):
    """Stands in for sensor_msgs/CompressedImage: imgconv reads .format and .data."""

    def __init__(self, fmt, data):
        self.format = fmt
        self.data = data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--threads", default="default")
    ap.add_argument("--spin", default="default", choices=("default", "0", "1"))
    ap.add_argument("--via", default="auto", choices=("auto", "param", "patch"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()

    import onnxruntime as ort

    import jbl_nodes
    from jbl_nodes import policy_backends
    from jbl_nodes.imgconv import compressed_to_bgr
    from jbl_nodes.preprocess import action_to_twist, preprocess

    threads = None if args.threads == "default" else int(args.threads)
    has_param = "intra_op_threads" in inspect.signature(policy_backends.create_backend).parameters
    via = args.via if args.via != "auto" else ("param" if has_param else "patch")
    kwargs = {}
    if via == "param":
        if not has_param:
            sys.exit("this jbl_nodes has no intra_op_threads argument")
        kwargs["intra_op_threads"] = threads or 0
    if (via == "patch" and threads is not None) or args.spin != "default":
        original = ort.InferenceSession

        def session(path, sess_options=None, providers=None, provider_options=None, **kw):
            so = sess_options or ort.SessionOptions()
            if via == "patch" and threads is not None:
                so.intra_op_num_threads = threads
            if args.spin != "default":
                so.add_session_config_entry("session.intra_op.allow_spinning", args.spin)
            return original(path, sess_options=so, providers=providers, provider_options=provider_options, **kw)

        ort.InferenceSession = session

    z = np.load(args.frames)
    data, offsets, fmt = z["data"], z["offsets"], str(z["format"])
    frames = [Frame(fmt, data[offsets[i]:offsets[i + 1]].tobytes()) for i in range(len(offsets) - 1)]

    before = tids()
    backend = policy_backends.create_backend("onnxruntime", args.model, **kwargs)
    created = sorted(tids() - before)
    x0 = np.zeros(policy_backends.INPUT_SHAPE, dtype=np.float32)
    for _ in range(2):  # PolicyRunner.WARMUP_RUNS
        backend.infer(x0)

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    pre, inf, tot = [], [], []
    next_call = time.perf_counter()
    cpu0 = wall0 = ru0 = None
    thread_cpu0 = {}
    for i in range(args.warmup + args.n):
        if i == args.warmup:
            wall0, cpu0, ru0 = time.perf_counter(), time.process_time(), resource.getrusage(resource.RUSAGE_SELF)
            thread_cpu0 = {t: thread_cpu_s(t) for t in [os.getpid()] + created}
        if period:
            now = time.perf_counter()
            if now < next_call:
                time.sleep(next_call - now)
            now = time.perf_counter()
            next_call += period  # rcl_timer_call: advance by one period from the scheduled time,
            if next_call < now:  # and skip the periods already missed
                next_call += period * (1 + int((now - next_call) / period))
        msg = frames[i % len(frames)]
        t0 = time.perf_counter()
        frame = compressed_to_bgr(msg)
        x = preprocess(frame, is_bgr=True)
        t1 = time.perf_counter()
        action = backend.infer(x)
        t2 = time.perf_counter()
        action_to_twist(action)
        t3 = time.perf_counter()
        if i >= args.warmup:
            pre.append((t1 - t0) * 1e3)
            inf.append((t2 - t1) * 1e3)
            tot.append((t3 - t0) * 1e3)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    thread_cpu = {}
    for t, c0 in thread_cpu0.items():
        c1 = thread_cpu_s(t)
        thread_cpu[str(t)] = None if c0 is None or c1 is None else round(c1 - c0, 2)

    result = {
        "label": args.label,
        "model": args.model,
        "frames": args.frames,
        "threads": args.threads,
        "spin": args.spin,
        "via": via if (threads is not None or via == "param") else "default options",
        "rate_hz": args.rate,
        "n": len(inf),
        "ort": ort.__version__,
        "backend": backend.description,
        "jbl_nodes": os.path.dirname(jbl_nodes.__file__),
        "cpu_view": cpu_view(),
        "session_threads_created": len(created),
        "session_threads": [thread_info(t) for t in created],
        "main_thread_affinity": sorted(os.sched_getaffinity(0)),
        "preprocess_ms": stats(pre),
        "infer_ms": stats(inf),
        "total_ms": stats(tot),
        "ticks_over_100ms": int(sum(1 for t in tot if t > 100.0)),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "cores_used": round(cpu / wall, 3),
        "thread_cpu_s": thread_cpu,
        "nvcsw": ru.ru_nvcsw - ru0.ru_nvcsw,
        "nivcsw": ru.ru_nivcsw - ru0.ru_nivcsw,
        "loadavg_end": open("/proc/loadavg").read().strip(),
    }
    backend.close()
    line = json.dumps(result)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
