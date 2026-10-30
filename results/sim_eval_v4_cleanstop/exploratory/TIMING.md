# Timing: why v4's policy inference was 2–3 times slower than v3's

> **EXPLORATORY.** Post hoc: decided after the v4 run and its review, not pre-registered. It
> changes no v4 result, prediction verdict or comparison in [SUMMARY.md](../SUMMARY.md); the v4
> data and verdicts stay as they are. **SIMULATION ONLY:** every number comes from `fake_robot`
> and `ros:jazzy-ros-base` containers on the same 4-core x86 VM that ran v3 and v4, measured on
> 29 October 2026 between 21:00 and 22:10 UTC, with repeats after the review between 23:27 and
> 23:47 UTC. Nothing here was measured on autoJetsonBot or on a Jetson.

## Question

In v4 (SUMMARY.md, exploratory item 3), `policy_runner` (ONNX Runtime 1.31.1 on CPU, the BC
ResNet18 at 224 × 224, 10 Hz) inferred in p50 60–69 ms and p99 116–123 ms, against p50 22–34 ms
and p99 48–67 ms in v3. A fallback zero (a stale-frame (0, 0) request) fell in the last second
of 37 of the 130 policy trials, against 2 of 200 in v3. The candidates were:

- the cpuset (v4 pinned its simulation containers with `docker run --cpuset-cpus 0-2`, v3 did
  not), with the way ONNX Runtime sizes its thread pool;
- the other work on the host during parts of v4 (RUN_LOG.md, "Shared host");
- the models (`bc_v4` against v3's `bc_v2`);
- the preprocessing.

## Answer

- **The cause of the slow inference is the cpuset combined with ONNX Runtime's default thread
  pool.** Inside `--cpuset-cpus 0-2`, ONNX Runtime 1.31.1 still sizes its intra-op pool from
  the host's 4 cores: 4 threads, 3 of them workers pinned to CPUs 1, 2 and 3. The pin to CPU 3
  fails, and that error is in the launch log of every v4 segment and of no v3 segment. The
  workers spin after each inference, so the policy process keeps about 2 cores (1.9–2.5) busy
  at 10 Hz, on 3 cores it shares with the simulator's nodes.
- **Reproduced.** In a controlled benchmark of `policy_runner`'s own decode, preprocess and
  ONNX Runtime code on recorded frames, with the simulator's nodes as background on an idle
  host, only the cpuset was changed. Inference went from p50 / p99 23.0 / 51.9 ms to
  58.2 / 111.2 ms (repeat: 26.8 / 50.4 → 61.8 / 97.9; a third run in the cpuset, after the
  review: 58.2 / 111.0). That is v3's level (21.7–33.6 / 47.9–67.1 ms) and v4's (60.5–69.4 /
  115.7–122.6 ms).
- **The real node agrees on the timing.** In the simulator, in two runs of 10 `noise` episodes
  in the cpuset, v4's setting (the default pool) gave 62.7 / 111.8 and 64.2 / 109.4 ms and an
  image age p99 of 186.9 and 185.0 ms, as in v4; 2 threads gave 22.6 / 40.7 and
  27.2 / 47.2 ms (section 5).
- **The other candidates are not needed to explain the inference time.** The benchmark host was
  idle and no servo node ran. The four BC models are one architecture of 11,303,554 parameters
  and cost the same. The preprocessing code is unchanged, and its v4 increase is reproduced by
  the cpuset alone.
- **The fallback zeros and the validity flags are explained only in part.** With v4's default
  in the cpuset, 1.20 % and 0.95 % of the real node's requests were fallback zeros (none
  unpinned). v4's own `noise` session had 2.41 %, and none of the check's 20 episodes had a
  fallback zero in its last second, against 10 of v4's 30 (Fisher's exact test, one-sided,
  p = 0.003). Those last-second zeros raised v4's 21 validity flags; what adds the rest was not
  found (sections 5 and 7).
- **What restores v3-like latency in the 3-core cpuset: 2 intra-op threads** (the calling
  thread and one unpinned worker). With the simulator's nodes: 27.0 / 43.9 ms (repeats
  25.6 / 45.6, 27.4 / 41.7 and 24.6 / 39.2), no tick over 100 ms, about 1 core. Under load it
  had the lowest p99 of the sizes tried; 3 threads matched it at p50 in one of three runs.
  - `policy_runner` now has `intra_op_threads` (default 0 = unchanged).
  - `config/sim.yaml` sets 2.
  - Setting 0 reproduces v4.

## 1. The session bags: where the time went

Source: every `/policy/latency` message of the v3 and v4 evaluation sessions (one per inference
on a fresh frame), read by [timing/timing_bags.py](timing/timing_bags.py)
([timing/timing_bags.csv](timing/timing_bags.csv)).

- **The fields.** `preprocess_ms` is JPEG decode + resize + normalise; `infer_ms` is
  `backend.infer`; `total_ms` runs from tick start to publish; `image_age_ms` is taken at tick
  start.
- **Percentiles** are numpy's linear ones, as `tools/analyze_trials.py` computes them, so the
  v4 rows repeat SUMMARY.md's timing table.
- **"Consecutive inferences > 150 ms apart"** counts gaps between `/policy/latency` messages
  inside one policy episode, from the bag's integer-nanosecond receive times.
- **"Fallback zeros"** are the policy arm's exact (0, 0) requests on `/cmd_vel_in` during its
  own episodes, over all its requests there, not only the last second.

| Run | Session (policy arm) | Condition | Inferences | Preprocess p50 / p99, ms | Infer p50 / p90 / p99, ms | Total p50 / p99, ms | Image age p50 / p99, ms | Consecutive inferences > 150 ms apart | Fallback zeros / policy ticks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3 | bc_v2 | clean_ref | 1571 | 4.5 / 9.9 | 21.7 / 33.1 / 47.9 | 28.3 / 56.5 | 41.7 / 75.9 | 1 of 1561 | 0 / 1571 (0.00 %) |
| v3 | bc_v2 | noise | 4904 | 5.0 / 12.4 | 33.6 / 46.9 / 67.1 | 40.7 / 75.8 | 53.0 / 93.3 | 4 of 4849 | 2 / 4882 (0.04 %) |
| v3 | bc_v2 | wide | 6387 | 5.1 / 12.1 | 26.0 / 45.3 / 66.8 | 33.4 / 74.8 | 40.3 / 100.3 | 46 of 6347 | 4 / 6381 (0.06 %) |
| v3 | bc_v2 | hfov62 | 9021 | 5.0 / 11.5 | 31.8 / 45.6 / 64.7 | 38.9 / 72.7 | 46.8 / 91.7 | 26 of 8969 | 6 / 9006 (0.07 %) |
| v3 | bc_v2sw | clean_ref | 1513 | 4.7 / 9.4 | 22.6 / 35.8 / 62.9 | 29.5 / 67.8 | 45.7 / 88.8 | 5 of 1503 | 0 / 1513 (0.00 %) |
| v3 | bc_v2sw | noise | 4671 | 5.3 / 13.3 | 27.0 / 38.5 / 65.4 | 34.8 / 75.1 | 53.3 / 94.3 | 18 of 4641 | 4 / 4675 (0.09 %) |
| v3 | bc_v2sw | wide | 6522 | 5.8 / 15.1 | 29.7 / 43.1 / 61.9 | 37.7 / 73.4 | 50.9 / 94.0 | 13 of 6485 | 2 / 6517 (0.03 %) |
| v3 | bc_v2sw | hfov62 | 9002 | 5.4 / 13.1 | 27.8 / 39.2 / 61.5 | 35.7 / 73.1 | 52.4 / 93.0 | 27 of 8972 | 7 / 9009 (0.08 %) |
| v4 | 1: servo + bc_v4 | clean_ref | 2058 | 8.7 / 20.8 | 69.4 / 93.9 / 122.6 | 80.9 / 139.4 | 56.3 / 188.0 | 33 of 2036 | 21 / 2067 (1.02 %) |
| v4 | 1: servo + bc_v4 | noise | 6421 | 9.1 / 21.2 | 68.0 / 90.8 / 116.6 | 79.8 / 133.4 | 71.0 / 192.1 | 141 of 6366 | 158 / 6554 (2.41 %) |
| v4 | 1: servo + bc_v4 | wide | 7964 | 8.9 / 22.7 | 66.2 / 90.7 / 119.7 | 78.3 / 136.2 | 64.1 / 192.1 | 210 of 7915 | 237 / 8183 (2.90 %) |
| v4 | 1: servo + bc_v4 | hfov62 | 8781 | 8.6 / 21.2 | 60.5 / 86.5 / 116.6 | 71.8 / 134.2 | 71.3 / 192.4 | 185 of 8731 | 209 / 8970 (2.33 %) |
| v4 | 2: bc_v4_62 | hfov62 | 6362 | 7.4 / 21.0 | 62.2 / 86.8 / 115.7 | 72.6 / 132.2 | 68.5 / 193.8 | 104 of 6309 | 121 / 6460 (1.87 %) |

- **Most of the regression is inference.**
  - Inference p50 rose by 29–48 ms, to 1.9–3.2 times v3's in the same condition.
  - Preprocessing p50 rose by 2–4 ms (4.5–5.8 → 7.4–9.1 ms).
  - The rest of the tick (`total` − preprocess − infer) stayed at about 1 ms.
- **The image age followed.**
  - Its p99 rose from 76–100 ms to 188–194 ms, just under the 200 ms limit.
  - The mechanism: after a tick overruns 100 ms, the next one is due at once, and rclpy's
    executor serves a ready timer before a ready subscription. When the newer frame arrived
    during the slow tick, the next tick runs on the old frame, now older by the slow tick's
    length.
  - Fallback zeros went from 0.00–0.09 % of the policies' requests to 1.0–2.9 %.
- **The same code ran in both.**
  - `policy_runner.py`, `policy_backends.py`, `preprocess.py` and `imgconv.py` are identical
    between v3's commit (13afb1b) and v4's (b72ba83).
  - Both runs installed ONNX Runtime 1.31.1 in the same image digest.
  - `policy_runner` created its session with no session options in both.

## 2. The models are the same network

Source: [timing/model_info.py](timing/model_info.py)
([timing/model_info.jsonl](timing/model_info.jsonl)). The timing check is
[timing/model_speed.py](timing/model_speed.py) ([timing/model_speed.txt](timing/model_speed.txt)).

| Arm | Run | sha256 (start) | ONNX nodes | Op types | Initialisers | Parameters | Input / output | Opset | Bytes | 1 thread, p50 / p99 |
|---|---|---|---:|---|---:|---:|---|---:|---:|---:|
| `bc_v2` | v3 | `f6e0b754…` | 51 | 20 Conv, 18 Relu, 8 Add, 2 Gemm, MaxPool, GlobalAveragePool, Flatten | 44 | 11,303,554 | `image` [1, 3, 224, 224] / `action` [1, 2] | 13 | 45,226,792 | 36.0 / 61.8 ms |
| `bc_v2sw` | v3 | `e018fd40…` | 51 | the same | 44 | 11,303,554 | the same | 13 | 45,226,792 | 35.7 / 61.7 ms |
| `bc_v4` | v4 | `ed36c9b6…` | 51 | the same | 44 | 11,303,554 | the same | 13 | 45,226,792 | 35.9 / 58.7 ms |
| `bc_v4_62` | v4 | `ea29531c…` | 51 | the same | 44 | 11,303,554 | the same | 13 | 45,226,792 | 35.8 / 59.4 ms |

- **One architecture.** The op-type sequence and every initialiser's shape are identical in
  all four; only the weights differ.
- **No slow weights.** None of the four has a subnormal weight.
- **The same cost.** With one intra-op thread, 300 inferences each on the same recorded frames,
  interleaved round-robin (on the host, ONNX Runtime 1.31.0), all four took 35.7–36.0 ms p50.
- **Also under the v3 and v4 setups** (section 4, simulator background).
  - With no cpuset, `bc_v2` on its own v3 frames ran as fast as `bc_v4`: 21.7 / 48.6 ms
    against 23.0 / 51.9 and 26.8 / 50.4 ms.
  - With the cpuset it slowed the same way: 46.5 / 88.3 ms against 58.2 / 111.2 and
    61.8 / 97.9 ms. One run of `bc_v2` cannot separate its 20 % lower p50 from the run-to-run
    spread of an oversubscribed container, and the single-thread timing above shows no
    difference in cost.

## 3. What ONNX Runtime's default resolves to inside the cpuset container

`policy_runner` creates its session with no session options, so ONNX Runtime picks the
intra-op thread count. [timing/harness.py](timing/harness.py) recorded which threads the
session created and their CPU affinity.

| Container | What the process sees | Session's intra-op pool, default options |
|---|---|---|
| no cpuset (v3) | `os.cpu_count()` 4, affinity {0,1,2,3}, `/proc/cpuinfo` 4 processors | 4 threads: the calling thread + 3 workers pinned to CPU 1, 2 and 3 |
| `--cpuset-cpus 0-2` (v4) | `os.cpu_count()` 4, affinity {0,1,2}, cgroup cpuset `0-2`, `/proc/cpuinfo` 4 processors | **also 4 threads**: workers pinned to CPU 1 and 2; the third worker's pin to CPU 3 fails (outside the cpuset), and it keeps {0,1,2} |

- **The failed pin is logged.** ONNX Runtime prints `pthread_setaffinity_np failed for thread:
  …, index: 2, mask: {3, }, error code: 22 error msg: Invalid argument. Specify the number of
  threads explicitly so the affinity is not set.`
  - The v4 sessions: the line is in `policy_runner`'s start-up output in the launch log of all
    5 v4 segments (`sim_work/v4*/eval_v2/jazzy/*/logs/seg_*.launch.log`, not in git), and in
    none of the 8 v3 segments.
  - Here: every default-options run in the cpuset container printed it (7 of 7), and no other
    run did.
- **An explicit count is not sized from the host and not pinned.** With
  `intra_op_num_threads = N` the session made N − 1 workers, each with the container's whole
  affinity.
- **The workers spin.** ONNX Runtime's default is `session.intra_op.allow_spinning = 1`: after
  each inference a worker busy-waits before it sleeps.
  - At 10 Hz each worker used 50–75 % of a core (per-thread CPU time in
    [timing/results.jsonl](timing/results.jsonl)), for an inference of 15–65 ms.
  - So with the default pool the policy process used 1.9–2.5 cores.
  - With spinning off, the same pool used 0.5 cores (section 4).

## 4. Controlled benchmark

**Method.**

- **Image and code.**
  - Base image: `mirror.gcr.io/library/ros:jazzy-ros-base`, the digest v3 and v4 used
    (`sha256:8eed0a46…b617`).
  - Prepared as the simulation prepares its container: `scripts/sim_e2e.sh --inner setup`
    builds the workspace and `--inner onnxruntime` installs ONNX Runtime 1.31.1.
  - Built at commit 4f0723a, whose `jbl_nodes` policy code is v3's and v4's.
  - The container was committed as a temporary image, so every configuration started from the
    same files.
- **Containers.** Two from that image, started as `scripts/sim_eval.sh` starts its container
  (`--network host`, their own `ROS_DOMAIN_ID`): "no cpuset", as v3 ran, and "cpuset 0-2"
  (`--cpuset-cpus 0-2`), as v4 ran.
- **Harness** ([timing/harness.py](timing/harness.py), with the container's `python3` and the
  built workspace).
  - Each tick does what `PolicyRunner._tick` does, with the repo's own functions: it decodes a
    recorded `CompressedImage` (`imgconv.compressed_to_bgr`), runs `preprocess.preprocess`,
    then `backend.infer` on the backend from `policy_backends.create_backend("onnxruntime", …)`.
  - It times the same intervals as the node's `preprocess_ms` and `infer_ms`.
  - The backend gets the node's two zero-input warm-up runs, 20 more ticks are discarded, then
    400 ticks are measured.
  - Ticks follow an rcl timer at 10 Hz: a tick that overruns is followed at once by the next
    one, and missed periods are skipped.
- **Frames.** 600 consecutive camera frames (640 × 480 JPEG, 34 kB on average) recorded during
  `bc_v4`'s episodes of the v4 `noise` session, used in turn. `bc_v2` got 600 frames from its
  own v3 `noise` session.
- **Threads.**
  - "default" leaves the session exactly as `policy_runner` creates it (no session options).
  - N sets `SessionOptions.intra_op_num_threads = N` by wrapping
    `onnxruntime.InferenceSession`; the backend's own code path is otherwise unchanged.
  - "no spin" also sets `session.intra_op.allow_spinning = 0`.
- **Background**, in the same container and cpuset:
  - none;
  - the simulator's own nodes, `sim_dock.launch.py method:=none` (`fake_robot`,
    `safety_gate`, `dock_judge`, `trial_logger` and the bag recorder: a session without a
    controller, the robot standing still), which used 0.47–0.54 cores on their own;
  - or a busy loop of 2 processes at 80 % duty ([timing/busy.py](timing/busy.py)), which used
    1.55–1.57 cores on its own.
- **Idle host.**
  - Before every run, the CPU used on the host outside the two benchmark containers (root
    cgroup minus the containers' `cpuacct`, over 3 s) had to be below 0.25 cores, or the driver
    would wait.
  - It never had to wait: the largest value was 0.09 cores.
  - `docker ps` listed only the benchmark containers.
  - Every check, with `/proc/loadavg`, is in [timing/results.log](timing/results.log).
- **Repeats after the review** (23:28–23:32 UTC, `run_matrix.sh` with `PHASES=threads`): five
  more runs in the cpuset with the simulator's nodes, from an image rebuilt the same way from the
  working tree. First the default pool, to check the new image against the first one, then 3 and
  2 threads twice each, alternating. That image has the new parameter, so the harness passed
  `intra_op_threads` to `create_backend`; the session options are the same.

**Inference p50 / p99, ms** (`bc_v4`, 400 ticks at 10 Hz per cell). "default" is ONNX Runtime's
own pool: 4 threads, workers pinned. "4" is 4 threads set explicitly, so not pinned.

| Background | Container | default | 1 thread | 2 threads | 3 threads | 4 threads |
|---|---|---:|---:|---:|---:|---:|
| none | no cpuset | 15.3 / 44.9 | 37.5 / 60.1 | 22.0 / 38.4 | 17.8 / 31.2 | 14.7 / 26.5 |
| none | cpuset 0-2 | **34.5 / 64.5** | 36.5 / 61.2 | 22.4 / 37.8 | 18.7 / 35.8 | 35.6 / 53.1 |
| simulator nodes | no cpuset (v3) | **23.0 / 51.9** | 38.5 / 59.2 | 21.4 / 36.2 | 18.6 / 33.6 | 22.2 / 57.4 |
| simulator nodes | cpuset 0-2 (v4) | **58.2 / 111.2** | 41.9 / 69.0 | **27.0 / 43.9** | 37.1 / 58.6 | 48.5 / 131.2 |
| busy loop, 1.6 cores | no cpuset | 47.6 / 85.3 | 38.1 / 61.4 | 23.6 / 40.8 | 27.6 / 53.1 | — |
| busy loop, 1.6 cores | cpuset 0-2 | 64.9 / 94.9 | 38.9 / 60.9 | 31.8 / 55.9 | 52.9 / 85.9 | — |

Every run in full ([timing/make_tables.py](timing/make_tables.py) on
[timing/results.jsonl](timing/results.jsonl)). "Policy process, cores" is the harness process's
CPU time over the measured 40 s. The repeats ran after the busy-loop runs; those ending in `_r2`
and `_r3` ran after the review.

| Run | Background | Container | Model | Intra-op threads | Session threads' affinity | Preprocess p50 / p90 / p99, ms | Infer p50 / p90 / p99, ms | Total p50 / p90 / p99, ms | Ticks > 100 ms (of 400) | Policy process, cores |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| idle_free_tdefault | none | no cpuset | bc_v4 | default (4) | {1} {2} {3} | 5.2 / 7.4 / 9.6 | 15.3 / 20.1 / 44.9 | 21.4 / 26.7 / 52.4 | 0 | 1.91 |
| idle_free_t1 | none | no cpuset | bc_v4 | 1 | none | 5.1 / 7.0 / 8.1 | 37.5 / 53.3 / 60.1 | 43.1 / 59.5 / 67.4 | 0 | 0.46 |
| idle_free_t2 | none | no cpuset | bc_v4 | 2 | {0,1,2,3} | 5.1 / 7.2 / 9.8 | 22.0 / 30.4 / 38.4 | 27.6 / 37.3 / 46.9 | 0 | 0.94 |
| idle_free_t3 | none | no cpuset | bc_v4 | 3 | {0,1,2,3} {0,1,2,3} | 5.0 / 7.0 / 8.5 | 17.8 / 23.4 / 31.2 | 23.4 / 29.8 / 36.6 | 0 | 1.41 |
| idle_free_t4 | none | no cpuset | bc_v4 | 4 | {0,1,2,3} {0,1,2,3} {0,1,2,3} | 5.0 / 7.0 / 8.1 | 14.7 / 19.1 / 26.5 | 20.2 / 24.6 / 31.5 | 0 | 1.85 |
| idle_pin_tdefault | none | cpuset 0-2 | bc_v4 | default (4) | {1} {2} {0,1,2} | 4.8 / 6.6 / 9.4 | 34.5 / 40.2 / 64.5 | 39.5 / 45.9 / 70.5 | 0 | 2.24 |
| idle_pin_t1 | none | cpuset 0-2 | bc_v4 | 1 | none | 4.9 / 6.7 / 10.1 | 36.5 / 51.1 / 61.2 | 41.6 / 57.3 / 70.0 | 0 | 0.45 |
| idle_pin_t2 | none | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 5.0 / 7.0 / 9.4 | 22.4 / 29.6 / 37.8 | 27.9 / 36.3 / 44.2 | 0 | 0.95 |
| idle_pin_t3 | none | cpuset 0-2 | bc_v4 | 3 | {0,1,2} {0,1,2} | 5.1 / 7.1 / 9.6 | 18.7 / 23.9 / 35.8 | 24.0 / 30.1 / 44.7 | 0 | 1.44 |
| idle_pin_t4 | none | cpuset 0-2 | bc_v4 | 4 | {0,1,2} {0,1,2} {0,1,2} | 5.0 / 6.8 / 8.6 | 35.6 / 40.8 / 53.1 | 41.3 / 47.6 / 58.9 | 0 | 2.29 |
| sim_free_tdefault | simulator nodes | no cpuset | bc_v4 | default (4) | {1} {2} {3} | 5.4 / 7.9 / 11.5 | 23.0 / 35.6 / 51.9 | 29.5 / 42.8 / 59.1 | 0 | 2.08 |
| sim_free_t1 | simulator nodes | no cpuset | bc_v4 | 1 | none | 5.0 / 7.0 / 10.2 | 38.5 / 52.4 / 59.2 | 43.8 / 58.6 / 67.5 | 0 | 0.46 |
| sim_free_t2 | simulator nodes | no cpuset | bc_v4 | 2 | {0,1,2,3} | 5.1 / 7.0 / 8.3 | 21.4 / 29.2 / 36.2 | 26.8 / 36.1 / 42.3 | 0 | 0.92 |
| sim_free_t3 | simulator nodes | no cpuset | bc_v4 | 3 | {0,1,2,3} {0,1,2,3} | 5.0 / 6.7 / 9.8 | 18.6 / 25.3 / 33.6 | 23.8 / 31.1 / 42.5 | 0 | 1.41 |
| sim_free_t4 | simulator nodes | no cpuset | bc_v4 | 4 | {0,1,2,3} {0,1,2,3} {0,1,2,3} | 5.2 / 7.5 / 9.7 | 22.2 / 31.2 / 57.4 | 27.7 / 37.7 / 63.9 | 1 | 2.04 |
| sim_free_tdefault_bcv2 | simulator nodes | no cpuset | bc_v2 (v3 frames) | default (4) | {1} {2} {3} | 4.9 / 7.1 / 9.3 | 21.7 / 34.1 / 48.6 | 27.5 / 40.2 / 55.4 | 0 | 1.97 |
| sim_pin_tdefault | simulator nodes | cpuset 0-2 | bc_v4 | default (4) | {1} {2} {0,1,2} | 7.3 / 12.4 / 16.6 | 58.2 / 85.4 / 111.2 | 67.0 / 96.3 / 123.5 | 28 | 2.38 |
| sim_pin_t1 | simulator nodes | cpuset 0-2 | bc_v4 | 1 | none | 5.2 / 7.2 / 10.8 | 41.9 / 55.2 / 69.0 | 48.0 / 62.0 / 74.0 | 0 | 0.48 |
| sim_pin_t2 | simulator nodes | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 4.9 / 7.0 / 10.1 | 27.0 / 35.8 / 43.9 | 32.8 / 42.0 / 49.4 | 0 | 0.97 |
| sim_pin_t3 | simulator nodes | cpuset 0-2 | bc_v4 | 3 | {0,1,2} {0,1,2} | 5.1 / 6.8 / 9.0 | 37.1 / 48.3 / 58.6 | 42.7 / 54.0 / 65.3 | 0 | 1.73 |
| sim_pin_t4 | simulator nodes | cpuset 0-2 | bc_v4 | 4 | {0,1,2} {0,1,2} {0,1,2} | 5.1 / 13.6 / 19.2 | 48.5 / 92.8 / 131.2 | 54.4 / 104.5 / 147.6 | 48 | 2.34 |
| sim_pin_tdefault_bcv2 | simulator nodes | cpuset 0-2 | bc_v2 (v3 frames) | default (4) | {1} {2} {0,1,2} | 4.8 / 9.3 / 14.4 | 46.5 / 72.6 / 88.3 | 52.1 / 78.5 / 95.1 | 2 | 2.34 |
| sim_pin_tdefault_nospin | simulator nodes | cpuset 0-2 | bc_v4 | default (4), no spin | {1} {2} {0,1,2} | 4.8 / 6.6 / 9.5 | 23.0 / 29.1 / 35.0 | 28.2 / 35.1 / 42.4 | 0 | 0.52 |
| busy_free_tdefault | busy loop 1.6 cores | no cpuset | bc_v4 | default (4) | {1} {2} {3} | 6.2 / 10.5 / 14.0 | 47.6 / 67.0 / 85.3 | 55.5 / 76.0 / 97.0 | 2 | 2.47 |
| busy_free_t1 | busy loop 1.6 cores | no cpuset | bc_v4 | 1 | none | 5.0 / 7.0 / 10.8 | 38.1 / 52.2 / 61.4 | 43.6 / 59.3 / 66.4 | 0 | 0.46 |
| busy_free_t2 | busy loop 1.6 cores | no cpuset | bc_v4 | 2 | {0,1,2,3} | 5.2 / 7.0 / 11.0 | 23.6 / 30.4 / 40.8 | 29.4 / 37.1 / 50.0 | 0 | 0.96 |
| busy_free_t3 | busy loop 1.6 cores | no cpuset | bc_v4 | 3 | {0,1,2,3} {0,1,2,3} | 5.2 / 7.9 / 10.9 | 27.6 / 39.1 / 53.1 | 33.3 / 45.3 / 61.2 | 0 | 1.59 |
| busy_pin_tdefault | busy loop 1.6 cores | cpuset 0-2 | bc_v4 | default (4) | {1} {2} {0,1,2} | 8.4 / 12.5 / 16.9 | 64.9 / 80.1 / 94.9 | 71.4 / 89.5 / 110.9 | 8 | 2.02 |
| busy_pin_t1 | busy loop 1.6 cores | cpuset 0-2 | bc_v4 | 1 | none | 5.2 / 7.0 / 8.8 | 38.9 / 53.7 / 60.9 | 44.5 / 60.6 / 68.6 | 0 | 0.48 |
| busy_pin_t2 | busy loop 1.6 cores | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 5.0 / 7.2 / 13.1 | 31.8 / 42.8 / 55.9 | 37.6 / 49.3 / 62.2 | 0 | 1.01 |
| busy_pin_t3 | busy loop 1.6 cores | cpuset 0-2 | bc_v4 | 3 | {0,1,2} {0,1,2} | 7.1 / 11.6 / 16.3 | 52.9 / 72.0 / 85.9 | 61.1 / 80.8 / 94.0 | 0 | 1.77 |
| rep_sim_free_tdefault | simulator nodes (repeat) | no cpuset | bc_v4 | default (4) | {1} {2} {3} | 5.3 / 7.3 / 11.2 | 26.8 / 36.9 / 50.4 | 33.2 / 43.2 / 56.7 | 0 | 2.08 |
| rep_sim_free_t2 | simulator nodes (repeat) | no cpuset | bc_v4 | 2 | {0,1,2,3} | 5.0 / 6.8 / 9.3 | 23.9 / 30.6 / 41.5 | 29.4 / 37.1 / 48.1 | 0 | 0.94 |
| rep_sim_pin_tdefault | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | default (4) | {1} {2} {0,1,2} | 5.6 / 13.1 / 18.0 | 61.8 / 81.8 / 97.9 | 68.9 / 93.3 / 113.5 | 21 | 2.40 |
| rep_sim_pin_t2 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 5.9 / 7.6 / 10.5 | 25.6 / 33.1 / 45.6 | 31.7 / 39.7 / 52.9 | 0 | 0.96 |
| rep_sim_pin_tdefault_r2 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | default (4) | {1} {2} {0,1,2} | 6.6 / 13.1 / 22.9 | 58.2 / 87.6 / 111.0 | 64.8 / 99.6 / 126.6 | 40 | 2.38 |
| rep_sim_pin_t3_r2 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | 3 | {0,1,2} {0,1,2} | 5.7 / 7.8 / 10.3 | 35.3 / 44.8 / 56.0 | 41.5 / 52.2 / 63.3 | 0 | 1.68 |
| rep_sim_pin_t2_r2 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 5.2 / 7.1 / 12.0 | 27.4 / 34.7 / 41.7 | 33.1 / 41.7 / 50.3 | 0 | 0.99 |
| rep_sim_pin_t3_r3 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | 3 | {0,1,2} {0,1,2} | 5.5 / 7.7 / 11.4 | 23.5 / 33.1 / 50.8 | 29.9 / 39.7 / 57.4 | 0 | 1.46 |
| rep_sim_pin_t2_r3 | simulator nodes (repeat) | cpuset 0-2 | bc_v4 | 2 | {0,1,2} | 5.0 / 7.0 / 8.9 | 24.6 / 33.5 / 39.2 | 29.9 / 39.9 / 46.6 | 0 | 0.95 |

What the benchmark shows:

- **The v3 and v4 sessions, reproduced.** With the simulator's nodes as background:
  - With no cpuset and the default pool, the harness timed what v3's sessions recorded:
    23.0 / 35.6 / 51.9 ms, repeat 26.8 / 36.9 / 50.4 (v3: 21.7–33.6 / 33.1–46.9 / 47.9–67.1).
  - The cpuset alone took it to what v4's sessions recorded: 58.2 / 85.4 / 111.2, repeats
    61.8 / 81.8 / 97.9 and 58.2 / 87.6 / 111.0 (v4: 60.5–69.4 / 86.5–93.9 / 115.7–122.6).
  - Preprocessing moved as in the sessions: p50 5.4 → 7.3 ms, p99 11.5 → 16.6 ms.
  - 5–10 % of the ticks overran 100 ms (none with no cpuset).
- **Even with nothing else running**, the default pool in the cpuset doubled the inference
  time: 15.3 → 34.5 ms p50.
  - 4 threads set explicitly, unpinned, did the same (35.6 ms), so the cause is 4 spinning
    threads on 3 cores, not the pins themselves.
  - Under load, the pins did not help either: 4 unpinned threads were as bad as the default
    (48.5 / 131.2 ms).
- **Spinning is the multiplier.** The default pool with spinning off ran at 23.0 / 35.0 ms in
  the cpuset with the simulator's nodes, at 0.52 cores instead of 2.38.
- **2 threads in the 3-core cpuset gave v3-like latency in every background**, with no tick
  over 100 ms and about 1 core (0.95–1.01) for the whole policy process:
  - idle 22.4 / 37.8 ms; with the simulator's nodes 27.0 / 43.9, repeats 25.6 / 45.6,
    27.4 / 41.7 and 24.6 / 39.2; with the 1.6-core busy loop 31.8 / 55.9.
  - Under either load it had the lowest p99 of the sizes tried. Idle, 3 threads was a little
    faster (18.7 / 35.8).
  - 3 threads varied with the simulator's nodes: 37.1 / 58.6 in the matrix, then 35.3 / 56.0
    and 23.5 / 50.8 in the repeats. At p50 it can match 2 threads there, but its p99 was higher
    in all three runs and it used 1.46–1.73 cores. Under the busy loop it was clearly worse
    (52.9 / 85.9).
  - 1 thread used the least CPU (0.45–0.48 cores) but cost about 40 ms p50 everywhere.
  - 2 is therefore the setting: the lowest p99 under load, and the least CPU of the sizes that
    ran at v3's speed.
  - With no cpuset, 2 threads was as fast as the default with the simulator's nodes
    (21.4 / 36.2 against 23.0 / 51.9) and better with the busy loop (23.6 / 40.8 against
    47.6 / 85.3).

## 5. End-to-end check with the real node

The benchmark leaves out the node's executor, its image subscription and a moving robot. So
`policy_runner` itself ran in the simulator ([timing/run_e2e.sh](timing/run_e2e.sh)):

- **Setup.** A fresh container per configuration, with the workspace rebuilt from the working
  tree, which has the new parameter.
- **Trials.** `scripts/sim_eval.sh`'s own trial loop (`--inner run`) on the `noise` condition:
  seed 201, randomisation on, test poses 0–9, 1 round, `bc_v4` alone, as v4's session 2 ran its
  policy arm alone.
- **The setting.** `intra_op_threads` came from an extra params file; 0 is v4's setting.
- **Two runs.** Run 1 (21:50–22:04 UTC) and, after the review, run 2 (23:32–23:47 UTC), each
  with the three configurations in the same order.
- **Analysis.** [timing/e2e_analyze.py](timing/e2e_analyze.py) ran `timing_bags.py`'s reader and
  `scripts/sim_eval_summary.py`'s validity check on each bag
  ([timing/e2e_summary.txt](timing/e2e_summary.txt), [timing/e2e.log](timing/e2e.log)).
  [timing/episode_zeros.py](timing/episode_zeros.py) set the fallback zeros against v4's own
  sessions ([timing/episode_zeros.txt](timing/episode_zeros.txt)).

| Run | Configuration | Inferences | Preprocess p50 / p99, ms | Infer p50 / p90 / p99, ms | Total p50 / p99, ms | Image age p99, ms | Consecutive inferences > 150 ms apart | Fallback zeros / policy ticks | Episodes with a fallback zero in the last 1 s | Warm-up, ms | ONNX Runtime's pin error | Container, cores | Load average at the end |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | cpuset 0-2, `intra_op_threads: 0` (v4) | 2155 | 8.3 / 20.0 | 62.7 / 84.4 / 111.8 | 73.2 / 127.4 | 186.9 | 20 of 2133 | 26 / 2169 (1.20 %) | 0 of 10 | 62.9 | logged | 2.48 | 7.94 |
| 2 | cpuset 0-2, `intra_op_threads: 0` (v4) | 2301 | 8.0 / 19.7 | 64.2 / 84.9 / 109.4 | 74.5 / 126.5 | 185.0 | 23 of 2278 | 22 / 2310 (0.95 %) | 0 of 10 | 87.2 | logged | 2.50 | 8.09 |
| 1 | cpuset 0-2, `intra_op_threads: 2` | 2151 | 4.9 / 8.8 | 22.6 / 30.5 / 40.7 | 28.5 / 47.9 | 82.9 | 0 of 2140 | 0 / 2150 (0.00 %) | 0 of 10 | 19.2 | none | 1.46 | 1.83 |
| 2 | cpuset 0-2, `intra_op_threads: 2` | 2215 | 5.0 / 11.9 | 27.2 / 35.7 / 47.2 | 33.7 / 55.1 | 76.8 | 2 of 2195 | 1 / 2206 (0.05 %) | 0 of 10 | 42.4 | none | 1.53 | 2.20 |
| 1 | no cpuset, `intra_op_threads: 0` (v3) | 2183 | 5.2 / 10.9 | 19.3 / 36.4 / 56.6 | 26.5 / 65.9 | 77.2 | 5 of 2173 | 0 / 2183 (0.00 %) | 0 of 10 | 21.2 | none | 2.35 | 4.25 |
| 2 | no cpuset, `intra_op_threads: 0` (v3) | 2214 | 4.9 / 12.4 | 25.1 / 44.2 / 66.0 | 31.9 / 75.6 | 81.5 | 14 of 2204 | 0 / 2214 (0.00 %) | 0 of 10 | 15.5 | none | 2.50 | 2.70 |

- **Where run 1's cells come from.** Its load checks, container CPU, pin errors and its
  default-pool start-up lines are in e2e.log, and its bag numbers in e2e_summary.txt. Its
  warm-up times, the `intra_op_threads: 2` start-up line and its per-trial outcomes were read
  from the launch logs and `trials.csv`, which were deleted after the run: e2e.log does not have
  them, because the driver did not log them then and its start-up grep could not match a line
  with ", 2 intra-op threads". The driver logs all of them now (run 2). Run 1's gaps were
  counted from float-second times (section 1), which can move a count by one.
- **Inference and image age: both runs reproduced v4 and v3.**
  - With v4's setting: inference p50 / p99 62.7 / 111.8 and 64.2 / 109.4 ms, preprocessing 8.3
    and 8.0 ms p50, image age p99 186.9 and 185.0 ms. v4's sessions: 60.5–69.4 /
    115.7–122.6 ms, 7.4–9.1 ms and 188–194 ms.
  - The 1-minute load average ended at 7.94 and 8.09; v4's session 2, which also ran a policy
    alone, ended at 7.94.
  - Unpinned with the default it ran as v3's did.
- **2 threads in the cpuset gave v3-like timing** in both runs, at about 1.5 cores for the
  whole container: at most 2 gaps over 150 ms, and one fallback zero in 20 episodes (0.05 % in
  run 2, within v3's 0.00–0.09 %). Its start-up line reads `onnxruntime 1.31.1
  CPUExecutionProvider, 2 intra-op threads` (e2e.log, run 2), ONNX Runtime logged no pin error,
  and the running node reported `intra_op_threads` 2.
- **Fallback zeros: about half of v4's.** With v4's setting, 1.20 % and 0.95 % of the requests
  were fallback zeros (1.07 % over both runs). v4's `noise` session, the same condition and
  model, had 2.41 %. Every one of its 30 `bc_v4` episodes had 1 to 11, and no random draw of
  10 or 20 of those episodes came down to the check's rate: the lowest of 20,000 draws was
  1.45 % for 10 episodes and 1.84 % for 20. The check matches v4's `clean_ref` (1.02 %) and is
  under half of its `noise`, `wide` and `hfov62` segments (2.41 %, 2.90 %, 2.33 %; session 2:
  1.87 %).
- **Last-second fallback zeros: none.** None of the 20 episodes with v4's setting had one,
  against 10 of the 30 in v4's `noise` session (Fisher's exact test, one-sided: p = 0.003 for
  both runs, 0.035 for either run alone). Those are the zeros that raised v4's 21 validity flags
  (SUMMARY.md section 3).
- **What adds the rest was not found.** The check ran a policy alone, for 10 trials, on an idle
  host. v4's session 1 also ran `servo_baseline` beside the policy and interleaved servo
  episodes with it (its `noise` segment: 60 trials in 24 minutes), and its host was shared with
  other work in parts of the run, with only spot checks of what else ran (RUN_LOG.md, "Shared
  host"). None of these was tested. Session 2 ran its policy alone, as the check did, and still
  had 1.87 % and 9 of 30 episodes with a last-second fallback zero (in another condition, with
  another model: `hfov62`, `bc_v4_62`).
- **Outcomes are not compared.** Run 1 scored 9/10 in each configuration, pose 5 failing in all
  three. Run 2 scored 8/10 with v4's setting (poses 1 and 5 failing) and 9/10 in the other two
  (pose 5). That says nothing at n = 10.
- **One provenance gap.** In run 2 with v4's setting, the query for the running node's
  `intra_op_threads` returned "Node not found" (e2e.log); the params file and the pin error show
  the setting. Section 8 has the fix.

## 6. Conclusion

- **Which factor.** The 3-core cpuset combined with ONNX Runtime's default intra-op pool.
  - The pool is sized from the host's cores, not the container's: 4 threads.
  - 3 of them are spinning workers; the policy process used 1.9–2.5 cores at 10 Hz.
  - With the simulator's own nodes on the same 3 cores, inference and preprocessing are starved,
    ticks overrun, frames age, and fallback zeros follow.
- **How much of the inference slowdown.** Essentially all of it.
  - With only the cpuset changed, inference p50 went from 23.0–26.8 to 58.2–61.8 ms (2.5 and
    2.3 times in the two replicates) and p99 from 50.4–51.9 to 97.9–111.2 ms.
  - The sessions changed from 21.7–33.6 to 60.5–69.4 ms (p50) and from 47.9–67.1 to
    115.7–122.6 ms (p99).
  - The real node reproduced v4's timing with v4's setting and v3's without the cpuset, in two
    runs.
  - For the inference time, nothing is left to attribute to the other candidates. The host was
    idle. The servo node did not run in the benchmark, and v4's session 2 was as slow without
    it. The models cost the same (section 2). The preprocessing increase is a symptom of the
    same starvation.
- **How much of the fallback zeros.** Only part. With v4's setting the real node's rate went
  from v3's level (0.00–0.09 %) to about 1 % (1.20 % and 0.95 %), half of v4's 2.41 % in the
  same condition, and none of its 20 episodes had the last-second fallback zero that 10 of v4's
  30 had (section 5). What adds the rest in v4's sessions was not found.
- **The shared host** may still have added to single windows, such as the `wide` window p99 of
  127.5 ms during the other workflow's downloads (SUMMARY.md, exploratory 3). It did not set
  the level of the inference time.
- **What restores v3-like latency in a 3-core cpuset.** 2 intra-op threads.
  - It is v3-like or better at p50 and p99 in every tested background, with no tick over
    100 ms, and in the end-to-end check it had one fallback zero in 20 episodes.
  - Under load it had the lowest p99 of the sizes tried. 3 threads can match it at p50 with the
    simulator's nodes, not at p99, and takes 1.5–1.7 cores.
  - It uses about 1 core instead of 1.9–2.5.
  - Turning off spinning with the default pool works as well in the benchmark. It is not
    implemented here, because one setting is enough and the thread count also bounds the CPU
    the policy can take from the simulator.

## 7. What it means for the v4 caveats

This changes no v4 number, prediction verdict or comparison. It bears on the caveats
SUMMARY.md states as follows.

- **"Inference was 2–3 times slower than in v3 … The cause was not investigated"**
  (exploratory 3, and "What this does not show").
  - The cause is a property of the v4 setup, `SIM_CPUSET=0-2` with ONNX Runtime's default
    pool. The models, their inputs and the other work on the host are not needed to explain it.
  - v4 did not measure the policies at v3's inference speed.
- **The handicap fell on the policy arms only.**
  - The servo does not use ONNX Runtime.
  - Every `bc_v4` and `bc_v4_62` trial ran with the handicap, in both sessions. So C2 compares
    two arms under it, and C1 and C3 compare an arm under it with one that had none.
  - Whether the C1 and C3 deficits would shrink at v3-like speed was not tested. That needs a
    new run: the end-to-end check's 10 episodes per configuration and run are too few for
    success rates.
- **The validity check's flags.**
  - The 21 flags all come from fallback zeros in a success's last second (SUMMARY.md
    section 3). Exploratory 1 found that in all 21 the policy's own requests met the still
    rule: the flags mark briefly stale inputs, not stops the policy did not make.
  - The cpuset with the default pool explains part of those stale inputs. Fallback zeros were
    0.00–0.09 % of the policies' requests in v3 and 1.0–2.9 % in v4; in the end-to-end check,
    1.20 % and 0.95 % with v4's setting, 0.00 % and 0.05 % with 2 threads.
  - The rest was not reproduced on an idle host. The check reached about half of v4's rate in
    the same condition, and none of its 20 episodes had a last-second fallback zero, against 10
    of 30 in v4's `noise` session. The servo node and the interleaved servo episodes of
    session 1, the length of its segments and the shared host remain candidates; none was
    tested (section 5).
  - So the sensitivity line with flagged successes counted as failures (17/30 for `bc_v4` in
    `noise`, 16/30 for `bc_v4_62` on `hfov62`) counts stale inputs, not stops the policies
    failed to make. This setup made such inputs more frequent; how many of the 21 flags it
    explains on its own is not known.
  - The primary results stay as scored.
- **The shared host** ("What ran"; RUN_LOG.md, "Shared host"): not needed to explain the
  inference slowdown. It remains uncontrolled noise in single windows, and one of the
  candidates for the fallback zeros the check did not reproduce.
- **Session 2's load average** (1.27 → 7.94 with only its container running; "not
  investigated").
  - This fits the default pool's 3 spinning workers, which keep about 2 cores busy while the
    policy runs (1.9–2.5 for the policy process, section 4), and session 2 ran a policy in every
    trial.
  - The end-to-end check with v4's setting, also a policy alone, ended at 7.94 and 8.09; with
    2 threads at 1.83 and 2.20.
  - This was not measured in session 2 itself.
- **Reproducing v4.** `config/sim.yaml` now sets `intra_op_threads: 2`. v4 ran without that
  line. `intra_op_threads: 0` restores v4's session creation exactly, including the failed pin.

## 8. The change

- **`policy_runner`**: a parameter `intra_op_threads` (integer, default 0), checked to be ≥ 0
  and passed to `create_backend`.
- **`policy_backends`**: `create_backend(..., intra_op_threads=0)` refuses a negative,
  non-integer or boolean value, for every backend.
  - `OnnxRuntimeBackend` with 0 creates the session exactly as before, with no session options.
  - With N > 0 it sets `SessionOptions.intra_op_num_threads = N`, and its description, which
    the node logs at start-up, ends in ", N intra-op threads".
  - The other backends ignore the value.
- **Documentation**: the node's docstring, `docs/INTERFACES.md` (Policy I/O), and
  `config/sim.yaml`, which sets 2 with a comment that every earlier run (the clean run, v2,
  v3, v4) ran without it. `config/robot.yaml` is unchanged.
- **Tests**: `test/test_policy_backends.py`, `test/test_camera_policy_ros.py` and
  `test/test_launch_ros.py`.
  - Session options, against a stand-in `onnxruntime`, so this also runs in the ROS
    containers.
  - Refused values.
  - The real pool's size and affinity (host).
  - The node's plumbing and its parameter check (ROS).
  - `sim_dock.launch.py` with the policy controller: the running `policy_runner` reports
    `intra_op_threads` 2 from `sim.yaml` (ROS). `test_launch_config.py` checks the key's type.
- **Provenance**: `scripts/sim_eval.sh` (with a policy arm) and `scripts/sim_e2e.sh` (with the
  policy controller) add `policy_runner.intra_op_threads`, as the running node reports it, to the
  parameters each segment or stage records. Their `node_params` now tries a query up to 5 times
  while it returns "Node not found": several queries at once in the 3-core container sometimes
  miss a running node, as in run 2 here (section 5) and for `servo_baseline.hfov_rad` in v4's
  `noise` segment (SUMMARY.md, "Provenance").

## Files and how to reproduce

[timing/](timing/) holds the scripts and their raw outputs. The bags, models and logs they read
stay in `sim_work/`, not in git.

- **The copies here** are the scripts that produced the numbers. The two drivers
  (`run_matrix.sh`, `run_e2e.sh`) take their paths from the environment instead of the run's
  scratch paths, and `model_info.py` lost an unused import.
- **Changed after the review.** `run_matrix.sh` gained the `threads` phase. `run_e2e.sh` has a
  start-up grep that matches ", N intra-op threads" and also logs the warm-up, the parameters
  the nodes report and every trial. `e2e_analyze.py` prints one line per episode.
  `timing_bags.py` counts gaps from integer nanoseconds, which moved one count in section 1
  (186 → 185). `episode_zeros.py` is new.
- **The benchmark image** was built from commit 4f0723a, before the new parameter existed, so
  the harness set thread counts by wrapping `onnxruntime.InferenceSession`. With a newer image
  it passes `intra_op_threads` to `create_backend` instead; the session options are the same.
  The repeats after the review used such an image, built the same way from the working tree.

| File | What |
|---|---|
| `timing_bags.py` → `timing_bags.csv` | section 1: the v3 and v4 session bags, per segment |
| `model_info.py` → `model_info.jsonl`; `model_speed.py` → `model_speed.txt` | section 2: the four ONNX models |
| `extract_frames.py` | the recorded frames the harness feeds |
| `harness.py`, `busy.py`, `hostload.py` | section 4: `policy_runner`'s tick on recorded frames; the calibrated load; the CPU check before every run |
| `run_matrix.sh` → `results.jsonl`, `results.log` | section 4: the matrix, one JSON line per run; every load check and ONNX Runtime message; the repeats after the review (`PHASES=threads`) appended |
| `make_tables.py` | section 4's tables from `results.jsonl` |
| `run_e2e.sh`, `e2e_analyze.py` → `e2e.log`, `e2e_summary.txt` | section 5, runs 1 and 2 |
| `episode_zeros.py` → `episode_zeros.txt` | sections 5 and 7: fallback zeros per policy episode in the v3 and v4 session bags; the end-to-end check against v4's `noise` session |

```bash
# The image: the simulation's own container setup, committed once. Behind the HTTPS proxy, also
# pass -e HTTPS_PROXY -e https_proxy=$HTTPS_PROXY and the CA bundle, as scripts/sim_eval.sh does.
mkdir -p sim_work/timing
C=$(docker run -d --network host -v "$PWD:/repo:ro" -v "$PWD/sim_work/timing:/work" \
    mirror.gcr.io/library/ros:jazzy-ros-base sleep infinity)
docker exec "$C" bash /repo/scripts/sim_e2e.sh --inner setup
docker exec "$C" bash /repo/scripts/sim_e2e.sh --inner onnxruntime
docker commit "$C" jbl_timing:local && docker rm -f "$C"

T=results/sim_eval_v4_cleanstop/exploratory/timing
$PYTHON $T/timing_bags.py . --csv $T/timing_bags.csv
$PYTHON $T/extract_frames.py sim_work/v4/eval_v2/jazzy/noise/bags/seg_20261029T144308Z bc_v4 600 \
    sim_work/timing/frames_v4_noise.npz
$PYTHON $T/extract_frames.py sim_work/v2/eval_v2/jazzy/noise/bags/seg_20261029T060235Z bc_v2 600 \
    sim_work/timing/frames_v3_noise.npz
$PYTHON $T/model_info.py sim_work/{v2,v2sw,v4,v4_62}/runs/bc/policy.onnx
PYTHONPATH=ros2_ws/src/jbl_nodes $PYTHON $T/model_speed.py sim_work/timing/frames_v4_noise.npz \
    sim_work/{v2,v2sw,v4,v4_62}/runs/bc/policy.onnx
IMG=jbl_timing:local PYTHON=$PYTHON bash $T/run_matrix.sh          # about 32 min
# the repeats after the review, about 5 min, appended as in timing/results.jsonl
IMG=jbl_timing:local PYTHON=$PYTHON PHASES=threads OUTNAME=results_threads bash $T/run_matrix.sh
cat sim_work/timing/results_threads.jsonl >> sim_work/timing/results.jsonl
python3 $T/make_tables.py sim_work/timing/results.jsonl
IMG=jbl_timing:local PYTHON=$PYTHON bash $T/run_e2e.sh pin_t0 0-2 0   # about 5 min each
IMG=jbl_timing:local PYTHON=$PYTHON bash $T/run_e2e.sh pin_t2 0-2 2
IMG=jbl_timing:local PYTHON=$PYTHON bash $T/run_e2e.sh free_t0 "" 0
$PYTHON $T/e2e_analyze.py . sim_work/timing/e2e pin_t0 pin_t2 free_t0
# run 2, after the review: the same with pin_t0_r2, pin_t2_r2 and free_t0_r2
$PYTHON $T/episode_zeros.py . --observed run1_pin_t0=26/2169/0/10 \
    --observed run2_pin_t0_r2=22/2310/0/10 --observed both_runs=48/4479/0/20
```

`$PYTHON` is a Python 3.12 with `requirements-tools.txt` (it needs `rosbags`, `onnx` and
`onnxruntime`). The matrix ran 21:17–21:49 UTC and the end-to-end check 21:50–22:04 UTC; the
repeats after the review ran 23:27–23:32 UTC (matrix) and 23:32–23:47 UTC (end-to-end run 2).
The temporary images, the containers, the frames and the bags were deleted afterwards.
