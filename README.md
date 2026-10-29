# autojetsonbot-learning

Teleop demonstrations → LeRobot dataset → learned policy → on-robot deployment and paired
trials, on a real low-cost robot: [autoJetsonBot](https://github.com/jakhon37/autoJetsonBot)
(Jetson Nano B01, ROS 2 Foxy, RPLidar A1, Pi Camera v2). The task is **camera-only docking**:
from one of 10 marked start poses, drive to a box with an ArUco marker on its front face and
stop facing it at docking range (0.18–0.32 m lidar range, marker within 10° of the image
centre; the rule does not measure squareness to the face). A classical baseline (ArUco bearing + lidar range servo) and
learned policies (behaviour cloning, ACT) are compared on the same poses with a referee node
that decides success, a C++ safety gate that is the only thing allowed to drive the motors,
and a statistics script that reports k/n with confidence intervals and paired McNemar tests.
Everything except the hardware runs in a kinematic simulator (`fake_robot`), so the whole
pipeline is testable on a laptop without a GPU.

> **Status.** All numbers in this README are from **simulation**. The real-robot protocol
> ([docs/PROTOCOL.md](docs/PROTOCOL.md)) is written and its tools are tested, but no trials
> have been run on the robot yet, and nothing has run on a Jetson.

## Architecture

The contract (topics, JSON schemas, parameters, geometry, dataset schema, policy I/O, success
rule) is [docs/INTERFACES.md](docs/INTERFACES.md). On the robot `camera_node`, the RPLidar
driver and `diffdrive_node` provide the sensors and motors; in simulation `fake_robot` provides
all three with the same topics.

```text
 SENSORS  (robot: camera_node, rplidar, diffdrive_node   |   simulation: fake_robot publishes all three)
   /camera/image_raw/compressed   JPEG 15 Hz, capture stamp ──▶ servo_baseline, policy_runner, dock_judge
   /scan                          LaserScan ~10 Hz          ──▶ safety_gate, servo_baseline, dock_judge
   /odom                          measured v, w             ──▶ dock_judge

 CONTROL  (one source at a time)
   teleop  |  servo_baseline  |  policy_runner (onnxruntime | tensorrt | torch)
        │
        │  /cmd_vel_in   requested (v, w)  ── the dataset's action; dock_judge's "stopped" needs it still too
        ▼
   safety_gate  (C++, 50 Hz: clamp, slow / stop on the front lidar sector, command and scan timeouts)
        │──▶ /cmd_vel              the only topic the motors obey ──▶ diffdrive_node | fake_robot
        └──▶ /safety_gate/status   2 Hz JSON: state, front_min_m, timer jitter ──▶ dock_judge (obstacle rule)

 EPISODES AND REFEREE
   episode CLI ── "start <method> <pose_id> <round>" ──┐
   dock_judge  ── "end success" | "end fail <reason>" ─┴─▶ /episode/event ──▶ fake_robot (teleport to the pose),
                                                                              controllers (enable / disable),
                                                                              dock_judge (start the clock),
                                                                              trial_logger ──▶ trials.csv
   dock_judge    ──▶ /dock_judge/status  10 Hz JSON: bearing, range, in_band, aligned, stopped, success
   policy_runner ──▶ /policy/latency     JSON per inference: preprocess, infer, total, image age

 OFFLINE  (host, Python 3.12)
   ros2 bag record ─▶ bag_to_lerobot.py ─▶ LeRobot v3 dataset ─▶ train_bc.py ─▶ export_onnx.py
       ─▶ policy.onnx + policy_meta.json ─▶ policy_runner (ONNX Runtime off-board, or a TensorRT engine on the Nano)
   trials.csv + bags ─▶ analyze_trials.py ─▶ report.md, success.png
```

| Package / dir | What it is |
|---|---|
| `ros2_ws/src/jbl_safety` | C++17 safety gate: ROS-free `SafetyGate` logic (gtest) + node; forward sector check with lidar dead-zone handling, command/scan timeouts, clamps, timing stats |
| `ros2_ws/src/jbl_nodes` | Python (3.8-compatible) nodes: `camera_node`, `dock_judge`, `servo_baseline`, `policy_runner`, `trial_logger`, `fake_robot`, the `episode` CLI, shared preprocessing, launch files and params |
| `tools/` | Host tools (Python 3.12, `requirements-tools.txt`): bag → LeRobot conversion, BC training, ONNX export, ACT training wrapper, TensorRT build script, marker generator, trial schedule, analysis and statistics |
| `docs/` | `INTERFACES.md` (contract), `PROTOCOL.md` (real-robot runbook) |
| `scripts/` | `docker_test.sh` (build + test on Foxy/Humble/Jazzy), `sim_e2e.sh` (whole pipeline in simulation), `record_robot.sh` (demos on the robot) |
| `results/sim_e2e/` | Small outputs of the simulation run: trial CSVs, `report.md`, `success.png`, training metrics |

Launch files (`ros2 launch jbl_nodes <file> --show-args` lists every argument):

| Launch file | Nodes |
|---|---|
| `sim_dock.launch.py` | `fake_robot`, `safety_gate`, `dock_judge`, `trial_logger` + one controller chosen by `method` / `controller` (`servo_baseline`, `policy_runner` with `model_path` and `backend`, or none) |
| `robot_record.launch.py` | `camera_node`, `safety_gate` (respawned), `dock_judge`, `trial_logger` (+ `servo_baseline` with `with_servo:=true`); teleop is external |
| `robot_policy.launch.py` | the same + `policy_runner` (`model_path`, `backend`, `method`) |

All three take `bag:=<new dir>` to record the INTERFACES topic list, `trials_csv:=`,
`params_file:=` / `extra_params_file:=`, and `forward_angle:=` / `hfov_rad:=` overrides that
reach every node that uses them. Parameters live in `config/sim.yaml` and `config/robot.yaml`
(robot calibration goes in its `/**` section once, so the gate, the judge and the baseline
cannot disagree).

## Quick start in simulation

Needs Docker and, for training, a Python 3.12 environment with `requirements-tools.txt`.

```bash
bash scripts/sim_e2e.sh servo                                   # servo baseline, 10 poses, ~5 min
PYTHON=.venv/bin/python bash scripts/sim_e2e.sh                 # everything below, ~50 min on 4 CPU cores
PYTHON=.venv/bin/python bash scripts/sim_eval.sh                # harder v2 evaluation of the trained model, ~35 min per condition
```

`sim_e2e.sh` starts a clean `ros:jazzy-ros-base` container (`--distro foxy|humble` also work),
builds the workspace and runs, in order:

1. **servo**: the baseline on the 10 test poses, episodes started with the `episode` CLI and
   ended by `dock_judge`, rows written by `trial_logger`, everything recorded with `ros2 bag`;
2. **demos**: scripted demonstrations from 20 *training* spots (`config/sim_demo_poses.yaml`,
   at least 10 cm from every test pose) with the servo under the method name `servo_demo`,
   two gain settings, and fake_robot's wheel slip, start jitter and image noise switched on;
3. **train** (host): `bag_to_lerobot.py` → `train_bc.py` (ResNet18 **from scratch**, CPU,
   `servo_demo` successes only, validation on held-out spots) → `export_onnx.py`;
4. **bc**: `policy_runner` with ONNX Runtime (CPU) on the 10 test poses;
5. **report**: `analyze_trials.py` → `results/sim_e2e/<distro>/report.md` and `success.png`.

By hand, inside a ROS 2 container with the workspace built:

```bash
ros2 launch jbl_nodes sim_dock.launch.py trials_csv:=/tmp/trials.csv          # terminal 1
ros2 run jbl_nodes episode --timeout 5 start servo 3 1                         # terminal 2
tail -f /tmp/trials.csv                                                        # dock_judge ends it
```

## On the real robot

[docs/PROTOCOL.md](docs/PROTOCOL.md) is the runbook: marker and box, floor layout of the 10
start poses, lidar `forward_angle` and camera HFOV calibration, teleop demonstrations, freezing
models before trials, the counterbalanced schedule, interventions and aborts, video rules,
analysis and honest reporting.

Build inside the robot's Foxy container (autoJetsonBot's `./robot.sh shell`; clone this repo
into the autoJetsonBot checkout so the container sees it under `/autonomous_ROS/`). The stock
container cannot open the CSI camera: recreate it once with `--runtime nvidia -v
/tmp/argus_socket:/tmp/argus_socket` and check it (PROTOCOL section 4). Then:

```bash
cd /autonomous_ROS/autojetsonbot-learning/ros2_ws && colcon build --symlink-install
source install/setup.bash
bash ../scripts/record_robot.sh    # camera check, then camera, gate, judge, logger + bag in data/robot/
```

Recordings go to `data/robot/` in this checkout, which is on the `/autonomous_ROS` bind mount
(`/root` in the container is not persistent). Teleop must publish `/cmd_vel_in`: the web
dashboard with its Command Velocity topic set to `/cmd_vel_in` (it publishes continuously while
a key is held), a joystick, or `teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in` with a
fast key repeat (PROTOCOL section 6).

Nothing but `safety_gate` may publish `/cmd_vel`: `diffdrive_node` has no command watchdog, so
any other publisher bypasses the gate. The gate publishes a zero twist on SIGINT, SIGTERM,
SIGHUP and SIGQUIT, and the launch files respawn it if it dies; a crash or SIGKILL of the gate
still leaves the motors on their last command until it is back (about 3 s on Foxy because of
DDS rediscovery). The ESP32 firmware's 500 ms `m` command timeout is what actually bounds
this, so verify it (PROTOCOL section 4) before anything drives on its own.

## Training

Copy the recordings off the robot first (the container writes them world-readable):

```bash
rsync -av <jetson>:<autoJetsonBot checkout>/autojetsonbot-learning/data/robot/ data/robot/
```

```bash
python tools/bag_to_lerobot.py data/robot/bags --root data/jbl_dock --repo-id <you>/jbl_dock   # Foxy .db3 or Jazzy .mcap
python tools/train_bc.py --root data/jbl_dock --out runs/bc --pretrained --methods teleop --outcomes success --split-by pose_id
python tools/export_onnx.py runs/bc/model.pt --dataset-root data/jbl_dock                 # -> runs/bc/policy.onnx + policy_meta.json
EPISODES=$(python -c 'import json; print(json.load(open("runs/bc/split.json"))["train"])') \
  DATASET_ROOT=data/jbl_dock bash tools/train_act.sh                                      # ACT via lerobot-train, BC's training episodes
```

`train_act.sh` without `EPISODES` selects the same way (`METHODS=teleop`, `OUTCOMES=success`,
split by pose_id, validation fraction 0.2, seed 0); `EPISODES=all` trains on everything,
including failed demos and trial rollouts. `bag_to_lerobot.py` drops each episode's idle start
(the zero commands before the operator's first one; `--keep-idle-start` keeps them).

The dataset is LeRobot v3 at 10 fps: `observation.images.front` (240×320 RGB from the JPEG
stream), `observation.state` (odometry v, w), `action` (the `/cmd_vel_in` command), plus
`meta/jbl_episodes.jsonl` (method, pose, round, outcome per episode). Training and the robot
share one preprocessing module (`jbl_nodes/preprocess.py`), and `export_onnx.py` refuses to
write a model whose ONNX Runtime output differs from PyTorch by more than 1e-4 on real
dataset frames. ACT trains with `tools/train_act.sh` but cannot be deployed yet: it outputs a
chunk of actions and also takes odometry, while the robot's model contract is image in, one
normalised (v, w) out (see Limitations).

## Deployment

The policy is an ONNX file with `policy_meta.json` beside it; `policy_runner` checks the meta
(input size, normalisation, action scale) before it loads the model and publishes
`/cmd_vel_in` at 10 Hz, or zero when the newest frame is older than 0.2 s.

**Off-board (laptop over Wi-Fi, ONNX Runtime).** The robot runs `robot_record.launch.py`
(camera, gate, judge, logger); the laptop, on the same `ROS_DOMAIN_ID`, runs

```bash
ros2 run jbl_nodes policy_runner --ros-args -p backend:=onnxruntime \
  -p model_path:=runs/bc/policy.onnx -p method_filter:=bc
```

Frames are aged by their capture stamp, so the two clocks must agree (chrony, offset < 5 ms)
or the runner sees stale frames and publishes zeros. Wi-Fi latency and dropped JPEGs count
against the policy; label such results "off-board". ONNX Runtime comes from PyPI, installed
next to the distro's own numpy, which python3-opencv is built against (`inner_onnxruntime` in
`scripts/sim_e2e.sh`). The simulation runs used:

- **Foxy:** Ubuntu's numpy 1.17 limits ONNX Runtime to 1.10.0, the last release that accepts
  numpy ≥ 1.16.6.
- **Jazzy:** ONNX Runtime 1.31.1.

**On-board (Jetson Nano, TensorRT).** Build the engine on the Nano itself with
`bash tools/build_trt_nano.sh policy.onnx` (TensorRT 8.0/8.2 from JetPack 4.6; the export is
opset 13 for that reason) and run `robot_policy.launch.py model_path:=policy_fp16.engine
backend:=tensorrt`. The TensorRT Python bindings of JetPack 4.6 are built for Python 3.6, but
autoJetsonBot's Foxy container is Ubuntu 20.04 / Python 3.8, so the practical route is a
[jetson-containers](https://github.com/dusty-nv/jetson-containers) L4T image that has ROS 2
Foxy built against JetPack's Python (`dustynv/ros:foxy-ros-base-l4t-r32.<x>`), with this
workspace built inside it. Pick the image tag that matches the first line of
`/etc/nv_tegra_release`: r32.6.1 for JetPack 4.6, r32.7.1 for 4.6.1 (if that tag is not
published, build the image with jetson-containers for your L4T). Start it with
`--runtime nvidia --network host --ipc host`: the nodes in autoJetsonBot's `auto_ros_foxy`
container talk to it over Fast DDS shared memory (its `docker-compose.yaml` already gives that
container `ipc: host`; `robot.sh`'s fallback `docker run` does not, so use the one in
PROTOCOL section 4). The tensorrt backend also needs pycuda, which JetPack does not install
(cuda-python needs CUDA 11 and Python ≥ 3.7, so it cannot replace it on JetPack 4.6's CUDA
10.2). Install and check it inside the container:

```bash
export PATH=/usr/local/cuda/bin:$PATH
pip3 install 'pycuda==2022.1'   # the last release for Python 3.6
python3 -c 'import tensorrt, pycuda.driver'
```

The ROS nodes avoid post-3.6 syntax and standard-library calls (checked with `vermin`, not run
on 3.6). **Untested**: no Jetson was available, so the TensorRT backend and the engine script
have only run against stand-ins.

## Results (SIMULATION)

> **SIMULATION ONLY.** Every number in this section comes from `fake_robot` in a Docker
> container on an x86 laptop CPU, not from autoJetsonBot. It shows that the pipeline runs end
> to end. It says nothing about how either method does on the real robot.

There are two runs. The **clean run** below has randomisation off and 1 round per pose. It is at
the ceiling for the servo (10/10) and the floor for BC (0/10), so it says nothing about how
either method degrades. The **harder evaluation (v2)** after it adds randomisation, start poses
beyond the training distribution and the robot's 62.2° lens, with 10 poses × 3 rounds per
condition. Its results are in [results/sim_eval_v2](results/sim_eval_v2/SUMMARY.md).

### Clean run (pipeline check; at ceiling for the servo, floor for BC)

Setup (`scripts/sim_e2e.sh`, October 2026):

- **Trials.** 10 test poses × 1 round per method. The `episode` CLI started each episode,
  `dock_judge` ended it, and `trial_logger` wrote the row. fake_robot's randomisation was off
  during the trials.
- **servo.** The classical ArUco-bearing + lidar-range baseline (`servo_baseline`).
- **bc.** ResNet18 trained from scratch on CPU for 5 epochs, on 40 scripted `servo_demo`
  demonstrations from 20 separate training spots (2 gain settings, randomisation on). The 10
  test poses were never trained on. The model was exported to ONNX (opset 13; ONNX Runtime
  matches PyTorch to 5.4e-7) and run by `policy_runner` with ONNX Runtime on CPU inside the
  ROS container.
- **Provenance.** The demonstrations and the BC model predate the repository's first commit
  (5a83553; the Jazzy demos also predate a later edit to `launch_common.py`). The servo and BC
  trials below were re-run on 29 October 2026 on 5a83553 plus uncommitted fixes (the review
  fixes to the judge, the gate and the tools). `results/sim_e2e/<distro>/provenance.txt`
  records per stage the commit and the number of changed paths, plus the image digest, the
  ONNX Runtime version and the model's sha256 (`7cfb1cbb…`, also in `bc_policy_meta.json`).
  The 45 MB `policy.onnx` is not in git.

| ROS 2 (container) | Method | Success | Wilson 95% CI | Time to dock, s | Final range error, cm | Final marker bearing, deg |
|---|---|---:|---:|---:|---:|---:|
| Jazzy (Python 3.12, ONNX Runtime 1.31.1) | servo | 10/10 | 72–100% | 13.8 (12.2–18.1) | 2.0 (2.0–2.0) | 3.7 (3.5–3.8) |
| Jazzy | bc | 0/10 | 0–28% | — | 2.0 (1.3–3.0) | 1.9 (1.2–2.3) |
| Foxy (Python 3.8, ONNX Runtime 1.10.0) | servo | 10/10 | 72–100% | 13.6 (12.5–18.2) | 2.0 (2.0–2.0) | 3.7 (3.4–3.9) |
| Foxy | bc | 0/10 | 0–28% | — | 2.0 (1.3–3.0) | 1.9 (1.1–2.3) |

Times and errors are median (IQR). Time to dock includes the judge's 1 s success hold; range
error is measured from the 0.25 m target; the marker bearing is the marker's angle from the
image centre, not the robot's heading to the box face. Paired exact McNemar test of bc against
servo on both distros: b = 10, c = 0, p = 0.002; difference in success rate −100 pp (95%
Newcombe CI −100 to −61 pp). All 10 bc failures are timeouts.

Training and runtime on the same CPU (4 cores):

- **Training.** 32 episodes / 3,801 frames for training, and 8 episodes / 948 frames from 4
  held-out spots for validation. Each epoch took 4–7 min. Validation MAE was 0.013 m/s and
  0.030 rad/s, against 0.038 m/s and 0.141 rad/s for always predicting the mean command.
- **Policy latency.** Inference took 23 ms (p50) and 65 ms (p99) on Jazzy, 22 ms and 40 ms on
  Foxy. Tick start to `/cmd_vel_in` took 27–29 ms (p50). At the start of a tick the newest
  frame was 45 ms (Jazzy) and 64 ms (Foxy) old (p50).
- **Safety gate timing.** The 50 Hz timer missed 7–13 deadlines per run of 9,000–17,000 ticks
  (≤ 0.13%); the period p99 was at most 24.8 ms. The worst single period, 341 ms in the Jazzy
  servo run, fell while a host test suite was running on the same CPU.

What this does and does not show:

- **The parts work together.** On Foxy, the robot's distro, and on Jazzy, recording,
  conversion, training, export, ONNX Runtime inference in a ROS node, the judge, the logger and
  the statistics all ran.
- **The servo docks and stops by itself** from all 10 poses on both distros: it commands zero
  once the lidar range is at most 0.27 m (target + tolerance) and the marker is inside its ±4°
  deadband. The judge's hold can start a little earlier, while the servo still creeps at
  0.017–0.019 m/s, just under the 0.02 m/s "still" threshold. On Jazzy, 2 of the 10 successes
  (poses 4 and 5) were declared before the zero command came.
- **The trained policy never stops, so it never succeeds.** It reaches the docking band facing
  the marker from all 10 held-out poses, but in the last 5 s of every episode it still
  requested 0.021–0.032 m/s; the gate's slow zone cut that to about 0.002 m/s near its stop
  distance, and the judge, which now also requires the request on `/cmd_vel_in` to be still,
  ran each episode into its 30 s timeout. The cause is the training data: the demonstrations
  were recorded with the earlier judge, whose odometry-only "stopped" test (≤ 0.02 m/s) passed
  while the servo was still creeping inside the slow zone, so every demonstration ended before
  the servo commanded zero and the dataset holds no stop examples.
- **The earlier 10/10 for bc was an artefact.** Before the judge checked the request, the same
  `policy.onnx` scored 10/10 on both distros: every success was declared while the robot was
  still being driven forward, slowed below 0.02 m/s only by the gate. (The earlier servo
  successes were also declared while it was still creeping, before its own stop.)
- **"Facing the marker" is not "square to the box".** Ground truth at the end of each servo
  success (`/fake_robot/ground_truth` in the bags): poses 1–6 and 9 ended 16–29° off square
  with 8–15 cm sideways offset; only poses 0, 7 and 8 ended within 4° of square. The judge
  does not measure this.
- **The servo's times grew** from 9.7 s to 13.8 s (median, Jazzy) because the judge now waits
  for it to stop and the gate's forward corridor (`footprint_half_width`) slows oblique final
  approaches; poses 5 and 7 took 26–27 s of the 30 s budget.
- **Small bc final marker bearing** (1.9° against the servo's 3.7°) comes from its continuous
  output: the servo stops turning inside its ±4° deadband.
- **Both Foxy runs reuse the model** trained on the Jazzy recordings.

Reports with every table, the outcome breakdown and the chart:
[results/sim_e2e/jazzy/report.md](results/sim_e2e/jazzy/report.md) and
[results/sim_e2e/foxy/report.md](results/sim_e2e/foxy/report.md). The demonstration log, BC
metrics, split and `policy_meta.json` are next to them.

### Harder evaluation (v2)

Run with `scripts/sim_eval.sh` on Jazzy, 29 October 2026, using the same `policy.onnx`. It was
not retrained: the later fixes change neither its preprocessing nor its dataset format.

- **Conditions.** `fake_robot` ran with `randomize:=true` in all three, with one seed per
  condition:
  - `noise`: the test poses with the randomisation the demonstrations were recorded with
    (lighting gain and offset, pixel noise, wheel bias and slip, lidar noise, start jitter).
  - `wide`: 10 start poses outside the demonstrations' distance / lateral / heading box
    (`config/sim_eval_wide_poses.yaml`: far, wide, steep, near).
  - `hfov62`: the robot's 62.2° lens in place of the simulator's 80°. The judge and the servo
    were told the true HFOV; BC was trained at 80°.
- **Trials.** 10 poses × 3 rounds per method per condition, with servo and bc interleaved in
  one launch in a `tools/make_schedule.py` order. That makes 180 trials: all scored, none
  aborted.
- **Outcomes.** The primary outcome is the unchanged judge rule. The secondary outcome is
  *zone reached*: in band and aligned for ≥ 1 s, with the stop requirement dropped, taken from
  the judge's status in the bags.

| Condition | Method | Success (judge) | Wilson 95% CI | b / c vs servo, exact McNemar p | Time to dock, s | Zone reached (stop ignored) | Time to zone, s | Min requested v, last 1 s, m/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noise | servo | 27/30 | 74–97% | — | 14.8 (12.7–18.1) | 30/30 | 9.3 | 0.000 |
| noise | bc | 3/30 | 3–26% | 24 / 0, p = 1.2e-7 | 13.2 (13.2–21.7) | 30/30 | 10.2 | 0.026 |
| wide | servo | 25/30 | 66–93% | — | 21.0 (17.8–22.0) | 30/30 | 14.9 | 0.000 |
| wide | bc | 1/30 | 1–17% | 24 / 0, p = 1.2e-7 | 21.8 | 30/30 | 16.6 | 0.027 |
| hfov62 | servo | 28/30 | 79–98% | — | 15.1 (12.8–18.7) | 30/30 | 9.3 | 0.000 |
| hfov62 | bc | 20/30 | 49–81% | 8 / 0, p = 0.008 | 25.0 (21.9–26.9) | 30/30 | 15.2 | 0.019 |

Times are medians (IQR) and requested speeds are medians over trials. The full tables, with
final range, bearing, ground-truth squareness and offset, collisions and the clean reference
row, are in [results/sim_eval_v2/SUMMARY.md](results/sim_eval_v2/SUMMARY.md). Each
condition's `report.md` (per-pose results, latency, gate timing), `trials.csv`,
`schedule.csv`, `episodes_gt.csv` and `provenance.txt` are in
`results/sim_eval_v2/<condition>/`.

What v2 shows and does not show:

- **The methods separate on stopping, not on getting there.** The servo beats BC on the judge's
  rule in every condition. Yet both reach the docking zone in 30/30 trials everywhere,
  including from the out-of-distribution poses, with BC 1–6 s slower to get there.
- **BC never commanded a stop in any trial.** Its smallest request in the last second was
  always ≥ 0.016 m/s. Each BC success is its leftover creep dipping under the judge's 0.02 m/s
  "still" threshold. Its 20/30 under the 62.2° lens comes from the narrower view making the
  marker look closer, so BC creeps at about 0.019 instead of 0.027 m/s; it is not robustness to
  the lens.
- **The servo's 10 failures come from three mechanisms.**
  - Pose 5 (5 of 6 trials in `noise` and `hfov62`): lidar noise toggles its request across its
    0.27 m stop range, so the 1 s hold never completes.
  - Wide pose 4 (2/3): about 39° off square, it pushes against the gate as BC does.
  - Wide pose 9 (3/3): it reaches the zone only at 26–27 s of the 30 s limit.

  Nothing collided.
- **What it does not show.**
  - How either method does on the robot (sim to sim, same renderer as the training images).
  - Out-of-distribution lighting (`noise` uses the training magnitudes).
  - Actuation latency (`fake_robot` has no command delay).
  - Independent trials: 3 rounds of the same pose are correlated, so the CIs are too narrow.

## Tests

```bash
bash scripts/docker_test.sh foxy humble jazzy        # colcon build + test of both packages per distro
HF_HUB_OFFLINE=1 python -m pytest tests ros2_ws/src/jbl_nodes/test -q   # host tools + pure-Python node logic
```

The ROS tests include gtest for the gate logic, launch tests for the gate, nodes on a live graph
(camera, policy runner, judge, servo, logger, simulator), and the three launch files (a servo
episode end to end with a bag, a constant policy through the gate, calibration overrides,
argument errors). The host suite covers bag conversion (synthetic Foxy-layout and Jazzy bags),
BC training and ONNX export parity, the statistics against statsmodels/scipy, and the analysis.

Last run (October 2026):

- **Containers.** 243 tests on each of Foxy and Jazzy, with 0 failures and 6 skips. Humble was
  last run before the v2 wide-pose check was added, at 242 tests with 0 failures and 6 skips.
  The skipped tests are the ONNX Runtime and TorchScript backend checks; the base images have
  neither package.
- **Host.** 330 passed and 5 skipped. The skipped tests need rclpy or `sensor_msgs`.

## Limitations

- **Simulation only so far.** `fake_robot` is a unicycle with acceleration limits, optional
  wheel slip and bias, a flat-shaded pinhole camera and an ideal 2D lidar: no motor deadband or
  latency, no lens distortion, blur or rolling shutter, no lighting changes beyond a per-episode
  gain. Sim success rates say the pipeline works end to end, not how a policy does on the robot.
- **The BC policy is trained on scripted servo demonstrations**, from scratch, on CPU, from a
  few thousand frames. It imitates the baseline, so it cannot be expected to beat it; the
  simulation run is a pipeline check with honest numbers, not a learning result. The harder
  simulation evaluation (v2: randomisation on, out-of-distribution poses, the 62.2° lens) is in
  the results above. It does not cover out-of-distribution lighting or actuation latency.
- **The published BC model was never shown a stop.** Its demonstrations were recorded with the
  earlier judge, whose "stopped" test looked at odometry only (≤ 0.02 m/s). Inside the gate's
  slow zone that held while the servo was still asking for 0.02–0.04 m/s, so every
  demonstration ended before the servo commanded zero and the dataset holds no stop examples.
  The judge now also requires the request on `/cmd_vel_in` to be still, so demonstrations
  recorded with it end with at least 1 s of zero command; the model has not been retrained on
  such demonstrations yet.
- **Recorded actions are requests, not motion.** `action` is `/cmd_vel_in`, before the gate
  scales or blocks it near the box. The bags also hold `/cmd_vel` and the gate status, but the
  dataset does not use them yet.
- **ACT is not deployable yet** (action chunks + odometry input vs the image-only, single-action
  ONNX contract); `train_act.sh` trains it, nothing exports or runs it on the robot.
- **TensorRT on the Nano, the real camera, the real lidar and the motor bridge are untested.**
  The lidar's 0° may point backwards on this robot (`forward_angle` may be π): calibrate it
  (PROTOCOL §4) before anything drives.
- **The simulator's chassis is not the URDF's.** `fake_robot` puts the bumper 0.033 m ahead of
  the lidar (the 0.25 m docking range is then ≈ 0.22 m bumper to box); the URDF puts it 0.094 m
  ahead (≈ 0.16 m) and the camera at x = 0.15 m. The success band and the gate thresholds are
  lidar ranges, so they do not change, but measure the real robot (PROTOCOL §4).
- **The success rule does not measure squareness.** "Facing the marker" is a bearing in the
  image, not the robot's heading to the box face or its sideways offset (see the simulation
  results).
- **Episode events need care.** `/episode/event` is transient-local and a fresh `episode` CLI
  process can take up to ~3.5 s to reach every node (Fast DDS discovery); the CLI waits for
  that (or for `--expect N` subscribers in scripts) and reports a partial delivery with exit 3.
  Use it, not `ros2 topic pub`.

## License

Apache-2.0, see [LICENSE](LICENSE).
