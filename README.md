# autojetsonbot-learning

Teleop demonstrations → LeRobot dataset → learned policy → on-robot deployment and paired
trials, on a real low-cost robot: [autoJetsonBot](https://github.com/jakhon37/autoJetsonBot)
(Jetson Nano B01, ROS 2 Foxy, RPLidar A1, Pi Camera v2). The task is **camera-only docking**:
from one of 10 marked start poses, drive to a box with an ArUco marker on its front face and
stop squarely in front of it. A classical baseline (ArUco bearing + lidar range servo) and
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
        │  /cmd_vel_in   requested (v, w)  ── this is what the dataset records as the action
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
into the autoJetsonBot checkout so the container sees it under `/autonomous_ROS/`):

```bash
cd /autonomous_ROS/autojetsonbot-learning/ros2_ws && colcon build --symlink-install
source install/setup.bash
bash ../scripts/record_robot.sh                          # camera, gate, judge, logger + bag of the demos
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in   # other terminal
```

Nothing but `safety_gate` may publish `/cmd_vel`: `diffdrive_node` has no command watchdog, so
any other publisher bypasses the gate (the launch files respawn the gate if it dies; a crash
or SIGKILL of the gate still leaves the motors on their last command until it is back).

## Training

```bash
python tools/bag_to_lerobot.py ~/jbl_bags --root data/jbl_dock --repo-id <you>/jbl_dock   # Foxy .db3 or Jazzy .mcap
python tools/train_bc.py --root data/jbl_dock --out runs/bc --pretrained --methods teleop --outcomes success --split-by pose_id
python tools/export_onnx.py runs/bc/model.pt --dataset-root data/jbl_dock                 # -> runs/bc/policy.onnx + policy_meta.json
DATASET_ROOT=data/jbl_dock bash tools/train_act.sh                                        # ACT via lerobot-train
```

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
Foxy built against JetPack's Python (e.g. `dustynv/ros:foxy-ros-base-l4t-r32.7.1`), with this
workspace built inside it. The ROS nodes avoid post-3.6 syntax and standard-library calls
(checked with `vermin`, not run on 3.6). **Untested**: no Jetson was available, so the
TensorRT backend and the engine script have only run against stand-ins.

## Results (SIMULATION)

> **SIMULATION ONLY.** Every number in this section comes from `fake_robot` in a Docker
> container on an x86 laptop CPU, not from autoJetsonBot. It shows that the pipeline runs end
> to end. It says nothing about how either method does on the real robot.

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

| ROS 2 (container) | Method | Success | Wilson 95% CI | Time to dock, s | Final range error, cm | Final bearing error, deg |
|---|---|---:|---:|---:|---:|---:|
| Jazzy (Python 3.12, ONNX Runtime 1.31.1) | servo | 10/10 | 72–100% | 9.7 (8.5–13.5) | 3.5 (2.9–3.6) | 3.6 (3.6–3.7) |
| Jazzy | bc | 10/10 | 72–100% | 10.7 (8.9–13.7) | 3.8 (3.5–4.8) | 1.1 (0.9–2.1) |
| Foxy (Python 3.8, ONNX Runtime 1.10.0) | servo | 10/10 | 72–100% | 10.1 (8.3–14.1) | 3.3 (3.0–3.5) | 3.6 (3.4–3.6) |
| Foxy | bc | 10/10 | 72–100% | 10.7 (8.8–13.7) | 3.9 (3.7–4.7) | 1.3 (1.0–2.0) |

Times and errors are median (IQR). Time to dock includes the judge's 1 s success hold; range
error is measured from the 0.25 m target. Paired exact McNemar test of bc against servo: no
discordant pairs (b = 0, c = 0, p = 1) on either distro.

Training and runtime on the same CPU (4 cores):

- **Training.** 32 episodes / 3,801 frames for training, and 8 episodes / 948 frames from 4
  held-out spots for validation. Each epoch took 4–7 min. Validation MAE was 0.013 m/s and
  0.030 rad/s, against 0.038 m/s and 0.141 rad/s for always predicting the mean command.
- **Policy latency.** Inference took 20 ms (p50) and 55 ms (p99) on Jazzy, 22 ms and 41 ms on
  Foxy. Tick start to `/cmd_vel_in` took 26–28 ms (p50). At the start of a tick the newest
  frame was 38 ms old (p50).
- **Safety gate timing.** The 50 Hz timer missed 1–5 deadlines per run of about 8,000 ticks
  (≤ 0.064%). The period p99 was at most 23.7 ms.

What this does and does not show:

- **The parts work together.** On Foxy, the robot's distro, and on Jazzy:
  - recording, conversion, training, export, ONNX Runtime inference in a ROS node, the
    judge, the logger and the statistics all ran;
  - the trained policy docked from all 10 held-out test poses.
- **The comparison is at a ceiling.** Both methods scored 10/10, so with 10 trials each the
  confidence intervals are wide (72–100%) and there is no evidence of a difference either way.
- **Why BC does well here.**
  - It imitates the servo in the same flat-shaded room it was trained in.
  - The trials had no disturbances.
  - Its smaller final bearing error comes from its continuous output: the servo stops turning
    inside a ±4° alignment deadband.
- **How the Foxy servo trials ran.** BC training was running on the same CPU at the time
  (niced). The Foxy trials reuse the model trained on the Jazzy recordings.

Reports with every table, the outcome breakdown and the chart:
[results/sim_e2e/jazzy/report.md](results/sim_e2e/jazzy/report.md) and
[results/sim_e2e/foxy/report.md](results/sim_e2e/foxy/report.md). The demonstration log, BC
metrics, split and `policy_meta.json` are next to them.

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

- **Containers.** 229 tests on each of Foxy, Humble and Jazzy, with 0 failures and 6 skips. The
  skipped tests are the ONNX Runtime and TorchScript backend checks; the base images have
  neither package.
- **Host.** 314 passed and 5 skipped. The skipped tests need rclpy or `sensor_msgs`.

## Limitations

- **Simulation only so far.** `fake_robot` is a unicycle with acceleration limits, optional
  wheel slip and bias, a flat-shaded pinhole camera and an ideal 2D lidar: no motor deadband or
  latency, no lens distortion, blur or rolling shutter, no lighting changes beyond a per-episode
  gain. Sim success rates say the pipeline works end to end, not how a policy does on the robot.
- **The BC policy is trained on scripted servo demonstrations**, from scratch, on CPU, from a
  few thousand frames. It imitates the baseline, so it cannot be expected to beat it; the
  simulation run is a pipeline check with honest numbers, not a learning result. Both methods
  reach 10/10 in the undisturbed simulator, a ceiling that cannot separate them; the trials
  were not repeated with fake_robot's randomisation on (`randomize:=true`), which would be the
  next, harder simulation check.
- **Recorded actions are requests, not motion.** `action` is `/cmd_vel_in`, before the gate
  scales or blocks it near the box. The bags also hold `/cmd_vel` and the gate status, but the
  dataset does not use them yet.
- **ACT is not deployable yet** (action chunks + odometry input vs the image-only, single-action
  ONNX contract); `train_act.sh` trains it, nothing exports or runs it on the robot.
- **TensorRT on the Nano, the real camera, the real lidar and the motor bridge are untested.**
  The lidar's 0° may point backwards on this robot (`forward_angle` may be π): calibrate it
  (PROTOCOL §4) before anything drives.
- **Geometry in INTERFACES.md is approximate.** The docking target (lidar range 0.25 m) is about
  0.22 m from bumper to box, not 0.12 m, and the robot's URDF places the chassis and camera
  differently from the contract; the success band absorbs this in simulation.
- **Episode events need care.** `/episode/event` is transient-local and a fresh `episode` CLI
  process can take up to ~3.5 s to reach every node (Fast DDS discovery); the CLI waits for
  that (or for `--expect N` subscribers in scripts) and reports a partial delivery with exit 3.
  Use it, not `ros2 topic pub`.

## License

Apache-2.0, see [LICENSE](LICENSE).
