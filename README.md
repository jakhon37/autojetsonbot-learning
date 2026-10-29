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
| `scripts/` | `docker_test.sh` (build + test on Foxy/Humble/Jazzy), `sim_e2e.sh` (whole pipeline in simulation), `sim_eval.sh` + `sim_eval_summary.py` (the harder paired evaluation), `stop_labels.py` (stop examples in demos, datasets and models), `record_robot.sh` (demos on the robot) |
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
PYTHON=.venv/bin/python bash scripts/sim_eval.sh --work sim_work/v2 --results results/sim_eval_v3_bcfix \
  --methods bc_v2 --pair-with results/sim_eval_v2 clean_ref noise wide hfov62   # one new arm, paired with earlier rows
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
`train_bc.py --stop-weight W` weights full-stop frames (|v| < 0.005 m/s and |w| < 0.05 rad/s)
W times in the loss. Check how many stop examples a dataset has, and where they are, with
`scripts/stop_labels.py dataset` before training (results/sim_eval_v3_bcfix shows why).

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
> container on an x86 CPU, not from autoJetsonBot. It shows that the pipeline runs end
> to end. It says nothing about how either method does on the real robot.

There are four runs. The **clean run** below has randomisation off and 1 round per pose. It is at
the ceiling for the servo (10/10) and the floor for BC (0/10), so it says nothing about how
either method degrades. The **harder evaluation (v2)** after it adds randomisation, start poses
beyond the training distribution and the robot's 62.2° lens, with 10 poses × 3 rounds per
condition. Its results are in [results/sim_eval_v2](results/sim_eval_v2/SUMMARY.md). **v3**
re-records the demonstrations under the corrected judge, retrains BC and evaluates it under the
same conditions: [results/sim_eval_v3_bcfix](results/sim_eval_v3_bcfix/SUMMARY.md). **v4**
changes the stop rules, adds demonstrations through the robot's lens, and was pre-registered
before it ran: [results/sim_eval_v4_cleanstop](results/sim_eval_v4_cleanstop/SUMMARY.md) (see
the end of this section).

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

### BC retrained with stop examples (v3)

Run on Jazzy, 29 October 2026; the full write-up is in
[results/sim_eval_v3_bcfix/SUMMARY.md](results/sim_eval_v3_bcfix/SUMMARY.md).

- **Data.** The 40 demonstrations were re-recorded with v1's recipe (spots, gains, seeds) under
  the corrected judge. BC was retrained with v1's command (`bc_v2`, sha256 `f6e0b754…`).
- **Remedy.** One pre-registered remedy was tried: full-stop frames weighted 30× in the loss
  (`bc_v2sw`, `e018fd40…`).
- **Pairing.** Each new arm ran bc-only (`sim_eval.sh --methods`) and was paired by pose ×
  round with the v2 session's servo and BC v1 rows (for `clean_ref`, the clean run's)
  (`--pair-with`). The randomised conditions share seeds and schedules across sessions, not
  noise draws.

| Condition | servo | bc (v1) | bc_v2 | bc_v2sw | bc_v2 vs servo, b / c, exact p | bc_v2 vs bc, b / c, exact p | Stop commanded, bc_v2 / bc_v2sw |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_ref | 10/10 | 0/10 | 10/10 | 10/10 | 0 / 0, p = 1 | 0 / 10, p = 0.002 | 0/10 / 0/10 |
| noise | 27/30 | 3/30 | 30/30 | 30/30 | 0 / 3, p = 0.25 | 0 / 27, p = 1.5e-8 | 1/30 / 1/30 |
| wide | 25/30 | 1/30 | 25/30 | 22/30 | 3 / 3, p = 1 | 0 / 24, p = 1.2e-7 | 0/30 / 0/30 |
| hfov62 | 28/30 | 20/30 | 0/30 | 0/30 | 28 / 0, p = 7.5e-9 | 20 / 0, p = 1.9e-6 | 29/30 / 30/30 |

What v3 shows:

- **The corrected judge did not give the data clean stops.** Only 2 of the 40 new
  demonstrations requested zero for their whole last second. The judge still calls a request of
  up to 0.02 m/s still, the servo's heading-scaled creep is just below that, and lidar noise
  toggles its 0.27 m stop. Full-stop labels went from 0.4 % to 3.1 % of frames. On the docked
  view, 20 % of labels are stops and 80 % creep at ~0.019 m/s.
- **BC v2 learned the average.** Offline, on held-out docked-view frames labelled a full stop,
  it predicts 0.013 m/s (BC v1: 0.024).
- **BC v2 now passes the judge by creeping, not by stopping.** With the 80° lens it requests
  ~0.017 m/s and commands a stop in 1 of 70 trials. The judge counts that creep as still, so BC
  v2 matches the servo there. (That 1, and `bc_v2sw`'s 1/30 in `noise`, turned out to be the
  policy runner's fallback zero on a stale frame, not a request of the policy; found while
  preparing v4, which counts stops without them.)
- **Under the 62.2° lens, BC v2 stops short.** The marker looks 1.39× larger, so BC v2 stops at
  ~0.34 m, just outside the band: 0/30, against BC v1's creep-driven 20/30.
- **The stop weighting changed the offline numbers, not the closed-loop outcome.** It brought
  the offline stop prediction to 0.004 m/s, but in closed loop the judge's hold completes at
  ~0.277 m while the request is still 0.012–0.019 m/s, before the stop view is reached.
- **The headline numbers are a threshold effect.** BC's improvement on the success rule comes
  from the 0.02 m/s threshold, not from learning to stop. Testing stopping needs demonstrations
  that end on a commanded stop.

### Clean-stop protocol (v4; pre-registered)

v3 showed that the judge's "still" threshold, not a learned stop, decided the BC results. From
v4 on, two defaults change; [results/sim_eval_v4_cleanstop/PREREG.md](results/sim_eval_v4_cleanstop/PREREG.md)
has the design, the predictions and the commands.

- **Judge.** A request on `/cmd_vel_in` counts as still only at |v| ≤ 0.005 m/s (`dock_judge`
  `request_still_v`; it was 0.02 m/s, the odometry limit). Creep of 0.012–0.019 m/s, which the
  v3 servo and BC ended on, no longer passes as a stop.
- **Servo.** A stop latch with hysteresis: v is exactly 0 once the range estimate is ≤ 0.27 m
  and stays 0 until it is above 0.30 m (`servo_baseline` `stop_hysteresis` 0.03 m), so lidar
  noise cannot toggle the stop and its successful demonstrations end on a commanded stop.
- **Tools.** `scripts/sim_e2e.sh --hfov 1.0856` records demonstrations through the robot's
  lens; `SIM_CPUSET=0-2` pins the simulation container of `sim_e2e.sh` and `sim_eval.sh`.
- **Analysis.** `stop_labels.py bags` writes the largest request of each demonstration's last
  second unrounded, so an exact stop can be read from the CSV. The controllers also send zero
  on stale input, and that stop is not the controller's decision. `sim_eval_summary.py`
  therefore counts "stop commanded" on the controller's own requests, without the policy
  runner's fallback zeros (exactly (0, 0), sent on a missing or too old frame, as on the tick
  that follows at once an inference that overran its 100 ms period). It flags successes whose
  last second had fewer than 5 camera frames or fresh policy inferences, or any fallback zero.
- **Run checks.** `sim_eval.sh` now ends with a nonzero status when a report fails (the trials
  are kept; rerun with `--report-only`), and its provenance counts the paths that differ from
  the commit without the run's own result files.
- **The numbers above used the old rules.** The clean run, v2 and v3 are reproduced with
  `request_still_v: 0.02` and `stop_hysteresis: 0.0` (e.g. in `config/sim.yaml`).

### Clean-stop protocol: results (v4)

> **SIMULATION ONLY.** `fake_robot` in `ros:jazzy-ros-base` containers on an x86 CPU, 29 October
> 2026. Nothing was measured on autoJetsonBot.

The run followed the pre-registered commands once each, from the pre-registration commit
(b72ba83), with no infrastructure failure. Its one deviation: the write-up file was saved after
the run, with the review's edits listed in its [run log](results/sim_eval_v4_cleanstop/RUN_LOG.md).
The write-up is in
[results/sim_eval_v4_cleanstop/SUMMARY.md](results/sim_eval_v4_cleanstop/SUMMARY.md).

- **Demonstrations.** 40/40 succeeded through the simulator's 80° lens (set A) and 39/40 through
  the robot's 62.2° lens (set B; in `fake_robot` only the pinhole camera's HFOV changes). Every
  success requested exactly v = 0 for its whole last second. Full-stop frames at the end make up
  7.1 % of set A (v3: 2.5 %).
- **Models.** `bc_v4` was trained on set A (sha256 `ed36c9b6…`) and `bc_v4_62` on set B
  (`ea29531c…`), each with v3's BC v2 training command unchanged (no stop weighting).

| Condition | servo | bc_v4 | bc_v4_62 | bc_v4 vs servo, b / c, exact p | Stop commanded, bc_v4 | BC successes flagged stale |
|---|---:|---:|---:|---:|---:|---:|
| clean_ref | 10/10 | 9/10 | — | 1 / 0, p = 1 | 9/10 | 1 |
| noise | 30/30 | 27/30 | — | 3 / 0, p = 0.25 | 27/30 | 10 |
| wide | 27/30 | 16/30 | — | 11 / 0, p = 0.001 | 18/30 | 3 |
| hfov62 | 30/30 | 0/30 | 23/30 | 30 / 0, p = 1.9e-9 | 30/30, short of the band | 0 (bc_v4), 7 (bc_v4_62) |

In `clean_ref` and `noise`, the `bc_v4` trials that commanded a stop are exactly its successes
(a success requires it); none of its 4 failures there, all on pose 5, did.

What v4 shows:

- **All five pre-registered predictions were met as scored**, which is how the
  pre-registration reads them. With the 21 BC successes that the validity check flags counted as
  failures, P4 (`bc_v4` 17/30 in `noise`, below 24; C1 there 13 / 0) and P5 (`bc_v4_62` 16/30,
  below 20) would not be met (see Timing).
- **In distribution, BC brings its request below 0.005 m/s, but not to zero.**
  - It commanded a stop in 36 of 40 `clean_ref` + `noise` trials, at a median ground-truth
    lidar range of 0.264 m.
  - It scored 9/10 and 27/30 against the servo's 10/10 and 30/30. That difference is not
    significant, but with this little power it is not evidence of equivalence either. With
    flagged successes counted as failures, `noise` is 17/30 against 30/30.
  - Its stop is a request just under the judge's limit: the largest request of each success's
    last second was 0.003–0.005 m/s (exploratory).
- **From out-of-distribution poses, BC is worse:** 16/30 against 27/30 in `wide` (11 / 0,
  p = 0.001 per slot; the 11 pairs come from 5 of the 10 poses).
- **Lens.**
  - Trained at 80°, BC stops short under the robot's lens (0.349 m; 0/30).
  - Trained through that lens, it no longer stops short (zone reached 30/30) and scores 23/30
    (against `bc_v4`: b / c = 0 / 23, p = 2.4e-7). That is still below the servo's 30/30
    (7 / 0, p = 0.016; all 7 on poses 1, 5 and 7).
  - The two BC arms ran in different sessions and trained on 32 and 30 episodes.
- **Timing.**
  - Policy inference was 2–3 times slower than in v3 (p50 60–69 ms, p99 116–123 ms), for
    reasons not investigated. Consistent with that, 37 of 130 policy trials had a fallback zero
    in their last second, and 21 BC successes are flagged stale.
  - Counted as failures, they would leave `bc_v4` at 17/30 in `noise` and `bc_v4_62` at 16/30.
  - An exploratory check made after seeing the data found that in all 21, the policy's own
    requests already met the still rule. Its scripts are in
    [results/sim_eval_v4_cleanstop/exploratory](results/sim_eval_v4_cleanstop/exploratory/);
    the bags they read are not in git.

What it does not show (details in the write-up):

- Anything about the real robot.
- Independent trials: the 3 rounds of a pose are correlated, so the CIs are too narrow and the
  p values too small, and no correction is applied across the 6 tests.
- Same-session pairs or controlled load for the `bc_v4_62` comparisons: that arm ran in a
  session of its own, and timing was not controlled.
- Training variance: there is one training seed per model.
- A like-for-like comparison with the tables above: v4 counts a request as still only at
  |v| ≤ 0.005 m/s and the servo latches its stop; the clean run, v2 and v3 used 0.02 m/s and no
  latch.

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

- **Containers.** 269 tests on each of Foxy, Humble and Jazzy (with the v4 stop rules), with 0
  failures and 6 skips. The skipped tests are the ONNX Runtime and TorchScript backend checks;
  the base images have neither package.
- **Host.** 377 passed and 5 skipped. The skipped tests need rclpy or `sensor_msgs`.

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
- **Before v4, no BC model here reliably commanded a stop inside the docking band; v4's gets its
  request just under the 0.005 m/s threshold, not to zero.**
  - BC v1's demonstrations were recorded with the earlier, odometry-only judge and hold no
    stop examples.
  - The v3 re-recording, under the judge that also checks the request, still ends mostly on
    creep. The judge's 0.02 m/s "still" threshold is above the servo's slowest creep, and lidar
    noise toggles its stop.
  - BC v2 and the stop-weighted BC therefore pass the judge by creeping. Under the 62.2° lens
    they stop short of the band (results/sim_eval_v3_bcfix).
  - Demonstrations that end on a commanded stop need a stricter demo-ending rule or a servo
    with hysteresis on its stop range. The v4 protocol adds both (results/sim_eval_v4_cleanstop,
    simulation only). BC trained on its demonstrations requests less than 0.005 m/s at the end
    of 36 of 40 in-distribution trials, but a few mm/s or less, not zero.
  - From out-of-distribution poses it succeeds in 16/30 (the servo: 27/30), and under a lens it
    was not trained on it stops short (0/30).
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
