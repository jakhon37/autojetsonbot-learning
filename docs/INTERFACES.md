# Interfaces (the contract every component follows)

Project: teleop → LeRobot dataset → learned policy → on-robot deployment and paired real
trials on autoJetsonBot (github.com/jakhon37/autoJetsonBot). Task: **camera-only docking**.
The robot starts from one of 10 marked floor poses and must drive to a box with an ArUco
marker on its front face and stop facing it at docking range (0.18–0.32 m lidar range, marker
within 10° of the image centre; "Docking geometry and success" below).

## Robot facts (from autoJetsonBot URDF and code)

| Item | Value |
|---|---|
| ROS 2 on the robot | Foxy (Ubuntu 20.04 container, Python 3.8, OpenCV 4.2) on Jetson Nano B01 |
| Drive | diff drive, wheel separation 0.212 m, radius 0.034 m; `diffdrive_node` subscribes `/cmd_vel` (Twist) |
| Chassis | 0.194 × 0.178 m. The URDF centres it 0.061 m ahead of base_link, so the front bumper is ≈ 0.158 m ahead of base_link and ≈ 0.094 m ahead of the lidar. `fake_robot` centres it on base_link: bumper 0.097 m ahead of base_link, 0.033 m ahead of the lidar. Measure the real robot |
| Lidar | RPLidar A1, `/scan` (sensor_msgs/LaserScan), frame `laser_frame` at x=0.064, z=0.161, yaw 0 in URDF, 360 samples, ~5.5–10 Hz, **minimum range 0.15 m**: closer returns come back as 0, inf or below `range_min` and mean "too close / unknown"; rplidar_ros also reports empty 1° bins as inf, so scattered invalid beams are normal |
| Camera | Raspberry Pi Camera v2 (CSI, IMX219), stock lens **HFOV 62.2° (1.0856 rad)**, published at 640×480, mounted at the front (URDF x = 0.1455, z ≈ 0.05 m above the chassis). V4L2 exposes only raw RG10 Bayer, which OpenCV cannot read: capture through `nvarguscamerasrc` in the full-sensor 1640×1232 mode (needs the container started with `--runtime nvidia -v /tmp/argus_socket:/tmp/argus_socket`; `config/robot.yaml`). `fake_robot`'s simulated camera is 80° (1.396 rad, parameter `camera_hfov_rad`) |
| Existing camera node | `jetson_bot_detection/camera_go.py` publishes raw `camera/image_raw` bgr8 at 30 Hz **with no header stamp**; it is replaced by `jbl_nodes/camera_node` for this project |

## Compatibility rules

- ROS 2 packages build and pass tests on **Foxy, Humble and Jazzy** (`scripts/docker_test.sh`).
- ROS Python nodes: **Python 3.8 compatible** (no `X | Y` unions, no `match`, no `list[int]` at
  runtime; use `typing`). Depend only on rclpy, std/sensor/geometry/nav msgs, numpy, OpenCV.
  **No cv_bridge** (not in the base images); convert `sensor_msgs/Image` with numpy.
- OpenCV ArUco: support the old API (`cv2.aruco.detectMarkers`, `Dictionary_get`, OpenCV ≤ 4.6)
  and the new API (`cv2.aruco.ArucoDetector`, `getPredefinedDictionary`, OpenCV ≥ 4.7).
- C++: C++17, rclcpp APIs that exist in Foxy. Pure logic in ROS-free classes with gtest.
- Host tools (`tools/`): Python 3.12 venv with `lerobot==0.6.1`, run off-robot.
- Everything is testable offline: no Hub access, no GPU.

## Topics

| Topic | Type | Producer | Notes |
|---|---|---|---|
| `/camera/image_raw` | sensor_msgs/Image, `bgr8`, 640×480 | `camera_node` | header.stamp = capture time, frame_id `camera_link`; default 30 Hz |
| `/camera/image_raw/compressed` | sensor_msgs/CompressedImage, `jpeg` | `camera_node` | same stamp; default 15 Hz, quality 85. **This is what gets recorded** |
| `/scan` | sensor_msgs/LaserScan | rplidar_ros | forward direction = angle `forward_angle` (param, default 0.0 rad) |
| `/odom` | nav_msgs/Odometry | diffdrive_node | measured v (twist.linear.x) and w (twist.angular.z) |
| `/cmd_vel_in` | geometry_msgs/Twist | teleop (keyboard remapped, or the web GUI with its Command Velocity topic set to `/cmd_vel_in`) **or** `policy_runner` **or** `servo_baseline` | requested motion; only one source active at a time. Also read by `dock_judge` ("stopped") |
| `/cmd_vel` | geometry_msgs/Twist | `safety_gate` (C++) | the only thing the motors obey |
| `/safety_gate/status` | std_msgs/String (JSON) | `safety_gate` | 2 Hz; schema below |
| `/dock_judge/status` | std_msgs/String (JSON) | `dock_judge` | 10 Hz; schema below |
| `/episode/event` | std_msgs/String | `episode` CLI / `dock_judge` (auto_end) | `start <method> <pose_id> <round>`, `end <success\|fail\|abort> [reason]`, `intervention [note]` (operator intervention: the judge fails the running episode, controllers stop, `trial_logger` counts it). Reliable + transient_local |
| `/policy/latency` | std_msgs/String (JSON) | `policy_runner` | per-inference timings |

### `/safety_gate/status` JSON
`{"state": "ok"|"slow"|"stop_obstacle"|"stop_cmd_timeout"|"stop_scan_timeout", "front_min_m": float|null, "scale": float, "period_ms": {"p50":..,"p99":..,"max":..}, "missed_deadlines": int, "ticks": int, "cb_latency_us": {"p50":..,"p99":..}}`

### `/dock_judge/status` JSON
`{"t": float, "marker_visible": bool, "marker_id": int|null, "bearing_deg": float|null, "marker_area_frac": float|null, "front_range_m": float|null, "speed_v": float, "speed_w": float, "in_band": bool, "aligned": bool, "stopped": bool, "success": bool, "success_hold_s": float}`

## Safety gate (C++, `jbl_safety`)

Inputs `/cmd_vel_in`, `/scan`; output `/cmd_vel` on a 50 Hz wall timer. Parameters and defaults
(read-only at runtime):

| Param | Default | Meaning |
|---|---|---|
| `rate_hz` | 50.0 | output timer |
| `forward_angle` | 0.0 | scan angle (rad) that points forward |
| `sector_half_angle` | 0.35 | ± rad around forward used for obstacle check |
| `footprint_half_width` | 0.15 | m; valid returns **outside** the sector that are ahead of the lidar with \|lateral\| ≤ this count at their forward distance (an obstacle ahead of a chassis corner); invalid readings there never count |
| `stop_distance` | 0.17 | lidar range (m) at or below which forward motion is zeroed |
| `slow_distance` | 0.45 | linear speed scales linearly from 1 at `slow_distance` to 0 at `stop_distance` |
| `treat_invalid_as_close` | true | sector readings of 0 / inf / < range_min count as blocked when a close reading (< `slow_distance`) came within the last 0.5 s, there are at least `min_invalid_beams` of them, and the nearest valid sector reading (if any) is within 0.05 m of max(`stop_distance`, range_min); a blocked scan refreshes the 0.5 s memory |
| `min_invalid_beams` | 3 | invalid sector beams needed for the rule above (or every sector beam, if the sector holds fewer): one empty bin does not stop the robot |
| `cmd_timeout` | 0.3 | s without `/cmd_vel_in` → publish zero |
| `scan_timeout` | 0.5 | s without `/scan` → forward motion blocked, rotation and reverse allowed |
| `max_linear` | 0.25 | m/s clamp |
| `max_angular` | 1.5 | rad/s clamp |

Reverse and pure rotation are never blocked by the obstacle check. Timer and scan callbacks
are in separate mutually exclusive callback groups (MultiThreadedExecutor). The node measures
timer period jitter and callback latency and reports them in the status JSON. On SIGINT,
SIGTERM, SIGHUP or SIGQUIT it publishes a zero twist before exiting; a SIGKILL or crash cannot,
and `diffdrive_node` has no command watchdog, so the motor-side timeout must be verified
(PROTOCOL section 4).

## Docking geometry and success (used by `dock_judge`, `servo_baseline`, `fake_robot`)

- Box: at least 0.25 m tall (so the lidar plane at z=0.161 hits it), front face ≥ 0.20 m wide,
  ArUco `DICT_4X4_50` id **7**, 0.10 m side, centred on the front face at camera height.
- Target: lidar front-sector median range **0.25 m**: ≈ 0.22 m from bumper to box with the
  chassis of `fake_robot` (bumper 0.033 m ahead of the lidar), ≈ 0.16 m with the URDF's
  (bumper 0.094 m ahead of the lidar).
- **Success** = for `success_hold_s` (1.0 s) continuously: front range in **[0.18, 0.32] m**
  (`in_band`), marker visible with |bearing| ≤ **10°** (`aligned`), and `stopped`: |v| ≤ 0.02 m/s
  and |w| ≤ 0.05 rad/s from `/odom` **and** the latest `/cmd_vel_in` within the same limits
  (fresh within 0.3 s, the gate's `cmd_timeout`; no fresh command counts as zero). A robot the
  gate's slow zone has merely slowed below 0.02 m/s while the controller still asks to drive is
  not stopped.
- The rule does not check the robot's heading to the box face or its sideways offset: a robot
  that drove straight at the marker from an oblique start passes while still 20–30° off square.
- **Fail** = 30 s timeout, an operator intervention, or `safety_gate` state `stop_obstacle`
  for > 2 s while the marker is not aligned (pushed into the wall).
- Bearing = horizontal angle of the marker centre from the image centre using the pinhole
  model with HFOV `hfov_rad`: 1.0856 rad on the robot (`config/robot.yaml`, stock Pi Camera v2
  lens; calibrate it, PROTOCOL section 5), 1.396 rad in simulation (`config/sim.yaml`, equal to
  `fake_robot`'s `camera_hfov_rad`).

## Dataset (LeRobot v3, written by `tools/bag_to_lerobot.py`)

| Feature | dtype / shape | Source |
|---|---|---|
| `observation.images.front` | video, 3×240×320 (H×W), RGB | `/camera/image_raw/compressed`, resized from 640×480 |
| `observation.state` | float32[2] | `/odom` twist (v, w), latest at each tick |
| `action` | float32[2] | `/cmd_vel_in` (v, w), latest operator command at each tick (zero if older than 0.3 s) |

- fps **10**; one ROS bag per episode, or a bag split by `/episode/event`.
- Time base: bag receive time (log time), because the stock camera node does not stamp.
- Task string: `"drive to the box with the marker and stop in front of it"`.
- Leading ticks whose action is zero (the robot standing at the start before the first command)
  are dropped (`--keep-idle-start` keeps them).
- Per-episode metadata saved alongside (`meta/jbl_episodes.jsonl`): `bag`, `pose_id`, `round`,
  `method`, `outcome`, `duration_s`, `idle_start_s` (the dropped idle start).
- Works for Foxy bags (sqlite3, `.db3`) and Jazzy bags (mcap) via the `rosbags` Python package.

## Policy I/O (BC and ACT; exported to ONNX)

- Input `image`: float32 [1,3,224,224], RGB, ImageNet mean/std normalised, resized from the
  240×320 frame by center-crop-free resize (squash).
- Output `action`: float32 [1,2] in **normalised** units, mapped to (v, w) with
  v = a0 × 0.25 m/s, w = a1 × 1.5 rad/s, then clipped. Stored in `policy_meta.json` beside the
  ONNX file together with input size, normalisation and dataset stats.
- `policy_runner` publishes `/cmd_vel_in` at 10 Hz from the latest image; drops stale images
  (> 0.2 s); backends: `onnxruntime`, `tensorrt` (TensorRT 8.x on Jetson), `torch`, `constant`
  (for tests).

## Evaluation protocol (`tools/analyze_trials.py`)

- 10 fixed start poses × 3 rounds per method; methods: `servo` (classical baseline), `bc`
  (`act` once it can be exported to this contract).
- Order counterbalanced per round (Latin square) so battery and lighting drift do not favour
  one method.
- Per-trial log (`trials.csv`): `method, pose_id, round, outcome, duration_s, final_range_m,
  final_bearing_deg, interventions, notes`.
- Report: success k/n with Wilson 95% CI per method; exact McNemar on pose×round pairs vs
  `servo` with a Newcombe (1998, method 10) CI for the paired difference; duration, final range
  error and final marker bearing medians; latency p50/p99 (on-board vs off-board); gate jitter.
