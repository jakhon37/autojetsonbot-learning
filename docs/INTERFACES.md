# Interfaces (the contract every component follows)

Project: teleop → LeRobot dataset → learned policy → on-robot deployment and paired real
trials on autoJetsonBot (github.com/jakhon37/autoJetsonBot). Task: **camera-only docking**.
The robot starts from one of 10 marked floor poses and must drive to a box with an ArUco
marker on its front face and stop squarely in front of it.

## Robot facts (from autoJetsonBot URDF and code)

| Item | Value |
|---|---|
| ROS 2 on the robot | Foxy (Ubuntu 20.04 container, Python 3.8, OpenCV 4.2) on Jetson Nano B01 |
| Drive | diff drive, wheel separation 0.212 m, radius 0.034 m; `diffdrive_node` subscribes `/cmd_vel` (Twist) |
| Chassis | 0.194 m long; front bumper about 0.097 m ahead of base_link |
| Lidar | RPLidar A1, `/scan` (sensor_msgs/LaserScan), frame `laser_frame` at x=0.064, z=0.161, yaw 0 in URDF, 360 samples, ~10 Hz, **minimum range 0.15 m**: closer returns come back as 0, inf or below `range_min` and mean "too close / unknown" |
| Camera | Raspberry Pi Camera v2 via OpenCV `VideoCapture(0)`, 640×480, HFOV ≈ 80° (1.396 rad), mounted at the front, z ≈ 0.05 m above the chassis |
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
| `/cmd_vel_in` | geometry_msgs/Twist | teleop (keyboard or web GUI) **or** `policy_runner` **or** `servo_baseline` | requested motion; only one source active at a time |
| `/cmd_vel` | geometry_msgs/Twist | `safety_gate` (C++) | the only thing the motors obey |
| `/safety_gate/status` | std_msgs/String (JSON) | `safety_gate` | 2 Hz; schema below |
| `/dock_judge/status` | std_msgs/String (JSON) | `dock_judge` | 10 Hz; schema below |
| `/episode/event` | std_msgs/String | `episode` CLI / `trial_runner` | `start <method> <pose_id> <round>`, `end <success\|fail\|abort> [reason]` |
| `/policy/latency` | std_msgs/String (JSON) | `policy_runner` | per-inference timings |

### `/safety_gate/status` JSON
`{"state": "ok"|"slow"|"stop_obstacle"|"stop_cmd_timeout"|"stop_scan_timeout", "front_min_m": float|null, "scale": float, "period_ms": {"p50":..,"p99":..,"max":..}, "missed_deadlines": int, "ticks": int, "cb_latency_us": {"p50":..,"p99":..}}`

### `/dock_judge/status` JSON
`{"t": float, "marker_visible": bool, "marker_id": int|null, "bearing_deg": float|null, "marker_area_frac": float|null, "front_range_m": float|null, "speed_v": float, "speed_w": float, "in_band": bool, "aligned": bool, "stopped": bool, "success": bool, "success_hold_s": float}`

## Safety gate (C++, `jbl_safety`)

Inputs `/cmd_vel_in`, `/scan`; output `/cmd_vel` on a 50 Hz wall timer. Parameters and defaults:

| Param | Default | Meaning |
|---|---|---|
| `rate_hz` | 50.0 | output timer |
| `forward_angle` | 0.0 | scan angle (rad) that points forward |
| `sector_half_angle` | 0.35 | ± rad around forward used for obstacle check |
| `stop_distance` | 0.17 | lidar range (m) at or below which forward motion is zeroed |
| `slow_distance` | 0.45 | linear speed scales linearly from 1 at `slow_distance` to 0 at `stop_distance` |
| `treat_invalid_as_close` | true | a sector reading of 0 / inf / < range_min after a recent close reading counts as blocked |
| `cmd_timeout` | 0.3 | s without `/cmd_vel_in` → publish zero |
| `scan_timeout` | 0.5 | s without `/scan` → forward motion blocked, rotation and reverse allowed |
| `max_linear` | 0.25 | m/s clamp |
| `max_angular` | 1.5 | rad/s clamp |

Reverse and pure rotation are never blocked by the obstacle check. Timer and scan callbacks
are in separate mutually exclusive callback groups (MultiThreadedExecutor). The node measures
timer period jitter and callback latency and reports them in the status JSON.

## Docking geometry and success (used by `dock_judge`, `servo_baseline`, `fake_robot`)

- Box: at least 0.25 m tall (so the lidar plane at z=0.161 hits it), front face ≥ 0.20 m wide,
  ArUco `DICT_4X4_50` id **7**, 0.10 m side, centred on the front face at camera height.
- Target: lidar front-sector median range **0.25 m** (≈ 0.12 m from bumper to box).
- **Success** = for `success_hold_s` (1.0 s) continuously: front range in **[0.18, 0.32] m**
  (`in_band`), marker visible with |bearing| ≤ **10°** (`aligned`), |v| ≤ 0.02 m/s and |w| ≤ 0.05 rad/s
  (`stopped`, from `/odom`).
- **Fail** = 30 s timeout, an operator intervention, or `safety_gate` state `stop_obstacle`
  for > 2 s while the marker is not aligned (pushed into the wall).
- Bearing = horizontal angle of the marker centre from the image centre using the pinhole
  model with HFOV 1.396 rad (configurable).

## Dataset (LeRobot v3, written by `tools/bag_to_lerobot.py`)

| Feature | dtype / shape | Source |
|---|---|---|
| `observation.images.front` | video, 3×240×320 (H×W), RGB | `/camera/image_raw/compressed`, resized from 640×480 |
| `observation.state` | float32[2] | `/odom` twist (v, w), latest at each tick |
| `action` | float32[2] | `/cmd_vel_in` (v, w), latest operator command at each tick (zero if older than 0.3 s) |

- fps **10**; one ROS bag per episode, or a bag split by `/episode/event`.
- Time base: bag receive time (log time), because the stock camera node does not stamp.
- Task string: `"drive to the box with the marker and stop in front of it"`.
- Per-episode metadata saved alongside (`meta/jbl_episodes.jsonl`): `bag`, `pose_id`, `round`,
  `method`, `outcome`, `duration_s`.
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

- 10 fixed start poses × 3 rounds per method; methods: `servo` (classical baseline), `bc`, `act`.
- Order counterbalanced per round (Latin square) so battery and lighting drift do not favour
  one method.
- Per-trial log (`trials.csv`): `method, pose_id, round, outcome, duration_s, final_range_m,
  final_bearing_deg, interventions, notes`.
- Report: success k/n with Wilson 95% CI per method; exact McNemar on pose×round pairs vs
  `servo`; duration and final-error medians; latency p50/p99 (on-board vs off-board); gate
  jitter.
