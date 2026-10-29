# Real-robot protocol: setup, demos, paired trials, reporting

The runbook for the camera-only docking task on autoJetsonBot. Names, topics, thresholds and
the success rule come from [INTERFACES.md](INTERFACES.md); if the two disagree, INTERFACES.md
wins. Every command runs inside the robot's Foxy container (`./robot.sh shell` in
autoJetsonBot, recreated once with the camera flags of section 4) with this workspace sourced,
unless it says "Jetson host" or "host" (the training machine: the Python 3.12 venv with
`requirements-tools.txt`).

Order of work: build the scene (1–3), calibrate (4–5), record demos (6), train off-robot, freeze
everything (7), run the scheduled trials (8–10), analyse and report (11–12).

## 1. Marker

1. Host: `python tools/make_marker.py --out-dir markers` writes
   `markers/aruco_4x4_50_id7_100mm.pdf` (and a PNG): ArUco `DICT_4X4_50`, id **7**, 100 mm
   black edge, white quiet zone, 50 mm scale bar.
2. Print the PDF at **100 % / actual size** (no fit-to-page) on matte paper; gloss throws
   glare into the camera. Measure: the scale bar is 50 ± 0.5 mm and the black square
   100 ± 1 mm, else reprint.
3. Glue it flat to stiff card or foam board (no curl, no bubbles). Cut along the grey line; do
   not trim into the white quiet zone.

## 2. Box

- Rigid, matte, light coloured (the RPLidar returns little from black, glossy or transparent
  surfaces), weighted (about 2 kg inside) so a bump does not move it.
- Height at least 0.25 m **and** at least 5 cm above the lidar's scan plane. Measure the scan
  plane on your robot: the URDF puts the laser 0.161 m above `base_link` (≈ 0.22 m above the
  floor), the README says 0.161 m above the floor.
- Front face flat, at least 0.20 m wide (0.30 m is easier).
- Marker centred horizontally on the front face, its centre at the **camera lens height**
  (measure it; ≈ 0.10–0.11 m above the floor from the URDF), within ±2 cm, so the whole marker
  stays in view when docked with the lens only ≈ 0.16 m from the face.
- Free space: at least 0.5 m clear on both sides of the box and the full start area (section 3)
  empty. Mark the box footprint on the floor (tape at its two front corners) so it goes back to
  exactly the same place if it is bumped.

## 3. Floor layout and the 10 start poses

`ros2_ws/src/jbl_nodes/config/start_poses.yaml` is the source of truth (fake_robot uses the
same file). Poses are relative to the box front face:

- **distance**: from the face plane back to the robot centre (`base_link`, the midpoint between
  the drive wheels), along the face normal;
- **lateral**: sideways from the marker centre line, **positive = left** when you look at the box;
- **heading**: robot yaw relative to facing the box squarely, **positive = turned left** (CCW
  seen from above).

Mark the floor:

1. Tape the **face line** along the bottom of the box front face.
2. Tape the **centre line** perpendicular to it through the marker centre, out to 1.8 m (use a
   carpenter's square or a 3-4-5 triangle).
3. For each pose put a small cross at **A** (where `base_link` goes) and a dot at **B**, 0.30 m
   ahead of A along the start heading. Use thin, low-contrast marks (6 mm masking tape or pencil):
   the camera sees the floor, and the marks must be identical for demos and every method.

Generated from the YAML (host: `python tools/make_schedule.py --pose-table
ros2_ws/src/jbl_nodes/config/start_poses.yaml`; regenerate if the YAML changes):

| Pose | A: distance m | A: lateral m | Heading deg | B: distance m | B: lateral m |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | +0.00 | 0 | 0.70 | +0.00 |
| 1 | 1.20 | +0.30 | 0 | 0.90 | +0.30 |
| 2 | 1.20 | -0.30 | 0 | 0.90 | -0.30 |
| 3 | 0.80 | +0.25 | -20 | 0.52 | +0.15 |
| 4 | 0.90 | -0.40 | +30 | 0.64 | -0.25 |
| 5 | 0.85 | +0.50 | +40 | 0.62 | +0.69 |
| 6 | 1.50 | -0.50 | -25 | 1.23 | -0.63 |
| 7 | 1.60 | +0.00 | +40 | 1.37 | +0.19 |
| 8 | 1.10 | +0.10 | -40 | 0.87 | -0.09 |
| 9 | 1.40 | -0.45 | -35 | 1.15 | -0.62 |

With the stock 62.2° Pi Camera v2 lens (±31°), poses 5, 6, 7, 8 and 9 start with the marker
out of view (the YAML comments give the bearings), so the robot has to search. The simulator's
80° camera sees pose 8 at the image edge.

Placing the robot: wheel-axle midpoint over A, robot centreline through B. Two small tape marks
on the chassis top (above the axle centre and at the front centre) make it a sighting job.
Target accuracy ±1 cm, ±2°.

## 4. Bring-up and the lidar `forward_angle`

**Container (once).** autoJetsonBot's stock container cannot reach the CSI camera: the Pi
Camera v2 is read through Argus (`nvarguscamerasrc`, see `config/robot.yaml`), which needs
NVIDIA's container runtime and the host's Argus socket (V4L2's `/dev/video0` is raw RG10 Bayer,
which OpenCV cannot use). On the Jetson host, in the autoJetsonBot checkout, recreate it with
`robot.sh`'s `docker run` line plus those flags (and `--ipc host`, as in its
`docker-compose.yaml`, for the TensorRT container of section 8):

```bash
docker rm -f auto_ros_foxy
docker run -d --name auto_ros_foxy --network host --ipc host --privileged \
  --runtime nvidia -v /tmp/argus_socket:/tmp/argus_socket \
  -v "$(pwd)":/autonomous_ROS auto_ros:foxy bash -c "while true; do sleep 3600; done"
```

Then, inside it, these must pass before anything else:

```bash
gst-inspect-1.0 nvarguscamerasrc | head -n 3      # the L4T GStreamer plugin is there
python3 -c "
import cv2, yaml, subprocess
share = subprocess.check_output(['ros2', 'pkg', 'prefix', 'jbl_nodes']).decode().strip()
dev = yaml.safe_load(open(share + '/share/jbl_nodes/config/robot.yaml'))['camera_node']['ros__parameters']['device']
cap = cv2.VideoCapture(dev, cv2.CAP_GSTREAMER); ok, frame = cap.read(); print('frame', frame.shape if ok else None)"
```

If the plugin is missing (an image without NVIDIA's L4T GStreamer plugins) or no frame comes,
bridge the camera on the Jetson host instead: `nvarguscamerasrc` (1640×1232) → `nvvidconv` →
`v4l2sink` on a `v4l2loopback` device in YUY2, pass that device into the container with
`--device`, and set `camera_node` `device: "/dev/videoN"` in `config/robot.yaml`.
`scripts/record_robot.sh` opens the camera the same way before every recording.

**Data.** Everything recorded goes under this checkout, which is on the `/autonomous_ROS` bind
mount; `/root` inside the container is neither persistent nor visible on the Jetson host. In
every shell that records:

```bash
export D=/autonomous_ROS/autojetsonbot-learning/data/robot   # bags in $D/bags, CSVs in $D/trials
```

**Base, lidar and teleop.** Base and lidar come from the robot's own launch (`./robot.sh robot`:
`diffdrive_node` on `/cmd_vel`, `rplidar_composition` on `/scan`). Nothing else may publish
`/cmd_vel`. The web dashboard's teleop pad publishes `/cmd_vel` by default and so bypasses the
gate: in its sidebar set **Topics → Command Velocity** to `/cmd_vel_in` (the browser saves it),
or change the default once in autoJetsonBot:

```bash
sed -i 's|id="cmdVelTopic" class="config-input" value="/cmd_vel"|id="cmdVelTopic" class="config-input" value="/cmd_vel_in"|' \
  src/jetson_bot_gui/web/index.html
```

(A browser that saved its settings before keeps its own value: check the field.) Do not run
Nav2 during demos or trials; its controller publishes `/cmd_vel` too. With the nodes below
running, `ros2 topic info /cmd_vel` must show one publisher, the gate.

**Nodes.** In one terminal; the launch file loads `config/robot.yaml`, respawns the gate, and
starts `camera_node`, `safety_gate`, `dock_judge` and `trial_logger`:

```bash
export FWD=0.0       # replace with the calibrated value below
export HFOV=1.0856   # stock Pi Camera v2 lens; replace with the calibrated value of section 5
ros2 launch jbl_nodes robot_record.launch.py forward_angle:=$FWD hfov_rad:=$HFOV \
  trials_csv:=$D/trials/calibration.csv
```

Once both values are calibrated, write them into the `/**` section of `config/robot.yaml`, so
every node gets them without arguments and the gate, the judge and the baseline cannot
disagree.

For debugging, the nodes also start one by one. Every `./robot.sh shell` is a new shell: run
the `export` lines in each terminal (or add them to the container's `~/.bashrc`), or the nodes
abort with `Couldn't parse parameter override rule`. This route has no gate respawn.

```bash
ros2 run jbl_nodes camera_node --ros-args \
  --params-file $(ros2 pkg prefix jbl_nodes)/share/jbl_nodes/config/robot.yaml
ros2 run jbl_safety safety_gate_node --ros-args \
  --params-file $(ros2 pkg prefix jbl_safety)/share/jbl_safety/config/safety_gate.yaml \
  -p forward_angle:=${FWD:?set FWD}
ros2 run jbl_nodes dock_judge --ros-args -p forward_angle:=${FWD:?set FWD} -p hfov_rad:=${HFOV:?set HFOV}
```

`forward_angle` is the `/scan` angle that points straight ahead. It must be the same in
`safety_gate_node`, `dock_judge` and `servo_baseline`.

**Motor timeout (once, wheels off the ground).** `diffdrive_node` has no command watchdog. The
gate publishes a zero twist when it is stopped with SIGINT, SIGTERM, SIGHUP or SIGQUIT, but if
it is killed (SIGKILL) or crashes the motors keep the last `/cmd_vel`. The launch respawns the
gate, yet on Foxy `diffdrive_node` hears the new one only after DDS rediscovery, measured at
2.5–3.6 s after the kill. Only a motor-side timeout bounds this (the ESP32 firmware's 500 ms
`m` command timeout), so check it: robot on a stand, drive forward through the gate (teleop on
`/cmd_vel_in`), then in another terminal

```bash
pkill -9 -f safety_gate_node
```

The wheels must stop within about 0.5 s. If they keep turning, run no demos or trials until
`diffdrive_node` has a watchdog. The change to autoJetsonBot's
`jetson_bot_diffdrive/diffdrive_node.py`:

- in `__init__`: `self.last_cmd = None`
- first line of `cmd_vel_callback`: `self.last_cmd = time.monotonic()`
- in `update_loop`, after the serial check: `if self.last_cmd is not None and
  time.monotonic() - self.last_cmd > 0.25: self.ser.write(b"m 0.000 0.000\r");
  self.ser.flush(); self.last_cmd = None`

**`forward_angle`:**

1. `ros2 topic hz /scan` should read 5–10 Hz.
2. Put the box squarely in front of the robot on its centreline, front face 0.40 m from the
   bumper, nothing else within 1 m.
3. Find the angle of the closest valid return (the foot of the perpendicular = straight ahead).
   This snippet averages 20 scans and handles either scan direction (tested on Foxy):

   ```bash
   python3 - <<'EOF'
   import math, rclpy
   from rclpy.qos import qos_profile_sensor_data
   from sensor_msgs.msg import LaserScan

   def closest(s):
       valid = [(r, s.angle_min + i * s.angle_increment) for i, r in enumerate(s.ranges)
                if math.isfinite(r) and s.range_min <= r <= s.range_max]
       return min(valid) if valid else None

   rclpy.init(); node = rclpy.create_node("find_forward_angle"); scans = []
   node.create_subscription(LaserScan, "/scan", scans.append, qos_profile_sensor_data)
   while len(scans) < 20:
       rclpy.spin_once(node, timeout_sec=1.0)
   s = scans[-1]
   print("angle_min %.3f  angle_increment %.5f  range_min %.2f" % (s.angle_min, s.angle_increment, s.range_min))
   hits = [h for h in map(closest, scans) if h]
   a = math.atan2(sum(math.sin(t) for _, t in hits), sum(math.cos(t) for _, t in hits))
   print("closest return %.3f m; forward_angle = %.3f rad (%.1f deg)" % (min(hits)[0], a, math.degrees(a)))
   EOF
   ```

4. Set `FWD` to that angle (round to 0.01 rad; one beam is 0.017 rad) and restart the launch.
5. Check `ros2 topic echo --full-length /dock_judge/status`: `front_range_m` = 0.40 m + your
   measured bumper-to-lidar-centre distance (≈ 0.09 m by the URDF; INTERFACES.md lists both
   chassis figures), within ±0.02 m. Write the measured distance down: it converts every lidar
   range below into a bumper distance. Check
   `ros2 topic echo --full-length /safety_gate/status`: `front_min_m` about the same. (The status
   topics are JSON strings; without `--full-length` Foxy cuts them at 128 characters.)
6. Slide the box 0.4 m to the robot's left: `front_range_m` must jump to the far background or
   `null`. Put it back.
7. Stop test: from 0.40 m, drive slowly straight at the box with teleop (section 6) and keep
   commanding forward. The gate status goes `slow` below a lidar range of 0.45 m and
   `stop_obstacle` at a lidar range of 0.17–0.22 m, where the robot stops (at 0.22 m or closer,
   three or more no-return beams in the sector already count as blocked; that is the bumper
   about 0.08–0.13 m from the box with the URDF's chassis): `ros2 topic echo /cmd_vel` shows
   `linear.x: 0.0` while `/cmd_vel_in` is still positive. Reverse and rotation must still pass.
   Approach rather than placing the box that close by hand: the lidar cannot see closer than
   0.15 m, and the gate treats those blind readings as blocked only after a recent close
   reading. Repeat with the box 0.12 m to one side of the centreline (its edge in front of a
   wheel, outside the ±20° sector): the gate's ±0.15 m forward corridor must slow and stop the
   robot too.

## 5. Camera HFOV and mount check

The bearing the judge and the servo baseline use is `atan((cx - u) / f)` with
`f = (W / 2) / tan(HFOV / 2)`, positive = marker to the left. INTERFACES.md and
`config/robot.yaml` assume the stock Raspberry Pi Camera v2 lens, 62.2° (1.0856 rad); another
lens, or a cropped sensor mode, changes it, so measure it. (The simulator's camera is 80°,
1.396 rad.)

1. Robot on the centre line, squarely facing the box, camera lens Z = 0.60 m from the face.
2. Read the bearing a few times:
   `ros2 topic echo --full-length /dock_judge/status | grep -o '"bearing_deg": [^,]*'`.
   It must be within ±1.5°. Otherwise the camera is yawed on its mount (or the robot is not
   square): fix the mount, because a 2° offset eats a fifth of the ±10° alignment tolerance.
   A positive reading with the marker on the robot's right means the image is mirrored: fix that
   first.
3. Move the robot sideways (heading unchanged) to lateral X = +0.20 m: the marker is now to the
   right and the true bearing is −atan(0.20 / 0.60) = −18.4°. Read the bearing b (deg).
4. Corrected HFOV (rad) from the reading b made with the assumed HFOV `h0` the judge was
   started with:

   ```bash
   python3 -c "import math; Z,X,b,h0=0.60,0.20,-18.4,1.0856; \
   print(2*math.atan(math.tan(h0/2)*(X/Z)/math.tan(math.radians(abs(b)))))"
   ```

   With the stock lens and the 1.0856 rad default the reading is about −18.4° and this prints
   ≈ 1.086. A reading of −13.5° would mean an 80° lens (1.394 rad). With h0 = 1.396 a 62.2°
   lens reads −24.9° and the formula gives 1.085 rad (checked against the judge's own detector
   on rendered frames).
5. Repeat at X = −0.20 m, average the two, write the result into `hfov_rad` in the `/**`
   section of `config/robot.yaml` (or `export HFOV=...`), restart the launch, and re-check: both
   readings within ±1° of ±18.4°.
6. Docked check: put the robot squarely at lidar range 0.25 m on the centre line. The status
   must show `marker_visible: true`, `marker_id: 7`, `in_band`, `aligned` and, once still,
   `stopped` and `success: true` after 1 s. If the marker is not visible here, first check that
   `camera_node` uses the 1640×1232 pipeline of `config/robot.yaml`: a cropped Argus sensor mode
   (1280×720, 1920×1080) narrows both HFOV and VFOV and cuts the marker off when docked. Then
   fix the marker height (section 2) before recording anything.

## 6. Teleop demonstrations

How many: **80** episodes (50 is the minimum worth training ACT on), from **20 training spots**
× 4. Training spots are scattered over the start region (distance 0.7–1.7 m, lateral within
±0.55 m, heading within ±45°) with base_link at least 10 cm from every test pose A whatever the
heading (the same rule as `config/sim_demo_poses.yaml`), so the test poses stay held out: a spot
on a test pose's floor mark with only the heading changed would put the test start view in the
training data. Give each spot an id 10–29 (0–9 are the test poses) and note them.

How to vary them:

- about a quarter start with the marker out of view (heading turned away), so the policy learns
  to search;
- the lighting and clutter the trials will see; if the trial room lighting varies over a day,
  record demos across the same range;
- one consistent driving style: turn towards the marker, approach at roughly constant speed,
  slow down, stop in the band facing the marker and hold still. ("Square up" is a driving-style
  instruction for the demos; the success rule does not measure squareness.) No showing off, no
  oscillation, as little reversing as possible. Speeds similar to the baseline (≈ 0.15 m/s,
  ≈ 0.6 rad/s).

Start the nodes of section 4 (`dock_judge` with `auto_end` true, the default) and the recorder
**before** the first episode. `bash scripts/record_robot.sh` does both: it checks the bring-up
and the camera, then runs `robot_record.launch.py` with a bag of the INTERFACES topics in
`$D/bags/demos_<date>_<time>` and the episode log in `$D/trials/<date>/demos.csv`. By hand:

```bash
ros2 launch jbl_nodes robot_record.launch.py bag:=$D/bags/demos_$(date +%Y%m%d_%H%M) \
  trials_csv:=$D/trials/$(date +%Y-%m-%d)/demos.csv
```

Drive with something that publishes `/cmd_vel_in` continuously while you command motion: the
gate zeroes its output 0.3 s after the last command.

- **Web dashboard (recommended)**, with Command Velocity set to `/cmd_vel_in` (section 4). It
  publishes at its Publish Rate (10 Hz) while a direction key or button is held and a zero
  twist on release. Set Max Linear Velocity to 0.15 and Max Angular Velocity to 0.6 (speed
  sliders at 1).
- **Joystick** (`apt install ros-foxy-joy ros-foxy-teleop-twist-joy`):
  `ros2 run joy joy_node --ros-args -p autorepeat_rate:=20.0` and
  `ros2 run teleop_twist_joy teleop_node --ros-args -r cmd_vel:=/cmd_vel_in` (with its
  enable-button and scale parameters set for your pad); `autorepeat_rate` keeps it publishing
  while the stick is held.
- **Keyboard** (`apt install ros-foxy-teleop-twist-keyboard` if it is missing; it needs a real
  terminal, so use `docker exec -it`):

  ```bash
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in
  ```

  `i` forward, `j`/`l` turn, `u`/`o` arcs, `,` back, `k` stop. It starts at 0.5 m/s and
  1.0 rad/s: press `x` 12 times (linear ×0.9 each, to ≈ 0.14 m/s) and `c` 5 times (angular to
  ≈ 0.6 rad/s). It publishes once per key press, so it moves the robot only while the key
  auto-repeats: set the key-repeat delay on the machine you type on to ≤ 250 ms
  (`xset r rate 200 30` on X11; macOS: Keyboard → Delay until repeat: shortest), or the robot
  stutters at the start of every key press.

Per episode:

1. Place the robot on the spot. Hands off.
2. `ros2 run jbl_nodes episode --timeout 5 start teleop <spot_id> <k>` (`k` = 1–4, the repeat at
   that spot). Exit status 0 means it was delivered; on 1, run it again.
3. Drive to the box, stop in the band facing the marker, and stop commanding (release the key,
   button or stick). The judge counts the robot as stopped only when `/cmd_vel_in` is still
   too (zero, or nothing for 0.3 s), so every successful demo ends with at least 1 s of
   standing still with a zero command: that is what teaches a policy to stop. `dock_judge`
   ends the episode itself with `end success` after the success conditions hold for 1 s, or
   `end fail timeout` at 30 s. Watch `ros2 topic echo /episode/event` in another terminal.
4. A demo you would not want imitated (bump, wrong turn, long dithering) is ended with
   `ros2 run jbl_nodes episode --timeout 5 end abort <reason>` before the judge ends it (an
   abort is dropped by `bag_to_lerobot.py`; train on successful teleop demos only:
   `train_bc.py --methods teleop --outcomes success --split-by pose_id`).

Leave the bag running for the whole session; `bag_to_lerobot.py` splits it on `/episode/event`.
Disk use is ≈ 2 GB per hour, mostly the 15 Hz JPEG stream.

## 7. Freeze before the trials

Before trial 1, write into a session log (and commit) what is being tested; nothing below may
change until the last trial:

- git commit of this repo; sha256 of each model file (ONNX / TensorRT engine) and its
  `policy_meta.json`;
- `FWD`, `HFOV`, all non-default parameters, the success thresholds;
- the schedule (host):

  ```bash
  python tools/make_schedule.py --methods servo,bc --poses 10 --rounds 3 --seed 20261102 \
    --out trials/schedule.csv --markdown trials/schedule.md
  ```

  Add `act` only once an exporter turns the ACT checkpoint into the image → [1, 2] contract
  (README, Limitations); until then it cannot be run, so it cannot be scheduled.

  The default order is `interleaved`: each pose is a block in which every method runs back to
  back, the method order rotates through a Latin square across blocks and rounds (each method
  goes first and second equally often), and the pose order is reshuffled per round with a
  fixed seed. That keeps the two halves of every McNemar pair minutes apart, so battery and
  lighting drift cannot favour one method. Use `--order blocked` only if the Nano cannot run all
  policies at once, and say so in the report. Print `schedule.md` as the run sheet.

If something must change mid-evaluation (a bug), stop, fix, and start again from round 1 with a
new seed; report both runs.

## 8. Trial setup (each session)

In the robot container (section 4: `FWD`, `HFOV` and `D` exported, or the calibration in
`config/robot.yaml`), camera, gate (respawned), judge, logger, the servo baseline and a bag of
the INTERFACES topics (including `/policy/latency`) in one command:

```bash
ros2 launch jbl_nodes robot_record.launch.py with_servo:=true forward_angle:=$FWD hfov_rad:=$HFOV \
  trials_csv:=$D/trials/2026-11-02/trials.csv bag:=$D/bags/trials_$(date +%Y%m%d_%H%M)
```

The BC policy with TensorRT runs in the L4T TensorRT container of the README ("Deployment",
on-board), not in `./robot.sh shell`: the Foxy container's Python 3.8 cannot import JetPack
4.6's TensorRT bindings. There, with this workspace built:

```bash
ros2 run jbl_nodes policy_runner --ros-args -r __node:=policy_bc \
  -p method_filter:=bc -p backend:=tensorrt -p model_path:=<bc engine>
```

(An `act` runner joins here only once ACT can be exported; section 7.) Each method drives only
after a `start` with its own name and stops on any `end` or `intervention`. Close the
teleop terminal during trials. For off-board inference (policy_runner on a laptop over Wi-Fi),
the laptop clock must be synchronised with the robot (chrony offset < 5 ms), otherwise frames
look stale and the runner publishes zeros; label those results "off-board".

Checklist before round 1 and after every break: `ros2 node list` shows all nodes;
`ros2 topic hz /camera/image_raw/compressed` ≈ 15, `/scan` 5–10, `/odom` steady; docked check of
section 5 passes; battery ≥ 12.0 V (3S pack); bag and video recording running; the trial logger
prints the CSV path.

## 9. Success definition (from INTERFACES.md, judged by `dock_judge`)

**Success**: for 1.0 s continuously, lidar front range in [0.18, 0.32] m, marker id 7 visible
with |bearing| ≤ 10°, and stopped: |v| ≤ 0.02 m/s and |w| ≤ 0.05 rad/s from `/odom` **and** the
latest `/cmd_vel_in` (fresh within 0.3 s; none counts as zero) with |v| ≤ 0.005 m/s
(`request_still_v`) and |w| ≤ 0.05 rad/s, so a method still asking to drive, even at a crawl
the gate's slow zone holds back, is not stopped. The rule does not measure the robot's heading
to the box face or its sideways offset.
**Fail**: 30 s without success, any operator intervention, or the safety gate in
`stop_obstacle` for more than 2 s while the marker is not aligned. The operator never overrules
the judge; a disagreement goes into the notes and the report.

## 10. Running the trials

For each row of the run sheet, in order, never skipping or reordering:

1. Hold the trial card (trial #, method, pose, round) up to the video camera for 2 s.
2. Place the robot on pose A/B. Step back behind the start area, out of the robot camera's view.
3. `ros2 run jbl_nodes episode --timeout 5 start <method> <pose_id> <round>` (copy the event
   from the sheet). Exit status 1 = not delivered: run it again.
4. Hands off until `/episode/event` shows the judge's `end ...`. Every `end` disables the
   method (a method that never asks to stop is held still by the gate and the episode ends as a
   timeout).
5. Write the result and the video time on the sheet; `tail -1` the CSV to check the row.
6. Lift the robot back for the next row.

Between rounds: recharge to ≥ 12.0 V and write the voltage on the sheet. If the pack drops below
11.4 V mid-round, finish the current pose block, then recharge.

**Interventions** are for safety only: the robot is about to hit something other than the box
face at speed, leave the area, snag a cable, or a person enters the area. Any touch of the
robot, the box or the scene, and any command sent during a trial, is an intervention and a fail.
Never help a run along. To intervene, stop the robot (lift it, or kill motor power), then:

```bash
ros2 run jbl_nodes episode --timeout 5 intervention <what happened>
```

The running method stops at once; `dock_judge` fails the episode and publishes
`end fail intervention`; `trial_logger` counts the intervention and adds your note to the row
(unless the judge's end overtakes it, which can happen as they come from different writers:
write the note on the run sheet too).
Where no judge runs (or `auto_end` is off), end it yourself with
`ros2 run jbl_nodes episode --timeout 5 end fail intervention <what happened>`, which the
logger also counts. (Do not use `ros2 topic pub --once` for events: on Foxy it dropped one of
three messages in testing.)

**Abort** (`episode end abort <reason>`) only for causes outside the method under test: wrong
pose or wrong method started, the box moved before the start, a person walked in before the
robot moved, the recorder or a sensor driver crashed. A crash of the method's own node or of an
off-board link is the method's fail, not an abort. Re-run an aborted slot immediately, at most
once; aborts are listed in the report and never counted.

**Video rules**: one continuous, unedited recording per round from a fixed tripod at 1× speed,
showing the whole start area and the box, with the trial card at the start of every trial. No
cuts, no re-takes: a re-run happens only after an abort, on camera. Publish the full round
videos (unlisted is fine) and keep the bags. Any highlight reel is labelled "edited
highlights", speed changes are written on screen ("4×"), and it links to the full videos.

## 11. Analysis (host)

Copy the robot's data to the training machine first (the container wrote it as root with
umask 022, so a normal user can read it):

```bash
rsync -av <jetson>:<autoJetsonBot checkout>/autojetsonbot-learning/data/robot/ data/robot/
python tools/analyze_trials.py data/robot/trials/2026-11-02/trials.csv \
  --schedule trials/schedule.csv --setting "real robot, on-board TensorRT" \
  --latency jetson=data/robot/bags/trials_20261102_1405 --gate data/robot/bags/trials_20261102_1405 \
  --out-dir reports/2026-11-02
```

It writes `report.md` and `success.png`: per method k/n, success % with Wilson 95% CI; exact and
mid-p McNemar against `servo` on pose_id × round pairs (discordant counts b and c) with a
Newcombe (1998, method 10) score CI of the paired difference; time to dock, final range error
and final marker bearing as median (IQR);
outcome breakdown; success per pose; scheduled-but-missing and unscheduled trials; policy
latency p50/p99 per source and backend; gate period jitter and missed deadlines; every abort and
data issue. Rules: each slot is scored once from its first success/fail row, so a re-run of a
finished trial can never change it; a success with an intervention counts as a fail.
`--latency` and `--gate` also take JSONL captured live
(`ros2 topic echo --csv --full-length /policy/latency std_msgs/msg/String > lat.jsonl`; without
`--full-length` Foxy truncates the JSON at 128 characters).

## 12. Honest reporting

Every number, chart and video carries its labels:

- **Setting**: real robot or simulation (`fake_robot`); on-board (Jetson Nano, backend and
  precision) or off-board (which machine, over Wi-Fi); autonomous or teleoperated.
- **Scale**: one robot, one room, one box and marker, 10 fixed start poses × 3 rounds = 30
  trials per method; demos: how many, by how many operators, whether the test poses were held
  out.
- **Statistics**: k/n with the Wilson 95% CI, never a bare percentage. McNemar p values are
  two-sided and were planned before the trials; with 30 pairs only large differences are
  detectable, so "not significant" is not "equal". Name any analysis decided after seeing the
  data as post hoc.
- **Completeness**: all scheduled trials reported, aborts listed with reasons, failures shown in
  the videos; nothing excluded silently. The report's schedule check and data-issue sections stay
  in the published report.
- **Video**: "1×, unedited, full round" or "edited highlights (full runs: link)".
