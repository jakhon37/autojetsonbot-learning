# v4 pre-registration: clean stops, and demonstrations through the robot's lens

> **SIMULATION ONLY.** Everything planned here runs in `fake_robot`, the kinematic simulator,
> in `ros:jazzy-ros-base` containers on an x86 CPU. Nothing is measured on autoJetsonBot, and
> nothing here predicts how either method does on the real robot.

Written on 29 October 2026, before any v4 demonstration was recorded. The design, the
predictions and the analysis below are fixed by the commit that adds this file. The run starts
from that commit with a clean working tree. A pilot recorded scripted demonstrations from
training spots only (section 10). Its data are not used for training or analysis, and its bags
have been deleted; its small summaries are in [pilot/](pilot/). Its median final range anchors
P3 and P5 (section 7).

## 1. Why v4

[results/sim_eval_v3_bcfix](../sim_eval_v3_bcfix/SUMMARY.md) found the following.

- **BC learned to creep instead of stopping.** BC v2 reached the docking zone, but the
  smallest request of its last second had a median of 0.016–0.017 m/s per condition with the
  80° lens; the stop-weighted `bc_v2sw` had 0.012–0.015 m/s. The judge counted a request of up
  to 0.02 m/s as still.
- **The servo demonstrations mostly ended on creep.** Lidar noise (σ 5 mm per beam) toggled the
  servo across its 0.27 m stop range. With `k_v` 0.8, its heading-scaled creep
  `k_v`·(r − 0.25)·(1 − |bearing|/25°) was already under 0.02 m/s, so the judge's 1 s hold
  completed before the servo stopped. Only 2 of 40 demonstrations requested zero for their
  whole last second. Measured again with this version of `stop_labels.py bags` on the v3 bags:
  - in the same 2 of the 40, every request of the last 1.00 s was exactly v = 0
    (`max_v_last1s`, P1's measure); 1 of the 40 had requested exactly v = 0 for at least 1 s
    (`zero_for_s`);
  - in 24 of the 40, the request went from a stop back to moving while the judge reported the
    robot in band.
- **Under the 62.2° lens BC stopped short of the band.** BC v2 stopped at judge ranges of
  0.315–0.344 m and `bc_v2sw` at 0.326–0.363 m. In ground truth (the v3 bags, measured with
  this version of `sim_eval_summary.py episodes`) the medians were 0.335 m (0.317–0.342 m) and
  0.350 m (0.328–0.360 m). The narrower lens magnifies the marker by tan 40° / tan 31.1° = 1.39.

## 2. Question

Under a protocol where demonstrations end on a commanded stop and the judge requires one:

- **(a)** does BC, trained with the unchanged recipe, learn to stop, and how does it compare
  with the servo?
- **(b)** does recording the demonstrations through the robot's lens (Pi Camera v2, HFOV 62.2°
  = 1.0856 rad) remove the `hfov62` failure seen in v3?

## 3. Protocol changes (the defaults from v4 on)

1. **Judge.** The controller's request (`/cmd_vel_in`) counts as still only if
   |v| ≤ `request_still_v` = 0.005 m/s, and |w| ≤ `stopped_w` = 0.05 rad/s as before.
   Everything else is unchanged: odometry still needs |v| ≤ 0.02 m/s and |w| ≤ 0.05 rad/s, the
   band is 0.18–0.32 m, the bearing limit 10°, the hold 1 s, the timeout 30 s, and the obstacle
   rule is the same.
2. **Servo: a stop latch with hysteresis.** It engages when the range estimate is ≤
   `target_range` + `range_tol` (0.27 m). While it is engaged, the servo requests v = 0
   exactly; w still centres the marker, and the search still rotates in place. It releases only
   when the range estimate is > `target_range` + `range_tol` + `stop_hysteresis` (0.30 m with the
   default `stop_hysteresis` of 0.03 m). It is updated on every tick that sees the marker, with
   the range estimate the law uses: the lidar, or the marker-size estimate when the lidar has no
   valid front reading. An unknown range keeps the state. The latch is reset at every episode
   start; a range from a scan or frame received before the start shows the previous pose and
   counts as unknown. With `stop_hysteresis` 0 the servo is exactly the v3 law (unit-tested on a
   grid of bearings, ranges and prior states).

Both values are set explicitly in `config/sim.yaml` and `config/robot.yaml`. The v2 and v3
results used the old behaviour, and they are reproduced with `request_still_v:=0.02` and
`stop_hysteresis:=0.0`. They are not re-run.

## 4. Design

### Demonstrations

There are two sets, each with the v3 recipe:

- the 20 training spots 10–29 of `config/sim_demo_poses.yaml`, 2 rounds;
- the servo gains of `scripts/sim_e2e.sh` per round: round 1 `k_v` 1.0, `k_w` 1.2, `v_max` 0.15;
  round 2 `k_v` 0.8, `k_w` 1.5, `v_max` 0.12;
- `fake_robot` with `randomize:=true`, `seed` 1 for round 1 and 2 for round 2;
- the latched servo under the method name `servo_demo`, ended by the v4 judge.

| Set | Lens in `fake_robot` (`camera_hfov_rad`), `dock_judge` and `servo_baseline` (`hfov_rad`) | Work folder |
|---|---|---|
| A | 1.396 rad (the simulator's 80°) | `sim_work/v4` |
| B | 1.0856 rad (the robot's 62.2°), `sim_e2e.sh --hfov 1.0856` | `sim_work/v4_62` |

Each set becomes a LeRobot dataset with `tools/bag_to_lerobot.py`, as in v3 (the idle start of
each episode is trimmed).

### Models

- **`bc_v4`** is trained on set A and **`bc_v4_62`** on set B.
- Each uses the v3 command unchanged: `train_bc.py --epochs 5 --batch-size 32 --device cpu
  --methods servo_demo --outcomes success --split-by pose_id --val-fraction 0.2 --seed 0`,
  ResNet18 from scratch, no `--stop-weight`.
- Each is exported to ONNX opset 13. `export_onnx.py` refuses a model whose ONNX Runtime output
  differs from PyTorch by more than 1e-4, and records the largest difference and the sha256.

### Before any rollout (reported; changes nothing)

For both sets and models:

- `stop_labels.py bags`: how every demonstration ended;
- `stop_labels.py dataset`: the stop-label statistics;
- `stop_labels.py offline`: the model's predicted v on its own validation frames, by label.

### Evaluation

- **Setup.** Jazzy with ONNX Runtime on CPU; results in `results/sim_eval_v4_cleanstop/`.
- **Session 1.** `sim_eval.sh --work sim_work/v4 --methods servo,bc_v4` on `clean_ref`,
  `noise`, `wide` and `hfov62`:
  - fake_robot seeds 0, 201, 202 and 203 and schedule seed 2026;
  - 10 poses × 3 rounds per condition (`clean_ref`: randomisation off, 1 round);
  - servo and `bc_v4` interleaved in one launch per condition.
- **Session 2.** `sim_eval.sh --work sim_work/v4_62 --methods bc_v4_62` on `hfov62` only,
  paired with the session-1 `hfov62` rows on pose_id × round.
- **Conditions** (as in v2 and v3):
  - `clean_ref`: the test poses 0–9 and the 80° lens, no randomisation;
  - `noise`: the test poses with the randomisation the demonstrations were recorded with;
  - `wide`: 10 start poses outside the demonstrations' distance / lateral / heading box;
  - `hfov62`: the test poses with the robot's lens in `fake_robot`. The judge and the servo are
    told the true HFOV; `bc_v4` was trained at 80° and `bc_v4_62` at 62.2°.
- **Machine.** The simulation containers are pinned to cores 0–2 (`SIM_CPUSET=0-2`). Pinning
  does not reserve those cores: other work on the host can still slow the nodes (section 6).
  The load average at the start and end of every session segment is in the provenance, with
  the stop rules and lens the running nodes report.

## 5. Outcomes

- **Primary: success under the v4 judge**, scored as `tools/analyze_trials.py` scores it.
- **Secondary:**
  - **Stop commanded:** the smallest |v| among the controller's own requests in the last 1 s
    before the end is below 0.005 m/s (`episodes_gt.csv` `own_v_min_last1s`). A policy arm's
    own requests leave out `policy_runner`'s fallback zeros (validity check, below); the
    servo's are all its requests. The same over every request (`cmd_v_min_last1s`, v3's
    measure) is reported as a sensitivity line.
  - **Ground-truth final lidar range and bearing, from the bags.** At the last
    `/fake_robot/ground_truth` pose before the end: the lidar front range `dock_judge` would
    measure there without noise, and the marker's true bearing (`final_range_gt_m`,
    `final_bearing_gt_deg`). The judge's measured final range and bearing (`trials.csv`) are
    reported next to them.
  - **Time to success:** start to the judge's end for successes (`duration_s`; it includes the
    1 s hold).
  - **Failure reason:** timeout or stop_obstacle.
- **Validity check: stops on stale input.** A controller can request a stop because its input
  is missing, not because it decided to.
  - **`policy_runner`** sends exactly (0, 0) instead of its network's output on a tick whose
    newest frame is missing or older than `max_image_age` (0.2 s), and after an inference
    error. A ROS timer whose callback overran its 100 ms period fires again at once; when the
    newest frame is by then older than 0.2 s, this tick is a fallback zero, a few ms after the
    policy's own request. The judge reads the latest request at each 10 Hz tick, so a fallback
    zero can stand in for the policy's own request. In v3, 2 of the 200 BC episodes had one in
    their last second, 1 and 5 ms after policy ticks that took 69 and 140 ms, and in each it
    was the only request below 0.005 m/s (`bc_v2` noise pose 8 round 2, `bc_v2sw` noise pose 9
    round 2).
  - **`servo_baseline`** sends (0, 0) when no frame arrived for 0.5 s.
  - **Counts.** For every episode, from the bags, in the last 1.0 s before the end event
    (`episodes_gt.csv`):
    - `camera_frames_last1s`: camera frames recorded on `/camera/image_raw/compressed`, for all
      arms (`fake_robot` sends 15 Hz);
    - `fresh_inferences_last1s`: for the policy arms (every arm but the servo), the
      `/policy/latency` messages. `policy_runner` publishes one per inference on a frame at
      most 0.2 s old, and none with a fallback zero; about 10 at its 10 Hz;
    - `fallback_zeros_last1s`: for the policy arms, the requests of exactly (0, 0). The
      network's own output is never exactly (0, 0) (a linear head; `policy_runner` applies no
      deadband), so these are the fallback zeros. In the v3 BC episodes, none of the 25 exact
      (0, 0) requests was followed by a `/policy/latency` message within 45 ms, and each of the
      other 43,529 requests was followed by one within 22 ms.

    What the recorder received stands in for what the controller received.
  - **Flag.** A success is flagged stale (`stale_last1s`) when that window had fewer than 5
    camera frames or, for a policy arm, fewer than 5 fresh inferences or any fallback zero.
  - **Why these thresholds.** Nominally the window holds 15 frames, 10 fresh inferences and no
    fallback zero; the 200 BC episodes of v3 had 14–16, 8–11 and 0–1. Fewer than 5 frames or
    inferences leaves at least a third of the second without input. One fallback zero is
    enough, because the judge may have read it at one of its ticks.
  - **Reporting.** Primary results are reported as scored. A sensitivity line per condition
    and arm gives the flagged count and k/n with the flagged successes counted as failures. A
    success the check cannot read (no `episodes_gt.csv` row, or a bag without a counted topic)
    is listed as not checked and stays a success in that line. v4's bags record every counted
    topic, so the write-up reports any such success as a recording fault.

## 6. Comparisons

| | Pair | Conditions |
|---|---|---|
| C1 | `bc_v4` vs servo | each of `clean_ref`, `noise`, `wide`, `hfov62` (4 tests) |
| C2 | `bc_v4_62` vs `bc_v4` | `hfov62` |
| C3 | `bc_v4_62` vs servo | `hfov62` |

- Each comparison uses the exact two-sided McNemar test on pose_id × round pairs and Newcombe's
  95 % CI of the paired difference. Every arm gets a Wilson 95 % CI.
- **No multiplicity correction** is applied across the 6 tests (C1 in four conditions, C2 and
  C3).
- **The 3 rounds of a pose are correlated**, so the CIs are too narrow and the p values too
  small.
- **C2 and C3 pair arms from different sessions.** They share the seed, the poses and the
  schedule rule, not the noise draws, and not the CPU load: the sessions run at different
  times, session 1 with `servo_baseline` beside `policy_runner` and session 2 without it. So
  C2 and C3 confound the lens with the noise draws and with the load.
- **CPU load is not controlled.** `SIM_CPUSET` pins the containers to cores 0–2 but does not
  reserve them on a host shared with other work, and the load average is sampled only at the
  start and end of each segment. Timing changes what a controller sends: a slow policy tick is
  followed by a fallback zero (section 5). Next to C2 and C3, the write-up therefore reports,
  per session and condition, the inference p50 / p99 and the gate's missed deadlines (each
  `report.md`) and the trials with a fallback zero (the validity check).
- **The v2 and v3 arms are descriptive only**, with no tests: they were judged under the old
  rule.

## 7. Predictions

Written before any v4 data, verbatim:

> P1 data: each demo set has >= 36/40 successes; every success ends with the servo requesting
> exactly v = 0 for >= 1 s; in set A, full-stop frames in the last 2 s of episodes are >= 5 % of
> the training frames (v3: 2.5 %).
>
> P2 servo: >= 9/10 in clean_ref and >= 27/30 in noise.
>
> P3 BC stops: bc_v4 commands a stop in >= 80 % of its clean_ref + noise trials, and the median
> ground-truth final lidar range of its stop-commanded clean_ref + noise trials is within
> 0.25-0.29 m (set-A pilot demonstrations ended at a median of 0.270 m).
>
> P4 BC vs servo: bc_v4 >= 24/30 in noise; C1 not significant in clean_ref and noise (low power:
> a non-significant result is not evidence of equivalence; say so).
>
> P5 lens: on hfov62, bc_v4 stops short (median final lidar range > 0.32 m; pinhole model: a
> marker view that ends set-A demos at r80 = 0.270 m (pilot median, ground truth) is matched
> through the 62.2 deg lens at r62 = 1.39 x (r80 - c) + c ~ 0.361 m, c = camera offset ahead of
> the lidar = 0.036 m (fake_robot camera_x 0.10 - lidar_x 0.064)) and succeeds in <= 5/30;
> bc_v4_62 succeeds in >= 20/30 and C2 is significant (p < 0.05) in its favour.

How each prediction is read is fixed now. Every number comes from a file that a step of
section 9 writes.

- **Denominators.** Every scheduled slot ends scored: a demonstration or evaluation step that
  leaves one unscored is an infrastructure failure and is rerun (section 8). So the k/40, k/30
  and k/10 thresholds are counts over the full schedule, and P3's 80 % is 32 of 40.
- **P1, successes:** successes among the 40 scheduled demonstrations of each set (step 3). The
  gate that stops the run (section 8) is a separate, lower bar.
- **P1, exact zero:** for every success, `max_v_last1s` is exactly 0 (`stop_labels.py bags`).
  - **The measure.** The largest requested |v| among the `/cmd_vel_in` requests recorded in
    the last 1.00 s before the end event, unrounded: the CSV writes it in full, and the `.txt`
    counts the successes where it is 0. A success without any request in that window does not
    count.
  - **Its margin.** The window lies inside the judge's 1 s hold, which began at a 10 Hz tick
    after the servo's first zero. The servo's last request with v ≠ 0 came one request period
    before that zero (50 ms at its 20 Hz), so it falls before the window by at least about that
    period, give or take message delivery (section 10).
  - **`zero_for_s` stays descriptive.** It is the time from the first request of the final run
    of exact zeros to the end event, on the recorder's clock. It exceeds 1.00 s by the wait for
    the judge's first tick, give or take message delivery: by 10 ms in the pilot's closest case.
- **P1, full stops:** in set A's `stop_labels.py dataset`, `all.stop_end.full_stops /
  all.frames` ≥ 0.05. These are the `servo_demo` successes BC trains and validates on: the same
  computation that gave v3's 163 / 6,545 = 2.5 %. The train-split value is reported next to it.
- **P2:** servo k/n in the `clean_ref` and `noise` rows of the primary table.
- **P3:** both numbers are in the first two lines of `P3_stops.txt` (step 9). Its
  `sim_eval_summary.py stops` counts "Stop commanded" as `SUMMARY_table.md` does, on the
  policy's own requests (`own_v_min_last1s`: `policy_runner`'s fallback zeros left out,
  section 5).
  - **Stop commanded.** `bc_v4`'s "Stop commanded" in `clean_ref` plus `noise` is at least 80 %
    of its scored trials there (32 of 40). For successes this part is nearly implied: the v4
    judge needs every request it reads during its 1 s hold at |v| ≤ 0.005 m/s, so a success
    misses "Stop commanded" (smallest |v| below 0.005 m/s) only at the 0.005 m/s boundary, in a
    hold without any request of its own, or when fallback zeros carried the hold (which the
    validity check flags). The part is informative for the failures.
  - **Where it stops.** The median ground-truth final lidar range (`final_range_gt_m`) of those
    stop-commanded trials, successes and failures together, is within 0.25–0.29 m (0.2500 to
    0.2900 m as printed). This says where `bc_v4` stops. With no stop-commanded trial, this
    part is not met. The 0.270 m comes from the pilot (section 10), which used training spots
    only.
  - **Sensitivity.** The third line of `P3_stops.txt` gives both numbers over every request
    (`cmd_v_min_last1s`, v3's measure, fallback zeros included). P3 is judged on the first two
    lines; the write-up reports the third next to them.
  - **Incomplete.** If a scored trial has no ground-truth row, `stops` says so in the file
    ("INCOMPLETE") and exits with status 2. P3 is not read from such a file (section 9).
- **P4:** `bc_v4` k/30 in `noise`; C1's exact p ≥ 0.05 in both `clean_ref` and `noise`.
- **P5:**
  - `bc_v4`'s median "GT lidar range" on `hfov62` (the final-errors table, which gives it to 4
    decimals) is above 0.320 m: 0.3201 m or more as printed;
  - `bc_v4` succeeds in at most 5 of 30 there, and `bc_v4_62` in at least 20 of 30;
  - C2's exact p is below 0.05, with c > b (more pairs where `bc_v4_62` succeeds and `bc_v4`
    fails than the reverse).
  - r80 is the pilot's set-A median (section 10); the thresholds do not depend on it.

## 8. Rules

- **Nothing changes once the first v4 demonstration is recorded:** not the code, the recipe,
  the judge, the servo, the conditions or the analysis.
- **Infrastructure failures are the only exception.** They are failures of the machinery
  around the methods (section 9 says how each shows):
  - a step that stops early: a crash of the script or of the container, a full disk, or a
    container restart;
  - a demonstration or evaluation step that ends with a scheduled slot unscored: a start that
    could not be delivered ("slot skipped"), which is also how a crash of `fake_robot`,
    `dock_judge`, `trial_logger` or the recorder shows, or an episode the judge did not end
    within 45 s ("aborting it"). An attempt the scripts abort and then repeat in the same slot
    ("start not delivered to every node") is not one: it is listed in the report and never
    counted;
  - a failure of the analysis inside a step (`sim_eval.sh` then ends with a nonzero status).

  Each is logged in `results/sim_eval_v4_cleanstop/RUN_LOG.md`, with what the discarded attempt
  had scored, and the affected step is rerun from scratch (section 9). A failure confined to
  the analysis is redone without new trials.
- **Which attempt counts.** The first attempt of a step that completes is its result; a
  complete attempt is never rerun. A step gets at most two reruns: if its third attempt also
  ends in an infrastructure failure, the run stops there and the write-up reports every attempt.
- **A crash of the method's own controller is that method's fail**, as in docs/PROTOCOL.md
  section 10, not an infrastructure failure: `servo_baseline` for the servo and the
  demonstrations, `policy_runner` for a policy arm. The scripts score such a slot
  `fail controller_down`, and the write-up lists each one.
- **The demonstration gate is separate from P1.**
  - **Below 30.** If a set has fewer than 30 successes of its 40 scheduled demonstrations, stop
    and report before training.
  - **30–35.** P1 is reported as not met and the run continues. BC trains on that set's
    successes, as the recipe does (`--outcomes success`).
  - **Training-set sizes.** The write-up states each set's size. Whenever the two sets' success
    counts differ, C2 also compares training-set sizes, and the held-out validation spots can
    differ as well (step 10).
- **Anything else is exploratory** and labelled so.
- **One simulation session at a time.** No demonstration or evaluation session overlaps another
  one.

## 9. Commands

Run these from the repository root, at the commit that adds this file, one step at a time. Each
long step runs detached; poll its log until the last line is `EXIT <status>`. Never start a
step while a simulation step is still running.

### Setup (in every shell)

```bash
git diff --quiet HEAD && test -z "$(git status --porcelain -- . ':(exclude)results/sim_eval_v4_cleanstop')" \
  && git rev-parse HEAD || echo 'DIRTY TREE: stop'   # prints the commit
export CA_BUNDLE=/root/.ccr/ca-bundle.crt   # the HTTPS proxy's CA on this machine
export HF_HUB_OFFLINE=1 SIM_CPUSET=0-2
export PYTHON=/path/to/venv/bin/python      # Python 3.12 with requirements-tools.txt
R=results/sim_eval_v4_cleanstop
mkdir -p sim_work
docker info > /dev/null && df -h . | tail -1   # Docker must be up; plan for about 6 GB
```

- **The first line** checks that no tracked file differs from the commit and that nothing is
  untracked outside `results/sim_eval_v4_cleanstop`, where this run's own outputs appear from
  step 3 on. It prints the commit, or `DIRTY TREE: stop`.
- **The provenance** of every stage and session segment records the paths that differ from the
  commit as "dirty: N paths outside results/, M tracked under results/". Both numbers must be
  0; the run's new, untracked result files are not counted.
- **`SIM_CPUSET`** pins the simulation containers of `sim_e2e.sh` and `sim_eval.sh`. Training
  and the analysis run on the host, unpinned, as in v3 (`train.provenance` says so).

### Step 1. Demonstrations, set A (about 15 min)

```bash
nohup bash -c 'bash scripts/sim_e2e.sh --work sim_work/v4 demos; echo "EXIT $?"' \
  > sim_work/v4_demos.out 2>&1 &
```

### Step 2. Demonstrations, set B (about 15 min, after step 1)

```bash
nohup bash -c 'bash scripts/sim_e2e.sh --work sim_work/v4_62 --hfov 1.0856 demos; echo "EXIT $?"' \
  > sim_work/v4_62_demos.out 2>&1 &
```

### Step 3. How the demonstrations ended: P1 and the gate

```bash
for s in v4 v4_62; do
  $PYTHON scripts/stop_labels.py bags sim_work/$s/bags/jazzy/demos_r1 sim_work/$s/bags/jazzy/demos_r2 \
    --out $R/demos_${s}_endings.csv | tee $R/demos_${s}_endings.txt
  awk -F, -v set=$s 'NR > 1 { k += $4 == "success"; n += $4 == "success" || $4 == "fail" }
    END { printf "%s: %d successes of 40 scheduled demonstrations, %d of 40 slots scored\n", set, k, n }' \
    sim_work/$s/trials/jazzy/demos.csv | tee -a $R/demos_${s}_endings.txt
done
```

- **Every slot scored.** The last line of each `.txt` must say "40 of 40 slots scored".
  Otherwise that set's demonstration step was an infrastructure failure (section 8): rerun it,
  then this step.
- **The gate (section 8).** The same line gives the set's successes of 40. If a set has fewer
  than 30, stop here and report. With 30–35, P1 is not met, and the run goes on to step 4.
- **The exact zero of P1.** The `.txt` files give it as "successes whose largest requested |v|
  in the last 1.00 s was exactly 0 (max_v_last1s, unrounded)". The line "successes that ended
  with a request of exactly v = 0 for >= 1 s" (`zero_for_s`) is descriptive.

### Step 4. Datasets and models (about 30 min each, one after the other)

```bash
nohup bash -c 'bash scripts/sim_e2e.sh --work sim_work/v4 train; echo "EXIT $?"' \
  > sim_work/v4_train.out 2>&1 &
# after EXIT 0:
nohup bash -c 'bash scripts/sim_e2e.sh --work sim_work/v4_62 train; echo "EXIT $?"' \
  > sim_work/v4_62_train.out 2>&1 &
```

The `train` stage runs `bag_to_lerobot.py` on the set's two demonstration bags, `train_bc.py`
with the command of section 4, and `export_onnx.py`.

### Step 5. Model checks and training records

```bash
sha256sum sim_work/v4/runs/bc/policy.onnx sim_work/v4_62/runs/bc/policy.onnx
for s in v4 v4_62; do
  mkdir -p $R/training/$s
  for f in sim_work/$s/runs/bc/{policy_meta.json,metrics.jsonl,split.json} \
           sim_work/$s/logs/jazzy/{episodes.txt,train_bc.log,export_onnx.log,demos.provenance,train.provenance} \
           sim_work/$s/logs/jazzy/demos_r{1,2}.params.txt sim_work/$s/trials/jazzy/demos.csv; do
    sed "s|$PWD/||g" "$f" > $R/training/$s/$(basename "$f")
  done
  $PYTHON -c "import json, sys; e = json.load(open(sys.argv[1]))['export']; print(sys.argv[1], e['sha256'], 'opset', e['opset'], 'max |ORT - torch|', e['parity']['max_abs_diff'])" $R/training/$s/policy_meta.json
done
```

The printed sha256 must match `sha256sum`, the opset must be 13, and the largest ONNX Runtime
vs PyTorch difference must be at most 1e-4 (`export_onnx.py` refuses to write the model
otherwise).

### Step 6. Before any rollout

```bash
for s in v4 v4_62; do
  $PYTHON scripts/stop_labels.py dataset --root sim_work/$s/dataset --split sim_work/$s/runs/bc/split.json \
    --out $R/dataset_${s}_stop_labels.json > /dev/null
  $PYTHON scripts/stop_labels.py offline --root sim_work/$s/dataset --split sim_work/$s/runs/bc/split.json \
    --model bc_$s=sim_work/$s/runs/bc/policy.onnx --out $R/offline_val_bc_$s.json > /dev/null
done
$PYTHON -c "import json; a = json.load(open('$R/dataset_v4_stop_labels.json'))['all']; print('P1, set A: full stops in the last 2 s: %d / %d = %.1f %%' % (a['stop_end']['full_stops'], a['frames'], 100.0 * a['stop_end']['full_stops'] / a['frames']))"
```

### Step 7. Evaluation, session 1 (about 2 h)

```bash
nohup bash -c 'bash scripts/sim_eval.sh --work sim_work/v4 --results results/sim_eval_v4_cleanstop \
  --methods servo,bc_v4 clean_ref noise wide hfov62; echo "EXIT $?"' > sim_work/v4_eval.out 2>&1 &
```

After `EXIT 0`, the session is complete when both checks pass; otherwise it was an
infrastructure failure (section 8):

```bash
ls $R/{clean_ref,noise,wide,hfov62}/{trials.csv,schedule.csv,episodes_gt.csv,report.md,success.png,provenance.txt} \
  | wc -l   # 24
grep -L 'Every scheduled trial has exactly one scored result' $R/{clean_ref,noise,wide,hfov62}/report.md   # nothing
```

### Step 8. Evaluation, session 2 (about 25 min, after step 7)

```bash
nohup bash -c 'bash scripts/sim_eval.sh --work sim_work/v4_62 --results results/sim_eval_v4_cleanstop \
  --methods bc_v4_62 --pair-with sim_work/v4/eval_v2/jazzy hfov62; echo "EXIT $?"' > sim_work/v4_62_eval.out 2>&1 &
```

`--pair-with` puts session 1's `hfov62` rows (its `trials.csv`, `schedule.csv` and
`episodes_gt.csv`) in front of session 2's in `$R/hfov62`, so the three arms pair on
pose_id × round. After `EXIT 0`, check `hfov62` again: its schedule must hold the three arms,
and every slot must be scored; otherwise the session was an infrastructure failure:

```bash
awk -F, 'NR > 1 { print $4 }' $R/hfov62/schedule.csv | sort | uniq -c   # 30 each: bc_v4, bc_v4_62, servo
grep -L 'Every scheduled trial has exactly one scored result' $R/hfov62/report.md   # nothing
```

### Step 9. Tables

```bash
$PYTHON scripts/sim_eval_summary.py table --results $R --conditions clean_ref noise wide hfov62 \
  --vs bc_v4 --out $R/SUMMARY_table.md
$PYTHON scripts/sim_eval_summary.py stops --results $R --conditions clean_ref noise --method bc_v4 \
  --out $R/P3_stops.txt
```

Both must exit with status 0. `stops` exits with 2 when a scored trial has no ground-truth row
(its file then says "INCOMPLETE"): redo the report of the session concerned (its step's
command with `--report-only`), then this step.

Where each number is read:

- **Success, Wilson CIs, C1 and C3** (b, c, exact p, Δ with Newcombe's CI): each condition's
  `report.md`, section "Paired comparison with `servo`". The primary table of
  `SUMMARY_table.md` repeats k/n, the CI, b / c and p.
- **C2:** `SUMMARY_table.md`, the table "Paired with `bc_v4`" (b / c, exact p, Δ with
  Newcombe's CI).
- **Stop commanded, time to success and failure reasons:** `SUMMARY_table.md`, the primary and
  secondary tables.
- **Ground-truth final lidar range and bearing:** the final-errors table of `SUMMARY_table.md`
  ("GT lidar range, m", "GT |bearing|, deg"), from each condition's `episodes_gt.csv`.
- **P3:** `P3_stops.txt`. Its first two lines: `bc_v4`'s trials in `clean_ref` plus `noise`
  that commanded a stop with their own requests, and the median ground-truth final lidar range
  of those. Its third line: the same over every request (the sensitivity line).
- **Validity check:** `SUMMARY_table.md`, the table "Validity check: stale inputs at the end".
  Per condition and arm, it gives:
  - the camera frames and fresh inferences of the last 1 s, as median (min);
  - the trials with a fallback zero in the last 1 s;
  - the successes flagged stale, and how many could not be checked;
  - k/n with the flagged successes counted as failures.

  Each episode's counts and flag are in its condition's `episodes_gt.csv`.
- **Timing:** each condition's `report.md`, the tables of policy latency (inference p50 / p99
  per session) and of the safety gate's period (missed deadlines per session).

### Step 10. Write-up

`$R/SUMMARY.md` reports:

- each prediction as met or not met, with its number;
- each demonstration set's size (successes of 40), and whenever the two sets' counts differ,
  that C2 also compares training-set sizes (section 8);
- each model's validation spots (`training/<set>/split.json` with `episodes.txt`):
  `train_bc.py` holds out spots until it has 20 % of the episodes, so a set with failures can
  hold out other spots than a full one;
- the validity check per condition and arm, next to the primary results, which stay as
  scored, and P3's sensitivity line next to P3;
- C1–C3 with b / c, the exact p and Newcombe's CI, and next to C2 and C3, per session and
  condition, the inference p50 / p99, the gate's missed deadlines and the trials with a
  fallback zero (section 6);
- every `fail controller_down`, every node that died during a session ("process has died" in
  its launch logs) and when, and every infrastructure failure in `RUN_LOG.md` with what its
  discarded attempt had scored;
- what the results do not show, including the caveats of section 6;
- anything beyond this file, labelled exploratory.

### Infrastructure failures

How each failure of section 8 shows:

- **A step that stops early:** a crash or a full disk shows as an `EXIT` line with a nonzero
  status; a container restart as a log without an `EXIT` line and no `sim_e2e.sh` or
  `sim_eval.sh` process left (`pgrep -f scripts/sim_`).
- **A slot left unscored:** the log has "slot skipped" or "aborting it", and the step's check
  fails (step 3 for the demonstrations, after steps 7 and 8 for the sessions).
- **A failed analysis:** `sim_eval.sh` logs "report failed" and ends with "report failed: …;
  the trials are recorded" and a nonzero `EXIT`.

Then:

1. **Steps 7 and 8 only:** move the session's work folder aside,
   `mv sim_work/v4/eval_v2 sim_work/v4/eval_v2.failed.$(date -u +%Y%m%dT%H%M%SZ)` for session
   1, or the same with `sim_work/v4_62/eval_v2` for session 2. Never resume a session:
   `sim_eval.sh` skips slots that are already scored, which would mix two launches. (A failed
   analysis only keeps its trials; see 4.)
2. Add a line to `$R/RUN_LOG.md`: the UTC time, the step, what failed, what was moved aside,
   and what the discarded attempt had scored, or "none scored":
   - steps 1 and 2: the step-3 `awk` line on the set's `demos.csv`, run before the rerun
     deletes it;
   - steps 7 and 8: per arm and condition, k/n of the primary table of
     `$PYTHON scripts/sim_eval_summary.py table --results <moved-aside folder>/jazzy --conditions
     clean_ref noise wide hfov62 --out <moved-aside folder>/SUMMARY_table.md`.
3. Remove leftover containers: `docker ps -a --filter name=jbl_ -q | xargs -r docker rm -f`.
4. Rerun:
   - **Steps 1, 2 and 4:** rerun the same command. The `demos` stage deletes the set's
     `demos.csv` and bags first, and `train` overwrites the dataset and the model.
   - **Steps 7 and 8:** rerun the same command, which starts a new `eval_v2` folder.
   - **A failed analysis only** (every trial ran and was scored; the failure came after
     "ground truth, then analyze_trials"): rerun the same command with `--report-only`, which
     runs no trial. Section 8 allows it: it changes nothing that was recorded.
   - **Steps 3, 5, 6 and 9** run on the host and only write files: rerun them.
   - **An analysis that fails the same way twice:** stop and report, since the analysis cannot
     change (section 8).

## 10. Pilot (training spots only)

This was a dry run of the demonstration recording, on the training spots only; no test pose
was used. It ran on 29 October 2026, 08:56–09:15 UTC, before this file was written, with the
code of the commit that adds this file (then uncommitted). Of that code, only analysis and
logging changed afterwards; the nodes, the configs and the recording did not:

- **`stop_labels.py bags`** now writes `zero_for_s` to 1 ms and `max_v_last1s` unrounded,
  counts the 1 s holds and P1's exact zero from unrounded values, and writes a judge range of
  null as empty.
- **`sim_eval_summary.py`** gained the validity check (its `episodes` columns, with the
  fallback zeros, and its table), "Stop commanded" on the controller's own requests, the
  `stops` command, and the ground-truth range to 4 decimals in the final-errors table.
- **`sim_eval.sh`** ends with a nonzero status when a report fails, and
  **`tools/analyze_trials.py`** no longer fails on a bag without the topic it reads.
- **The provenance** of `sim_e2e.sh` and `sim_eval.sh` counts the paths that differ from the
  commit without the run's own result files, and the `train` stage records that it is not
  pinned.

Tests (with the synthetic-bag generator) and documentation were also edited. The commands
were:

```bash
SIM_CPUSET=0-2 bash scripts/sim_e2e.sh --work sim_work/v4_pilot --demo-rounds 1 demos                 # 80°
SIM_CPUSET=0-2 bash scripts/sim_e2e.sh --work sim_work/v4_pilot62 --demo-rounds 1 --hfov 1.0856 demos # 62.2°
```

- **Recipe.** Each run used round 1 only (`k_v` 1.0, seed 1) on spots 10–29: 20
  demonstrations per lens.
- **What the nodes reported.** `request_still_v` 0.005 and `stop_hysteresis` 0.03, and the
  intended lens in `fake_robot`, `dock_judge` and `servo_baseline`.
- **Analysis.** `stop_labels.py bags`, `stop_labels.py dataset` (on a scratch dataset) and
  `sim_eval_summary.py episodes` ran on the pilot bags.
- **What was kept.** The bags and the scratch datasets were deleted afterwards. The small
  summaries are in [pilot/](pilot/), one folder per lens (section 12).

| | Pilot, 80° | Pilot, 62.2° | v3, round 1 (80°, old rules) |
|---|---:|---:|---:|
| Successes | 20/20 | 20/20 | 20/20 |
| Every request of the last 1.00 s exactly v = 0 (`max_v_last1s`, P1's measure) | 20/20 | 20/20 | 2/20 |
| Ended with exactly v = 0 for ≥ 1 s (`zero_for_s`) | 20/20 (1.06–1.18 s) | 20/20 (1.01–1.27 s) | 1/20 |
| Request went from a stop back to moving while in band | 0/20 | 0/20 | 19/20 |
| Duration, median (range) | 14.1 s (9.0–27.9) | 14.3 s (9.5–29.9) | 15.0 s (8.7–27.7) |
| Final range, judge | 0.264–0.277 m | 0.268–0.279 m | 0.266–0.276 m |
| Final range, ground truth (noise-free lidar) | 0.268–0.274 m | 0.268–0.275 m | — |
| Full-stop frames in the last 2 s | 221 / 3,099 = 7.1 % | 223 / 3,375 = 6.6 % | 2.5 % (both rounds) |
| Final run of stop frames per episode | 10–12 frames | 10–13 frames | median 0 (both rounds) |

What it showed:

- **The latch removes the toggling, and every demonstration ends on a commanded stop.** The
  latch engaged once per approach, at a measured range of 0.265–0.270 m, and never released
  (apart from one start transient per lens, below).
  The robot then held still for the judge's 1 s hold, with the marker 3.5° (median, ground
  truth) from the camera axis, inside the servo's 4° deadband.
- **`zero_for_s` has a small margin by construction, so P1 reads `max_v_last1s`.**
  - **`zero_for_s`.** It counts from the servo's first zero to the judge's end event on the
    recorder's clock. The judge starts its 1.0 s hold at a 10 Hz tick after that zero (the first
    one at which the odometry is still too) and ends the episode at the first tick at least
    1.0 s later. So `zero_for_s` is at least 1.0 s plus the wait for that first tick, give or
    take message delivery. The pilot's values were 1.01–1.27 s; the smallest, 1.01 s (the
    pilot's CSV rounds to 0.01 s), was about 10 ms above 1 s.
  - **`max_v_last1s`.** Its window, the last 1.00 s, lies inside the hold. The servo requests
    at 20 Hz, so its last v ≠ 0 came 50 ms before its first zero, and the margin grows by that
    period (section 7).
  - **The pilot row for `max_v_last1s`.** The pilot's CSV gave it to 0.1 mm/s, 0.0000 in all
    40. The exact zeros follow from `zero_for_s` ≥ 1.01 s: the final run of exact zeros began
    before the last 1.00 s, and every success had requests in that window
    (`stop_share_last1s` 1.00). The v3 column comes from the v3 round-1 bags and the current
    `stop_labels.py`.
- **Some demonstrations come close to the 30 s timeout.** Three demonstrations at 80° and six
  at 62.2° took 21–30 s; spot 28 at 62.2° succeeded at 29.9 s. There are two causes:
  - **A long search.** The servo searches to the left first. A spot whose marker starts out of
    view on the right needs 10–11.5 s of search: spots 15, 19 and 21 at both lenses, and also
    spots 13 and 28 at 62.2°. Their markers start at −36° and −33°, inside the 80° lens's ±40°
    but outside the 62.2° lens's ±31°.
  - **An oblique final approach.** Spot 29 at 62.2° ended 36° off square. The servo asked for
    0.02–0.03 m/s, but the safety gate passed ≤ 0.01 m/s, and the last 4 cm took 20 s.

  Round 2 (`k_v` 0.8, `v_max` 0.12, `search_w` 0.6) was not piloted. Its slower approach could
  push a few of these demonstrations past 30 s, mostly with the 62.2° lens. P1 predicts at most
  4 failures per set. The gate in section 8 stops the run before training only below 30/40;
  with 30–35, P1 is not met and the run goes on.
- **The full-stop share was 7.1 %** of the frames in round 1 at 80°, against 5 % in P1.
- **A start transient survives in 1 of 19 starts per lens.** A reading of the previous pose,
  delivered just after the start, set the latch. The first in-view reading of the new pose
  released it, 0.57 s and 0.19 s after the start. The robot was searching, with v = 0 either
  way. No start pose lies within 0.30 m, so the latch cannot hold the robot at a start.
- **The pinned container had spare capacity.** It used a median of 105–108 % of one core
  (about 200 % while building the workspace) of the 3 cores it was pinned to. The load average
  was 0.6–2.5.
- **The anchor of P3 and P5.** The set-A pilot demonstrations ended at a ground-truth range of
  0.270 m (median 0.2701 m; 0.268–0.274 m; `pilot/v4_pilot/pilot_episodes_gt.csv`,
  `final_range_gt_m`). P5 uses it as r80, and P3's 0.25–0.29 m is centred on it. With
  c = 0.036 m, the matched range through the 62.2° lens is 0.361 m.

## 11. Analysis tools

All three are fixed at the commit that adds this file.

- **`tools/analyze_trials.py`** runs inside `sim_eval.sh`'s report for each condition (it
  writes `report.md` and `success.png`). It scores the trials: success needs outcome success
  and zero interventions, and aborted attempts are never counted. It gives the Wilson CIs, the
  exact McNemar tests on pose_id × round pairs against servo, and Newcombe's CI of the paired
  difference, plus times, final errors, policy latency and gate timing, and checks that every
  scheduled slot has exactly one scored result. A failure of any step of the report makes
  `sim_eval.sh` end with a nonzero status.
- **`scripts/sim_eval_summary.py`** has four commands:
  - `episodes` reads one row per episode from the bags (`episodes_gt.csv`): requested speeds
    (the smallest of the last 1 s over every request, `cmd_v_min_last1s`, and over the
    controller's own, `own_v_min_last1s`), zone reached, collisions, the ground-truth final
    pose with the noise-free lidar range and true marker bearing there, and the validity check
    (`camera_frames_last1s`, `fresh_inferences_last1s`, `fallback_zeros_last1s`,
    `stale_last1s`);
  - `merge` pairs the sessions;
  - `table` writes `SUMMARY_table.md` (`--vs bc_v4` adds C2), ending with the validity check's
    sensitivity table;
  - `stops` pools one method's scored trials over conditions: "Stop commanded" on the
    controller's own requests and the ground-truth final lidar range of those trials (P3),
    then both over every request (the sensitivity line). Its n is every scored trial in
    `trials.csv`; a scored trial without a ground-truth row makes it exit with status 2.
- **`scripts/stop_labels.py`** has three commands:
  - `bags`: how each demonstration ended (`max_v_last1s`, written unrounded, `zero_for_s`,
    `stop_for_s`, `restarts_in_band`, the final judge range);
  - `dataset`: stop-label counts by position in the episode, per split;
  - `offline`: each model's predicted v on its validation frames, by label class.

## 12. Files this run adds to `results/sim_eval_v4_cleanstop/`

- **This file**, `PREREG.md`.
- **The pilot's summaries** (section 10), committed with this file: `pilot/v4_pilot/` (80°)
  and `pilot/v4_pilot62/` (62.2°), each with
  - `demos.csv`, `demo_r1.yaml` and `demos.provenance`: the demonstrations as `trial_logger`
    wrote them, the servo gains and lens, and the stage's provenance;
  - `pilot_endings.csv` and `.txt`: `stop_labels.py bags` as it was then (`max_v_last1s` to
    4 decimals);
  - `pilot_episodes_gt.csv`: `sim_eval_summary.py episodes` as it was then, before the
    validity check (the ground-truth final range of P3 and P5);
  - `pilot_dataset_stop_labels.json`: `stop_labels.py dataset` on a scratch dataset of the
    pilot's bag (its `root`, a deleted scratch folder, is shortened);
  - `pilot_latch_log.txt`, `pilot_phases.txt` and `docker_stats.txt`: the latch's log lines,
    the phases of each demonstration, and the container's CPU and memory every 21 s.
- **From steps 3 and 6:**
  - `demos_v4_endings.csv` and `.txt`, `demos_v4_62_endings.csv` and `.txt`;
  - `dataset_v4_stop_labels.json`, `dataset_v4_62_stop_labels.json`;
  - `offline_val_bc_v4.json`, `offline_val_bc_v4_62.json`.
- **From step 5:** `training/v4/` and `training/v4_62/`.
- **From steps 7 and 8:** `clean_ref/`, `noise/`, `wide/` and `hfov62/`, each with
  `trials.csv`, `schedule.csv`, `episodes_gt.csv` (with each episode's validity check),
  `report.md`, `success.png` and `provenance.txt`.
- **From step 9:** `SUMMARY_table.md` (with the validity check's table) and `P3_stops.txt`.
- **From step 10:** `SUMMARY.md`.
- **Only after an infrastructure failure:** `RUN_LOG.md`, with what each discarded attempt had
  scored (section 9).

The bags, datasets and models stay in `sim_work/v4` and `sim_work/v4_62`, which are not in
git.
