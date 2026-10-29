# v4 results: clean stops, and demonstrations through the robot's lens

> **SIMULATION ONLY.** Every number here comes from `fake_robot` (the kinematic simulator) in
> `ros:jazzy-ros-base` containers on a 4-core x86 CPU, 29 October 2026. Nothing was measured on
> autoJetsonBot. Nothing here predicts how either method does on the real robot.

The design, the predictions, the rules and the commands are in [PREREG.md](PREREG.md), committed
(b72ba83, 12:26:09 UTC) and pushed (12:26:22 UTC) before any data of this run existed; step 1
started at 12:28:57 UTC. A disclosed pilot on training spots came before the pre-registration
(PREREG section 10). This run followed its section 9, steps 1 to 10, once each, from that commit
with a clean tree. No step failed, none was rerun, and no attempt was discarded; the one
deviation, in step 10, concerns how this file was saved ("Run record"). The start and end of each
step are in [RUN_LOG.md](RUN_LOG.md).

## Question

From [PREREG.md](PREREG.md) section 2, under a protocol where demonstrations end on a commanded
stop and the judge requires one (a request counts as still only at |v| ≤ 0.005 m/s; the servo has
a stop latch with 0.03 m of hysteresis):

- **(a)** does BC, trained with the unchanged recipe, learn to stop, and how does it compare with
  the servo?
- **(b)** does recording the demonstrations through the robot's lens (Pi Camera v2, HFOV 62.2° =
  1.0856 rad) remove the `hfov62` failure seen in v3?

## What ran

- **Demonstrations.** `scripts/sim_e2e.sh … demos`, the v3 recipe: the 20 training spots 10–29, 2
  rounds (round 1 `k_v` 1.0, seed 1; round 2 `k_v` 0.8, seed 2), randomisation on, the latched
  servo as `servo_demo`, ended by the v4 judge.
  - Set A: the simulator's 80° lens (`sim_work/v4`).
  - Set B: the robot's 62.2° lens (`--hfov 1.0856`, `sim_work/v4_62`).
- **Models.** For each set, `bag_to_lerobot.py`, then the v3 command unchanged (`train_bc.py
  --epochs 5 --batch-size 32 --device cpu --methods servo_demo --outcomes success --split-by
  pose_id --val-fraction 0.2 --seed 0`, ResNet18 from scratch), then ONNX opset 13.
- **Session 1.** `sim_eval.sh --methods servo,bc_v4` on `clean_ref`, `noise`, `wide` and
  `hfov62`: 10 poses × 3 rounds (`clean_ref`: 1 round, no randomisation), servo and `bc_v4`
  interleaved in one launch per condition, 14:35–16:03 UTC.
- **Session 2.** `sim_eval.sh --methods bc_v4_62 --pair-with sim_work/v4/eval_v2/jazzy` on
  `hfov62`, 16:06–16:20 UTC, paired with session 1's `hfov62` rows on pose_id × round.
- **Machine.**
  - The simulation containers were pinned to cores 0–2 (`SIM_CPUSET=0-2`), with ONNX Runtime
    1.31.1 on CPU in the container.
  - Image `mirror.gcr.io/library/ros@sha256:8eed0a46…b617`.
  - Training and the analysis ran on the host, unpinned (Python 3.12).
  - Other work shared the host, according to the other agents' own logs and file times
    ([RUN_LOG.md](RUN_LOG.md)):
    - During steps 1–4, another agent's C++ builds and benchmarks ran under `taskset -c 3` at
      nice 15. Its own affinity log shows threads of its ONNX Runtime test on cores 1 and 2,
      inside `SIM_CPUSET`, at 12:31:52–12:31:54 (step 1), before it added a shim that kept them
      on core 3 (12:34).
    - During session 1's `wide` segment, a second workflow downloaded about 0.9 GB of model and
      dataset files into RAM (15:11–15:15), decoded dataset video frames twice (15:19 and
      15:34), compiled a small library (15:25) and imported `transformers` (15:31). Whether it
      stayed on core 3 is not recorded. Afterwards it only wrote scripts and polled; during
      session 2 it only polled.
    - Spot checks at 14:41, 15:39, 15:49 and 16:14 UTC found nothing running beside the
      simulation container. They missed that `wide`-segment work, and their output was not
      saved.
- **Provenance.** Every stage and segment recorded commit b72ba83 with "dirty: 0 paths outside
  results/, 0 tracked under results/". The running nodes reported `request_still_v` 0.005,
  `stop_hysteresis` 0.03 and the intended lens. One gap: in the `noise` segment, the query for
  `servo_baseline.hfov_rad` returned "Node not found". The servo's own start-up line in that
  segment's log reads "hfov 1.3960 rad … stop_hysteresis 0.030 m".
- **`git.dirty: true` in the model records.** Both `training/*/policy_meta.json` files say so
  because `train_bc.py` counts untracked files: step 3's result files (`demos_v4*_endings.*`,
  written at 13:04) were untracked when training started. The stage provenance, which leaves the
  run's own result files out, records 0 changed paths.

## Models

| Arm | Model | sha256 | Trained on | Validation spots | Best epoch | Validation MAE v / w (mean-command baseline) |
|---|---|---|---|---|---:|---:|
| `bc_v4` | `sim_work/v4/runs/bc/policy.onnx` | `ed36c9b606586622a3f4f4a627d3652efa48fd9c11bc5ba077394b23558eff2e` | set A (80°): 32 episodes, 5,062 frames, spots 10–19, 21–23, 25, 27, 29 | 20, 24, 26, 28: 8 episodes, 1,333 frames | 4 of 5 | 0.011 m/s / 0.020 rad/s (0.043 / 0.071) |
| `bc_v4_62` | `sim_work/v4_62/runs/bc/policy.onnx` | `ea29531c146345340a663122cd9803b754b57c5f9812c98752558a8cdcbb0632` | set B (62.2°): 30 episodes, 5,195 frames, spots 11–19, 21–23, 25, 27, 29 | 10, 20, 24, 26, 28: 9 episodes, 1,481 frames | 5 of 5 | 0.009 / 0.019 (0.045 / 0.122) |

- **Export.** Both models are opset 13. The largest ONNX Runtime vs PyTorch difference is
  2.98e-07 for each (the limit is 1e-4). The printed sha256 matches `sha256sum` (step 5).
- **Datasets.** Fingerprint (meta + parquet) `09f7490b…` for set A and `59b636b8…` for set B.
- **Training time.** The epochs took 390, 368, 374, 412 and 973 s (set A) and 735, 394, 375,
  370 and 387 s (set B). The slow epochs ran while the other agent's work loaded the host.
- **Training-set sizes and validation spots differ.**
  - Set A has 40 successes of 40 and set B 39 of 40. `bc_v4` trained on 32 episodes (5,062
    frames) and `bc_v4_62` on 30 (5,195 frames; set B's demonstrations are longer, median 16.2
    against 15.4 s).
  - Set B's failed demonstration (spot 28, round 2) is in neither split. With spots 20, 24, 26
    and 28, set B's held-out share would have been 7 of 39 episodes (< 20 %). So `train_bc.py`
    also held out spot 10, and `bc_v4_62` never trained on it.
  - C2 therefore also compares 32 against 30 training episodes and different validation spots
    (PREREG section 8).
- **Records.** The policy metas, metrics, splits, logs, demonstration CSVs and provenance are in
  [training/](training/).

## Predictions

Each prediction as written in PREREG section 7, read as section 7 fixes. All five are **met** as
scored. The validity check (section 3 below) is reported next to them and does not change how
they are read.

| | Prediction | Result | Source | Verdict |
|---|---|---|---|---|
| P1 | Each demonstration set ≥ 36/40 successes | set A 40/40, set B 39/40 | last line of [demos_v4_endings.txt](demos_v4_endings.txt), [demos_v4_62_endings.txt](demos_v4_62_endings.txt) | met |
| P1 | Every success requested exactly v = 0 over its last 1.00 s (`max_v_last1s` exactly 0) | set A 40/40, set B 39/39 | the same files ("…was exactly 0 (max_v_last1s, unrounded)") | met |
| P1 | Set A: full-stop frames in the last 2 s ≥ 5 % of the frames | 454 / 6,395 = 7.1 % (train split: 362 / 5,062 = 7.2 %) | [dataset_v4_stop_labels.json](dataset_v4_stop_labels.json) `all.stop_end.full_stops / all.frames` | met |
| P2 | Servo ≥ 9/10 in `clean_ref` and ≥ 27/30 in `noise` | 10/10 and 30/30 | primary table of [SUMMARY_table.md](SUMMARY_table.md) | met |
| P3 | `bc_v4` commands a stop in ≥ 80 % (32 of 40) of its `clean_ref` + `noise` trials | 36 of 40 (90.0 %) | [P3_stops.txt](P3_stops.txt), line 1 | met |
| P3 | Median ground-truth final lidar range of those trials within 0.2500–0.2900 m | 0.2643 m (IQR 0.2618–0.2655, range 0.2587–0.2672; 36 with ground truth) | [P3_stops.txt](P3_stops.txt), line 2 | met |
| P4 | `bc_v4` ≥ 24/30 in `noise` | 27/30 | primary table | met as scored (17/30 with flagged successes counted as failures) |
| P4 | C1 not significant in `clean_ref` and `noise` | exact p = 1 (b / c = 1 / 0) and p = 0.25 (3 / 0) | [clean_ref/report.md](clean_ref/report.md), [noise/report.md](noise/report.md) | met |
| P5 | `bc_v4`'s median GT lidar range on `hfov62` > 0.320 m (≥ 0.3201 as printed) | 0.3486 m (IQR 0.3445–0.3503) | final-errors table of [SUMMARY_table.md](SUMMARY_table.md) | met |
| P5 | `bc_v4` ≤ 5/30 on `hfov62` | 0/30 | primary table | met |
| P5 | `bc_v4_62` ≥ 20/30 on `hfov62` | 23/30 | primary table | met as scored (16/30 with flagged successes counted as failures) |
| P5 | C2 exact p < 0.05 with c > b | b / c = 0 / 23, p = 2.38e-07 | "Paired with `bc_v4`" in [SUMMARY_table.md](SUMMARY_table.md) | met |

- **P1 and the pilot.** The pilot (PREREG section 10: training spots, round 1 only) had already
  shown 20/20 exact zeros at both lenses and 7.1 % full stops at 80°. Round 2 (`k_v` 0.8) and the
  repetition were new.
- **P3's sensitivity line** (line 3 of [P3_stops.txt](P3_stops.txt), every request counted,
  fallback zeros included) is the same: 36 of 40 (90.0 %), median 0.2643 m.
- **What P3's stop part adds.** The 36 trials that commanded a stop are exactly `bc_v4`'s 36
  successes there, as PREREG section 7 expects (a success requires it). None of its 4 failures
  commanded a stop: all were on pose 5, with a smallest own request of 0.012–0.020 m/s.
- **P4: low power.** A non-significant C1 is not evidence that `bc_v4` and the servo are
  equivalent. With 10 and 30 pairs, the 95 % CIs of the difference reach −40.4 pp and −25.6 pp.
- **With flagged successes counted as failures** (the validity check's sensitivity line, not the
  scoring rule), two of the counts fall below their thresholds:
  - `bc_v4` in `noise`: 17/30, against P4's 24;
  - `bc_v4_62` on `hfov62`: 16/30, against P5's 20.

  Counted that way, C1 in `noise` would also be significant, b / c = 13 / 0, exact p = 0.0002,
  so P4's second part would not hold either (exploratory: the PREREG defines this sensitivity
  line as k/n only). C1 in `clean_ref` would be 2 / 0 (p = 0.5), C2 0 / 16 (p = 3.1e-05) and C3
  14 / 0 (p = 0.0001). The predictions are read on the scored results, as section 5 fixes.
  Section 3 below and the exploratory section bear on how much weight the flags deserve.

## 1. The demonstrations end on a commanded stop (P1 and the gate)

Source: `stop_labels.py bags`, in [demos_v4_endings.txt](demos_v4_endings.txt) and
[demos_v4_62_endings.txt](demos_v4_62_endings.txt) (per demonstration: the `.csv` files).

| | Set A (80°) | Set B (62.2°) |
|---|---:|---:|
| Successes of 40 scheduled; slots scored | 40; 40 of 40 | 39; 40 of 40 |
| Every request of the last 1.00 s exactly v = 0 (`max_v_last1s`, P1's measure) | 40/40 | 39/39 |
| Ended with exactly v = 0 for ≥ 1 s (`zero_for_s`, descriptive) | 40/40 (1.011–1.243 s) | 39/39 (1.038–1.198 s) |
| Ended on creep the v2/v3 judge called still (0.005 ≤ \|v\| ≤ 0.02) | 0/40 | 0/39 |
| Request went from a stop back to moving while in band | 0/40 | 1/40 (spot 17, round 2; below) |
| Duration of successes, median (range) | 15.4 s (10.0–29.2) | 16.2 s (9.8–28.3) |
| Final judge range of successes | 0.267–0.276 m (median 0.272) | 0.264–0.278 m (median 0.271) |

- **The gate passed.** Both sets have at least 30 successes, and every scheduled slot was
  scored.
- **The one failure.** Set B spot 28, round 2 (the slower round-2 gains) timed out at 30.06 s.
  It was still requesting 0.014–0.018 m/s at a judge range of 0.2725 m. Its latch never engaged
  (exploratory: the servo's latch log lines, next bullet).
  - The pilot's slowest 62.2° demonstration was also spot 28 (29.9 s, round 1).
  - In round 1 here, spot 28 took 27.9 s.
- **The latch** (exploratory: from the servo's "stop latch" log lines in
  `sim_work/v4*/logs/jazzy/demos_r*.launch.log`, not in git). It engaged once per successful
  approach, at a measured range of 0.266–0.270 m (set A) and 0.267–0.270 m (set B), and never
  released during an approach.
  - It released only in 3 start transients in set A round 1, 0.15–0.16 s after the start, when a
    reading of the previous pose had set it (as in the pilot). The robot was searching with
    v = 0 either way.
  - Set B's single restart in band came 0.36 s after the start of spot 17 round 2 (exploratory:
    a post-hoc read of the bag with [exploratory/restart.py](exploratory/restart.py)). The judge's
    status still showed the previous pose's range (0.2684 m; spot 16 had ended at 0.2671 m)
    while the robot, already at its new start about 1 m from the box, began to drive. It is not
    a stop toggling near the box.

**Dataset stop labels.** Source: [dataset_v4_stop_labels.json](dataset_v4_stop_labels.json) and
[dataset_v4_62_stop_labels.json](dataset_v4_62_stop_labels.json) (`stop_labels.py dataset`; the
`servo_demo` successes, i.e. every frame BC trains or validates on). A stop frame has
|v| < 0.005 m/s; a full stop also |w| < 0.05 rad/s.

| Dataset | Episodes | Frames | Stop frames | Full stops | Full stops in the last 2 s | Episodes with a stop in the last 2 s | Final run of stop frames per episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Set A | 40 | 6,395 | 908 (14.2 %) | 459 (7.2 %) | 454 (7.1 %) | 40/40 | 10–13 (median 11) |
| Set A, train split | 32 | 5,062 | 801 (15.8 %) | 367 (7.3 %) | 362 (7.2 %) | 32/32 | 10–13 (median 11) |
| Set A, validation | 8 | 1,333 | 107 (8.0 %) | 92 (6.9 %) | 92 (6.9 %) | 8/8 | 11–12 |
| Set B | 39 | 6,676 | 1,379 (20.7 %) | 447 (6.7 %) | 433 (6.5 %) | 39/39 | 10–12 (median 11) |
| Set B, train split | 30 | 5,195 | 1,162 (22.4 %) | 344 (6.6 %) | 333 (6.4 %) | 30/30 | 10–12 (median 11) |
| Set B, validation | 9 | 1,481 | 217 (14.7 %) | 103 (7.0 %) | 100 (6.8 %) | 9/9 | 10–12 (median 11) |

- **Every full stop at the end.** Every frame of the last 2 s with a stop label is a full stop:
  454 of 454 in set A, 433 of 433 in set B. In v3, 163 of 6,545 frames (2.5 %) were such frames,
  and the final run of stop frames was a median 0 frames long.
- **Turns in place.** Nearly all the other stop frames are the servo turning to find or centre
  the marker, at the start and mid-episode: 237 + 212 turns in set A (449 of 454) and 435 + 497
  in set B (932 of 946). That fits longer searches through the narrower lens. The rest are
  mid-episode full stops, 5 in set A and 14 in set B; what they are was not investigated.
- **Idle starts.** The trim removed idle frames from the starts of 10 episodes (2.2 s) in set A
  and 7 episodes (1.9 s) in set B.

**Offline, before any rollout** (reported; it changes nothing). Source:
[offline_val_bc_v4.json](offline_val_bc_v4.json) and
[offline_val_bc_v4_62.json](offline_val_bc_v4_62.json). Each model ran on its own validation
frames. Predicted v is in m/s, clipped as `policy_runner` clips it.

| Model (its validation frames) | Frames | n | Label v, mean | Predicted v, mean (median) | Share < 0.005 / ≤ 0.02 |
|---|---|---:|---:|---:|---:|
| `bc_v4` (set A, 1,333) | full stops in the last 2 s | 92 | 0.000 | 0.0098 (0.0103) | 0 % / 100 % |
| | moving in the last 2 s | 68 | 0.018 | 0.0126 | 0 % / 100 % |
| | moving elsewhere | 1,158 | 0.084 | 0.0839 | 0.1 % / 6.6 % |
| `bc_v4_62` (set B, 1,481) | full stops in the last 2 s | 100 | 0.000 | 0.0064 (0.0061) | 32 % / 100 % |
| | moving in the last 2 s | 80 | 0.018 | 0.0098 | 7.5 % / 100 % |
| | stopped mid-episode (110 turns in place, 3 full stops) | 113 | 0.000 | −0.0005 | 100 % / 100 % |
| | moving elsewhere | 1,184 | 0.083 | 0.0790 | 0.2 % / 8.1 % |

On held-out docked views labelled a full stop, `bc_v4` predicted about 0.010 m/s, twice the
judge's 0.005 m/s. In closed loop it nevertheless ended below 0.005 m/s in most trials (section
2), and nearer the box than the servo (ground truth 0.264 m against 0.270 m in `clean_ref` and
`noise`). That is consistent with BC creeping until the view looks closer than where the servo
stops; it is an interpretation, not tested.

## 2. Closed loop (SIMULATION)

**Primary: the judge's success rule** (in band 0.18–0.32 m, marker within 10°, robot still and
request still, |v| ≤ 0.005 m/s and |w| ≤ 0.05 rad/s, for 1 s; 30 s limit). Source:
[SUMMARY_table.md](SUMMARY_table.md) and each condition's `report.md` (section "Paired
comparison with `servo`"). b = servo succeeded and the method failed, c = the reverse; Δ =
method − servo with Newcombe's 95 % CI. Times are median (IQR) over successes. The last two
columns are the validity check (section 3); the primary columns stay as scored.

| Condition | Method | Success k/n | Wilson 95 % CI | b / c vs servo | Exact p | Δ vs servo (95 % CI) | Time to dock, s | Fails | Successes flagged stale | k/n, flagged counted as failures |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| clean_ref | servo | 10/10 | 72.2–100.0 % | — | — | — | 14.7 (13.1–19.0) | none | 0 | 10/10 |
| clean_ref | bc_v4 | 9/10 | 59.6–98.2 % | 1 / 0 | 1 | −10.0 pp (−40.4 to +18.9) | 18.7 (17.8–23.9) | timeout 1 | 1 | 8/10 |
| noise | servo | 30/30 | 88.6–100.0 % | — | — | — | 14.2 (12.5–19.0) | none | 0 | 30/30 |
| noise | bc_v4 | 27/30 | 74.4–96.5 % | 3 / 0 | 0.25 | −10.0 pp (−25.6 to +3.1) | 20.0 (19.0–23.8) | timeout 3 | 10 | 17/30 |
| wide | servo | 27/30 | 74.4–96.5 % | — | — | — | 20.4 (18.0–21.3) | timeout 3 | 0 | 27/30 |
| wide | bc_v4 | 16/30 | 36.1–69.8 % | 11 / 0 | 0.000977 | −36.7 pp (−53.5 to −17.0) | 25.0 (23.3–27.6) | timeout 14 | 3 | 13/30 |
| hfov62 | servo | 30/30 | 88.6–100.0 % | — | — | — | 14.6 (12.6–18.9) | none | 0 | 30/30 |
| hfov62 | bc_v4 | 0/30 | 0.0–11.4 % | 30 / 0 | 1.86e-09 | −100.0 pp (−100.0 to −83.9) | — | timeout 30 | 0 | 0/30 |
| hfov62 | bc_v4_62 | 23/30 | 59.1–88.2 % | 7 / 0 | 0.0156 | −23.3 pp (−40.9 to −7.1) | 18.3 (15.1–23.2) | timeout 7 | 7 | 16/30 |

Every scheduled slot has exactly one scored result in every condition, no attempt was aborted,
and nothing collided.

**The pre-registered comparisons** (exact two-sided McNemar on pose_id × round pairs; Newcombe's
95 % CI of the paired difference; no multiplicity correction across the 6 tests). b = the
reference arm (servo for C1 and C3, `bc_v4` for C2) succeeded and the other failed; c = the
reverse; Δ = other − reference.

| | Pair | Condition | Pairs | b / c | Exact p | Δ success (95 % CI) | Source |
|---|---|---|---:|---:|---:|---:|---|
| C1 | `bc_v4` vs servo | clean_ref | 10 | 1 / 0 | 1 | −10.0 pp (−40.4 to +18.9) | [clean_ref/report.md](clean_ref/report.md) |
| C1 | `bc_v4` vs servo | noise | 30 | 3 / 0 | 0.25 | −10.0 pp (−25.6 to +3.1) | [noise/report.md](noise/report.md) |
| C1 | `bc_v4` vs servo | wide | 30 | 11 / 0 | 0.000977 | −36.7 pp (−53.5 to −17.0) | [wide/report.md](wide/report.md) |
| C1 | `bc_v4` vs servo | hfov62 | 30 | 30 / 0 | 1.86e-09 | −100.0 pp (−100.0 to −83.9) | [hfov62/report.md](hfov62/report.md) |
| C2 | `bc_v4_62` vs `bc_v4` | hfov62 | 30 | 0 / 23 | 2.38e-07 | +76.7 pp (+55.7 to +88.2) | [SUMMARY_table.md](SUMMARY_table.md) |
| C3 | `bc_v4_62` vs servo | hfov62 | 30 | 7 / 0 | 0.0156 | −23.3 pp (−40.9 to −7.1) | [hfov62/report.md](hfov62/report.md) |

The zone outcome (stop ignored) pairs the same way in C2: b / c = 0 / 30, p = 1.86e-09.

**Timing next to C2 and C3 (and the other conditions), per session and condition** (PREREG
section 6). Source: each condition's `report.md` (policy latency and safety gate tables; in
[hfov62/report.md](hfov62/report.md) "earlier1" is session 1 and "session" is session 2), the
segments in each `provenance.txt`, and the validity table of [SUMMARY_table.md](SUMMARY_table.md).

| Session | Condition | Segment (UTC) | Load average, start → end | Infer p50 / p99, ms | Tick to `/cmd_vel_in` p50 / p99, ms | Image age p50 / p99, ms | Gate missed deadlines | Gate period max, ms | Trials with a fallback zero |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1: servo + `bc_v4` | clean_ref | 14:35:03–14:43:08 | 1.24 → 5.07 | 69.4 / 122.6 | 80.9 / 139.4 | 56.3 / 188.0 | 25 of 23,863 (0.105 %) | 351.7 | `bc_v4` 1/10 |
| 1 | noise | 14:43:08–15:06:53 | 5.07 → 4.76 | 68.0 / 116.6 | 79.8 / 133.4 | 71.0 / 192.1 | 98 of 70,841 (0.138 %) | 335.7 | `bc_v4` 10/30 |
| 1 | wide | 15:06:53–15:35:45 | 4.76 → 5.03 | 66.2 / 119.7 | 78.3 / 136.2 | 64.1 / 192.1 | 137 of 85,967 (0.159 %) | 348.9 | `bc_v4` 8/30 |
| 1 | hfov62 | 15:35:45–16:03:12 | 5.03 → 4.60 | 60.5 / 116.6 | 71.8 / 134.2 | 71.3 / 192.4 | 94 of 82,066 (0.115 %) | 330.5 | `bc_v4` 9/30 |
| 2: `bc_v4_62` alone | hfov62 | 16:06:31–16:19:34 | 1.27 → 7.94 | 62.2 / 115.7 | 72.6 / 132.2 | 68.5 / 193.8 | 78 of 38,775 (0.201 %) | 331.3 | `bc_v4_62` 9/30 |

- **The two `hfov62` sessions had similar policy timing.** Inference p50 was 60.5 and 62.2 ms,
  p99 116.6 and 115.7 ms, and 9 of 30 trials per arm had a fallback zero. The rest was worse in
  session 2: the gate missed 0.201 % of its deadlines against 0.115 %, its callback p99 was
  6.2 ms against 3.7 ms, and the load average ended at 7.94 / 7.39 / 6.25 (1 / 5 / 15 min)
  against 4.60 / 5.43 / 5.37. The sessions also differed in whether `servo_baseline` ran beside
  `policy_runner`. Timing was not controlled.
- **The p99 inference time exceeded the 100 ms policy tick in every segment.** Fallback zeros
  were common, consistent with that (section 3).
- **Gate stalls.** Every segment had a worst gate period of 330–352 ms (20 ms nominal; v3:
  100–353 ms per condition); when they fell was not investigated.

**Secondary: zone reached, stop commanded, where each arm ended.** Source:
[SUMMARY_table.md](SUMMARY_table.md) (secondary and final-errors tables, median (IQR)), and the
judge's final range and |bearing| from each `trials.csv`. "Stop commanded" = the smallest |v|
among the controller's own requests in the last 1 s was below 0.005 m/s
(`own_v_min_last1s`).

| Condition | Method | Zone reached | Time to zone, s | Stop commanded | Min own requested \|v\|, last 1 s, m/s | GT lidar range, m | Judge range, m (median) | GT \|bearing\|, deg | Judge \|bearing\|, deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_ref | servo | 10/10 | 9.1 (8.0–13.2) | 10/10 | 0.000 (0.000–0.000) | 0.2697 (0.2696–0.2699) | 0.2697 | 3.6 (3.4–3.8) | 3.7 (3.4–3.8) |
| clean_ref | bc_v4 | 10/10 | 10.1 (8.5–14.2) | 9/10 | 0.002 (0.001–0.002) | 0.2649 (0.2643–0.2660) | 0.2649 | 3.2 (3.0–3.5) | 3.1 (2.8–3.4) |
| noise | servo | 30/30 | 9.3 (7.9–13.2) | 30/30 | 0.000 (0.000–0.000) | 0.2709 (0.2701–0.2721) | 0.2716 | 3.5 (3.3–3.8) | 3.5 (3.3–3.8) |
| noise | bc_v4 | 30/30 | 10.6 (9.5–14.7) | 27/30 | 0.003 (0.002–0.004) | 0.2642 (0.2612–0.2657) | 0.2643 | 3.8 (2.1–5.0) | 3.8 (2.0–4.9) |
| wide | servo | 30/30 | 15.1 (11.7–17.0) | 30/30 | 0.000 (0.000–0.000) | 0.2717 (0.2707–0.2734) | 0.2722 | 3.6 (3.4–3.7) | 3.6 (3.4–3.7) |
| wide | bc_v4 | 30/30 | 16.8 (12.7–18.6) | 18/30 | 0.004 (0.002–0.012) | 0.2665 (0.2638–0.2690) | 0.2653 | 3.7 (2.6–5.2) | 3.8 (2.5–5.3) |
| hfov62 | servo | 30/30 | 9.4 (7.8–13.2) | 30/30 | 0.000 (0.000–0.000) | 0.2704 (0.2698–0.2711) | 0.2705 | 3.6 (3.3–3.8) | 3.6 (3.4–3.8) |
| hfov62 | bc_v4 | 0/30 | — | 30/30 | 0.000 (0.000–0.000) | 0.3486 (0.3445–0.3503) | 0.3493 | 4.3 (4.0–4.6) | 4.3 (4.0–4.6) |
| hfov62 | bc_v4_62 | 30/30 | 14.0 (9.1–17.7) | 26/30 | 0.003 (0.003–0.004) | 0.2690 (0.2678–0.2702) | 0.2695 | 2.9 (2.3–3.6) | 2.9 (2.3–3.7) |

Stop commanded over every request (`cmd_v_min_last1s`, fallback zeros included; the sensitivity
line of PREREG section 5, counted from each `episodes_gt.csv`) is the same for every arm and
condition except two: `bc_v4` in `wide`, 23/30 (5 more failures, whose only requests below
0.005 m/s were fallback zeros), and `bc_v4_62` on `hfov62`, 27/30.

How the arms failed (from each `episodes_gt.csv`; every failure was a timeout):

- **`bc_v4` in `clean_ref` and `noise`.** All 4 failures were on pose 5 (`clean_ref` round 1 and
  `noise` rounds 1–3). Each ended in the zone, still requesting 0.012–0.020 m/s (its smallest
  own request of the last second). The servo succeeded on pose 5 in all 4 slots. In v2 (whose
  servo rows v3 reused), it failed there in 5 of 6 `noise` and `hfov62` trials, when lidar
  noise toggled its stop.
- **`bc_v4` in `wide`.** Its 14 failures were on poses 2, 3, 4, 7, 8 and 9. All 14 ended in
  the zone. The smallest own request of the last second was 0.004–0.029 m/s: 12 kept creeping
  above 0.005 m/s, and 2 dipped below it without holding it for the judge's 1 s.
- **The servo in `wide`.** Its 3 failures were all on pose 9, which it reached only at
  26.2–26.3 s of the 30 s limit, as in v2.
- **`bc_v4` on `hfov62`.** It stopped in all 30 trials (own requests ≈ 0) at a ground-truth
  lidar range of 0.337–0.355 m, beyond the band's 0.32 m edge, and never reached the zone.
- **`bc_v4_62` on `hfov62`.** Its 7 failures were pose 1 (3/3), pose 5 (3/3) and pose 7 (1/3).
  All reached the zone.
  - Pose 1: its smallest own request of the last second was 0.006–0.028 m/s.
  - Pose 5: it ended 9.3–10.5° off the marker (true bearing), at the judge's 10° limit; one of
    the three ended out of the zone.
  - Pose 7, round 2: its smallest own request was 0.003 m/s, but it did not hold the stop for
    1 s.

**P3** ([P3_stops.txt](P3_stops.txt), `bc_v4`, `clean_ref` + `noise`):

```text
bc_v4, clean_ref + noise: stop commanded in 36 of 40 scored trials (90.0 %)
ground-truth final lidar range of those 36 trials: median 0.2643 m (IQR 0.2618-0.2655, range 0.2587-0.2672; 36 with ground truth)
sensitivity, every request counted (cmd_v_min_last1s, policy_runner's fallback zeros included): stop commanded in 36 of 40 scored trials (90.0 %); ground-truth final lidar range of those: median 0.2643 m (IQR 0.2618-0.2655, range 0.2587-0.2672; 36 with ground truth)
```

## 3. Validity check: stale inputs at the end

Source: the table "Validity check: stale inputs at the end" of
[SUMMARY_table.md](SUMMARY_table.md); each episode's counts and flag are in its condition's
`episodes_gt.csv`. Counts are over the last 1 s before the end event: camera frames, then fresh
inferences as median (min). A success is flagged when the window had fewer than 5 frames or
fresh inferences, or any fallback zero.

| Condition | Method | Camera frames | Fresh inferences | Trials with a fallback zero | Successes flagged stale | Success k/n as scored | k/n, flagged counted as failures |
|---|---|---:|---:|---:|---:|---:|---:|
| clean_ref | servo | 15 (min 14) | — | — | 0 | 10/10 | 10/10 |
| clean_ref | bc_v4 | 15 (min 15) | 10 (min 9) | 1/10 | 1 | 9/10 | 8/10 |
| noise | servo | 15 (min 14) | — | — | 0 | 30/30 | 30/30 |
| noise | bc_v4 | 15 (min 14) | 10 (min 8) | 10/30 | 10 | 27/30 | 17/30 |
| wide | servo | 15 (min 15) | — | — | 0 | 27/30 | 27/30 |
| wide | bc_v4 | 15 (min 14) | 10 (min 8) | 8/30 | 3 | 16/30 | 13/30 |
| hfov62 | servo | 15 (min 14) | — | — | 0 | 30/30 | 30/30 |
| hfov62 | bc_v4 | 15 (min 14) | 9 (min 7) | 9/30 | 0 | 0/30 | 0/30 |
| hfov62 | bc_v4_62 | 15 (min 14) | 10 (min 8) | 9/30 | 7 | 23/30 | 16/30 |

- **What raised the flags.** Every flag comes from a fallback zero. No window had fewer than 14
  camera frames or 7 fresh inferences.
- **Not checked: none.** Every success had a ground-truth row and every counted topic, so there
  was no recording fault.
- **Far more fallback zeros than in v3.** 37 of the 130 policy trials had one in their last
  second (`bc_v4` 28 of 100, `bc_v4_62` 9 of 30), against 2 of v3's 200 BC episodes. This is
  consistent with the slow inference above (not checked event by event): p99 was 116–123 ms
  against the 100 ms tick, and a tick that overruns is followed at once by one that may find the
  frame too old.
- **The primary results stay as scored.** Counting the 21 flagged successes as failures gives the
  last column. The exploratory analysis below finds that in all 21, every own request of the last
  second met the judge's still rule.

## Exploratory (not pre-registered)

Everything in this section goes beyond PREREG.md and was decided after seeing the data. The
scripts are in [exploratory/](exploratory/) (`own_still.py` with its output `own_still.csv`,
`boundary.py` and `restart.py`, whose printed output was not saved; each takes the repository
root as its argument). The logs and the bags they read are not in git, so the scripts can be
read here but not rerun from the repository alone. Items
7–9 and the 5-minute windows in item 3 were added during the review, from the `episodes_gt.csv`
files, the session bags (item 3) and `scripts/sim_eval_summary.py episodes` run on the
demonstration bags (item 7); those scripts and outputs are not in git either.

1. **In all 21 flagged successes, the policy's own requests met the still rule.**
   - **Method.** `sim_eval_summary.read_episodes` read the session bags. In the last 1.00 s
     before each end event, it took the policy's own requests, i.e. every request but those of
     exactly (0, 0). The test was whether all of them met the judge's still rule (|v| ≤ 0.005
     m/s, |w| ≤ 0.05 rad/s).
   - **Own requests.** They did in all 21 flagged successes. The largest own |v| was 0.0035–0.0049
     m/s.
   - **The judge's ticks.** At none of the judge status ticks in that window was the latest
     recorded request a fallback zero while the policy's latest own request was not still.
   - **Caveat.** This uses the recorder's clock and a 1.00 s window, while the judge's hold can
     start a few tens of ms earlier. Within those limits, the flags here mark inputs that were
     briefly stale, not stops the policy did not make. The 17/30 and 16/30 above are therefore a
     conservative bound.
2. **BC's stops are small requests, not zeros.** In its successes, the largest own |v| of the last
   second had these medians and ranges:
   - `bc_v4`: 0.0042 m/s in `clean_ref`, 0.0045 in `noise` and 0.0044 in `wide` (0.0033–0.0050
     overall);
   - `bc_v4_62`: 0.0045 m/s on `hfov62` (0.0028–0.0049).

   The servo's were exactly 0. In 56 of BC's 75 successes that largest request was at least
   0.004 m/s, within 1 mm/s of the judge's limit (58 with the values rounded to 0.1 mm/s, as
   `own_still.csv` writes them). One `noise` success (pose 1, round 3) sent 0.00501 m/s 7 ms
   before its end event, after the judge's last tick. The smallest own request of a success's
   last second was 0.0000–0.0044 m/s (`own_v_min_last1s`, to 0.1 mm/s).

   16 of BC's 18 failures in `clean_ref`, `noise` and `wide` kept every own request of the last
   second above 0.005 m/s; the other 2 (in `wide`) dipped below it but did not hold the stop for
   1 s. So "stop commanded" here means requesting less than 5 mm/s, and the margin is small. The
   same 0.005 m/s threshold defines both the judge's rule and "stop commanded". A stricter
   threshold, or a policy output with an explicit stop, was not tested.
3. **Inference was 2–3 times slower than in v3.** In the same image and ONNX Runtime version,
   inference p50 was 60.5–69.4 ms and p99 115.7–122.6 ms. v3's sessions had p50 22–34 ms and
   p99 48–67 ms. The cause was not investigated. Two candidates fit the timing poorly:
   - **The servo node.** Session 2, without `servo_baseline`, was as slow as session 1's
     `hfov62` segment.
   - **The other work on the host** ("What ran"). The slowdown was already there in `clean_ref`
     (14:35–14:43, p50 69.4 ms), before the second workflow started, and in session 2, while it
     only polled. In 5-minute windows of `/policy/latency` in the session bags, p50 stayed at
     55.6–70.7 ms throughout. The highest window p99 of the run, 127.5 ms, came in `wide` at
     15:12–15:17, during that workflow's downloads.

   One difference from v3 is that the v4 containers were pinned to 3 cores (`SIM_CPUSET=0-2`),
   and v3's were not. The pilot ran only the servo pinned, so no policy arm had run pinned
   before session 1, and `policy_runner` creates its ONNX Runtime session with the default
   options. Session 2's load average rose from 1.27 to 7.94 while its container was the only
   one running; whether its policy, active in all its trials against half of session 1's,
   caused that was not checked. `export_onnx.py` measured a host CPU p50 of 34.3 ms (`bc_v4`,
   while the other agent loaded the host) and 14.2 ms (`bc_v4_62`). The counts with flagged
   successes as failures may therefore reflect this setup as much as the policies.
4. **The pinhole model, with `bc_v4`'s own stopping range.**
   - At 80°, `bc_v4` stopped at 0.2643 m (P3's median), not at the demonstrations' 0.270 m.
   - Matched through the 62.2° lens, that gives 1.391 × (0.2643 − 0.036) + 0.036 = 0.354 m.
   - The observed median on `hfov62` was 0.3486 m (ground truth) and 0.3493 m (judge). P5's
     pilot-based 0.361 m overestimated it by about 1 cm.
5. **The servo's latch in the evaluation.** It released only in 3 start transients, 0.12–0.14 s
   after a start (1 in `noise`, 2 in `wide`), and never during an approach. The servo commanded
   a stop in all 100 of its trials.
6. **Descriptive comparison with v3** (different rules and sessions, no tests).
   - On the 62.2° lens, v3's `bc_v2` and `bc_v2sw` also stopped short (median ground-truth range
     0.335 and 0.350 m, 0/30 each). `bc_v4` stopped at 0.349 m (0/30).
   - Among the BC arms of v2–v4, `bc_v4_62` is the first to succeed under that lens with its own
     requests below 0.005 m/s. BC v1's 20/30 there in v2 came from creep under the old 0.02 m/s
     rule.
7. **BC's failures ended more oblique to the box face than its successes.** Ground-truth heading
   off square at the end (`final_heading_deg` in each `episodes_gt.csv`):
   - `bc_v4` in `clean_ref` and `noise`: its 36 successes ended 0.1–21.9° off square, its 4
     failures (pose 5) 29.5–33.4°. The servo succeeded on pose 5 at 29.0–29.9°, taking
     27.0–29.4 s of the 30 s limit.
   - `bc_v4` in `wide`: successes median 6.7°, failures median 29.2°. All 6 of its trials that
     ended 33° or more off square failed (poses 3 and 4); the servo succeeded in 5 such trials.
   - `bc_v4_62` on `hfov62`: its pose 1 failures ended 19–32° off square, where the servo ended
     16–17°, and its pose 5 failures 25–26°.
   - The set-A demonstrations ended between −23.8° and +33.8° off square (median |heading|
     9.3°); only 3 of 40 ended 25° or more off, all on the positive side (spots 10 and 29).
     Training used no mirror augmentation (`hflip` false in `policy_meta.json`). Pose 5's
     endings (−29.5 to −33.4°) therefore lie beyond every set-A demonstration's ending on that
     side.
   - So BC's in-distribution failures fit oblique final approaches that the demonstrations hardly
     cover; this was not tested.
8. **The per-slot p values and the poses.** The tests treat the 3 rounds of a pose as
   independent. C1's 11 discordant pairs in `wide` come from 5 poses (2, 3, 4, 7, 8) and C3's 7
   from 3 poses (1, 5, 7), all in the servo's favour. A two-sided sign test over poses, counting
   each pose once, gives p = 0.0625 for C1 in `wide` and p = 0.25 for C3. C2 (8 poses,
   p = 0.0078) and C1 on `hfov62` (10 poses, p = 0.002) stay below 0.05.
9. **Margin to the 30 s limit.** Successes at 25 s or later: servo 14 of 97, `bc_v4` 15 of 52
   (8 of its 16 in `wide`), `bc_v4_62` 5 of 23; at 28 s or later: 4, 8 and 2. A different limit
   could change some discordant pairs, in either direction.

## What this shows

1. **The protocol change worked on the data.**
   - Every one of the 79 successful demonstrations ended with every request of its last second
     exactly v = 0.
   - No stop toggled near the box.
   - Full-stop frames at the end rose to 7.1 % of set A, from 2.5 % in v3.
2. **(a) Partly: `bc_v4`, trained with the unchanged recipe, brings its request below 0.005 m/s
   at the end, but not to zero.**
   - It commanded a stop in 36 of 40 `clean_ref` + `noise` trials (P3), at a median ground-truth
     range of 0.264 m, 6 mm closer than the servo. Those 36 are exactly its successes, which
     require it; none of its 4 failures commanded a stop.
   - It scored 9/10 and 27/30 against the servo's 10/10 and 30/30. C1 is not significant (P4),
     with low power. With the flagged successes counted as failures, it would be 17/30 in
     `noise`, and C1 there significant ("Predictions").
   - Its stop is a request of a few mm/s, not a zero (exploratory 2), so this design cannot
     separate a learned stop from creep under a lower threshold.
   - Its failures are all on pose 5, where its request stayed above the limit. Its endings there
     (29.5–33.4° off square) were more oblique than any set-A demonstration's on that side
     (exploratory 7).
3. **Outside the demonstrations' distribution, `bc_v4` is worse than the servo on 5 of the 10
   poses.** In `wide` it scored 16/30 against 27/30 (b / c = 11 / 0, p = 0.001 per slot; over
   poses, p = 0.06, exploratory 8). It reached the zone in all 30 trials but brought its own
   request under 0.005 m/s in only 18 (23 with fallback zeros counted).
4. **(b) The lens.** Here "the robot's lens" is `fake_robot`'s pinhole camera set to the Pi
   Camera v2's 62.2° HFOV, without distortion or blur.
   - **Trained at 80°.** `bc_v4` stopped short under that lens (0.349 m; 0/30), as P5
     predicted.
   - **Recorded through that lens.** `bc_v4_62` removed the stop-short failure: it reached the
     zone in 30/30 trials and ended at a median ground-truth range of 0.269 m. It succeeded in
     23/30 (C2: b / c = 0 / 23, p = 2.4e-07); its 7 failures had other causes (section 2).
   - **Still below the servo on poses 1, 5 and 7.** It stayed below the servo's 30/30 (C3:
     7 / 0, p = 0.016 per slot; over the 3 poses, p = 0.25, exploratory 8).
   - **Confounds.** C2 confounds the lens with the session (noise draws, load) and with a
     training set of 30 rather than 32 episodes.
5. **Timing was the weak point of this run.** Inference was 2–3 times slower than in v3, and,
   consistent with that, the validity check flagged 21 BC successes. Counted as failures, P4's
   and P5's counts fall below their thresholds, and C1 in `noise` becomes significant.
   Exploratory 1 found that in all 21, every own request of the last second met the still rule.

## What this does not show

- **Anything about the real robot.** This is simulation to simulation: `fake_robot`'s renderer,
  an ideal lidar, no motor latency.
- **Equivalence of `bc_v4` and the servo** in `clean_ref` and `noise`. With 10 and 30 pairs, a
  non-significant McNemar test is weak evidence; the CIs of the difference reach −40 and −26 pp.
- **Independent trials.** The 3 rounds of a pose are correlated, so the Wilson CIs and Newcombe
  CIs are too narrow and the p values too small.
- **Family-wise error control.** No correction was applied across the 6 tests (C1 in four
  conditions, C2, C3).
- **Same-session pairs for C2 and C3.** They pair arms from two sessions that share the seed,
  the poses and the schedule rule, not the noise draws. They did not share the load either:
  session 1 ran `servo_baseline` beside `policy_runner`, session 2 did not. So C2 and C3
  confound the lens with the noise draws and the load. Their policy timing was similar, but the
  gate's timing and the load were worse in session 2 (section 2), and none of it was controlled.
- **Controlled CPU load.** `SIM_CPUSET` pins the containers but does not reserve the cores, and
  the load average was sampled only at the start and end of each segment. Other work shared the
  host ("What ran"). Inference ran 2–3 times slower than in v3, for reasons not investigated.
  That handicap falls only on the policy arms, so the deficits in C1 and C3 may partly reflect
  it; that BC's failures cluster on a few poses argues against a pure timing effect, but this
  was not tested.
- **A training-set-size-matched C2.** `bc_v4_62` trained on 30 episodes (set B had a failed
  demonstration) and `bc_v4` on 32, and set B held out spot 10 as well.
- **Lens robustness.** `bc_v4_62` was trained and tested at 62.2°. BC is never told the HFOV;
  the servo and the judge are.
- **Training variance.** There is one training seed per model.
- **Stopping in the strict sense.** BC's stop is not a zero request: the largest own request of
  each success's last second was 0.003–0.005 m/s (exploratory 2). The success rule and "stop
  commanded" share the 0.005 m/s threshold.
- **Squareness.** The judge does not score it. At the end, the ground-truth heading was 14–19°
  off square (median) for every arm, and BC's failures in `clean_ref`, `noise` and `wide` ended
  more oblique than its successes (exploratory 7).
- **Comparisons with v2 and v3.** Their numbers were judged with a request counted as still up
  to 0.02 m/s, and their servo had no latch; any comparison with them is descriptive (PREREG
  section 6).

## Run record

- **Infrastructure failures: none.**
  - Every step ran once and completed: steps 1, 2, 4, 7 and 8 with `EXIT 0`, and steps 3, 5, 6
    and 9 with status 0.
  - Every completion check passed: "40 of 40 slots scored" for both sets; the 24 session-1
    files and the schedule line in every `report.md`; the 30 + 30 + 30 `hfov62` schedule.
  - No attempt was discarded or rerun.
- **Slots.** No slot was skipped, aborted, or repeated after a partial start ("Aborted attempts:
  None" in every report).
- **`fail controller_down`: none.**
- **Nodes that died: none.** "process has died" appears in none of the launch or servo logs of
  the demonstrations or of either session.
- **Deviations.**
  - The commands ran as written. Bookkeeping was added around them: UTC times, exit statuses,
    and in step 9 the table's printed copy was discarded (`--out` writes the same text).
  - One deviation in step 10: the experimenter's environment refused to create this file (a
    guard against agents writing report files). Its text was handed to the coordinator and
    saved after a review, with the review's edits listed in [RUN_LOG.md](RUN_LOG.md).
  - `RUN_LOG.md` exists as a timing log, kept at the coordinator's request. PREREG section 12
    lists it only after an infrastructure failure.
- **Provenance gap.** One parameter query failed (the `noise` segment, `servo_baseline.hfov_rad`;
  see "What ran").

## Files and how to reproduce

The commands are PREREG.md section 9, steps 1–9, run from commit b72ba83 with
`CA_BUNDLE=/root/.ccr/ca-bundle.crt HF_HUB_OFFLINE=1 SIM_CPUSET=0-2` and `PYTHON` set to a Python
3.12 with `requirements-tools.txt`. The containers ran
`mirror.gcr.io/library/ros@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617`.

- `demos_v4_endings.csv` / `.txt`, `demos_v4_62_endings.csv` / `.txt`: step 3.
- `dataset_v4_stop_labels.json`, `dataset_v4_62_stop_labels.json`, `offline_val_bc_v4.json`,
  `offline_val_bc_v4_62.json`: step 6.
- `training/v4/`, `training/v4_62/`: step 5 (policy meta, metrics, split, episode list, training
  and export logs, demonstration CSV, parameters and provenance).
- `clean_ref/`, `noise/`, `wide/`, `hfov62/`: steps 7 and 8. Each holds `trials.csv`,
  `schedule.csv`, `episodes_gt.csv` (with each episode's validity check), `report.md`,
  `success.png` and `provenance.txt`. `hfov62/` holds both sessions' rows.
- `SUMMARY_table.md`, `P3_stops.txt`: step 9.
- `SUMMARY.md`: this file (step 10).
- `RUN_LOG.md`: the timing log, the record of infrastructure failures (none), the step-10
  deviation and the review's edits to this file.
- `PREREG.md` and `pilot/`: committed before the run.
- The bags, datasets and models stay in `sim_work/v4` and `sim_work/v4_62`; they are not in git
  and not archived elsewhere. The models' sha256 are under "Models" and the datasets'
  fingerprints in `training/*/policy_meta.json`; the bags have no checksums here. The printed
  output of the checks of steps 5, 7 and 8 was not saved; [RUN_LOG.md](RUN_LOG.md) summarises
  it.
- `exploratory/`: the exploratory scripts and `own_still.csv` (added by the coordinator after the
  review; see [RUN_LOG.md](RUN_LOG.md)).
