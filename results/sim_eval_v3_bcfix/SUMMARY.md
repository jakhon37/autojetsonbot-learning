# BC with stop examples (v3): re-recorded demonstrations, retrained BC, same evaluation

> **SIMULATION ONLY.** Every number here comes from `fake_robot` (the kinematic simulator) in a
> `ros:jazzy-ros-base` container on a 4-core x86 CPU, 29 October 2026. Nothing was measured on
> autoJetsonBot. Nothing here predicts how either method does on the real robot.

## Question

BC v1 (`sim_work/runs/bc/policy.onnx`, sha256 `7cfb1cbb…`) reached the docking zone in 100 of
100 v2 trials but never commanded a stop. Its demonstrations had been recorded under the old
judge, which accepted |odom v| ≤ 0.02 m/s as stopped, so every one of them ended while the servo
was still creeping. The judge now also requires the request on `/cmd_vel_in` to be still for
the 1 s hold.

Does re-recording the demonstrations with the corrected judge, and retraining with the
unchanged recipe, give a BC policy that stops?

## Design

- **Demonstrations: the same recipe, under the corrected judge.**
  `scripts/sim_e2e.sh --work sim_work/v2 demos` ran at commit 13afb1b on a clean tree. As for v1,
  it used:
  - the 20 training spots (`config/sim_demo_poses.yaml`, ids 10–29) and 2 rounds;
  - servo gains per round: round 1 `k_v` 1.0, `k_w` 1.2, `v_max` 0.15; round 2 `k_v` 0.8,
    `k_w` 1.5, `v_max` 0.12;
  - `randomize:=true`, with `seed:=1` for round 1 and `seed:=2` for round 2 (as in
    `sim_work/logs/jazzy/demos_r*.launch.log`).

  The noise draws are seeded but consumed in timing-dependent order, so they are not identical
  to v1's. The v1 bags, dataset and model are untouched. The new ones are in `sim_work/v2/`.
- **Dataset.** `bag_to_lerobot.py` built it from the two demonstration bags. The v1 dataset also
  held the 10 servo test-pose episodes, which `--methods servo_demo` filters out of training
  anyway. The current converter drops idle starts; the v1 conversion predates that.
- **BC v2.** It used v1's command unchanged: `train_bc.py --epochs 5 --batch-size 32 --device cpu
  --methods servo_demo --outcomes success --split-by pose_id --val-fraction 0.2 --seed 0`,
  ResNet18 from scratch. It was validated on the same held-out spots as v1 (20, 24, 26, 28). It
  took 27 min (5 epochs of 316–346 s) and was exported to ONNX opset 13. ONNX Runtime matches
  PyTorch to 1.8e-7.
- **Offline check before any rollout.** `scripts/stop_labels.py offline` ran each ONNX model on
  the same 1,280 v2 validation frames, grouped by label.
- **One pre-registered remedy, `bc_v2sw`.** [REMEDY_PREREG.md](REMEDY_PREREG.md) was written at
  05:25 UTC, after the offline check and before any closed-loop rollout of BC v2. The remedy's
  training started at 05:26:09 UTC; the file was last saved at 05:26:31, and the first rollout
  began at 05:59:10. Its single change from BC v2: full-stop frames (|v| < 0.005 m/s
  and |w| < 0.05 rad/s) weigh 30× in the loss. 30 is the inverse frequency in the training
  split, rounded and not tuned (`train_bc.py --stop-weight 30`).
- **Evaluation.** `scripts/sim_eval.sh` ran on Jazzy, ONNX Runtime 1.31.1 on CPU, with one
  policy arm per session: `--methods bc_v2`, then `--methods bc_v2sw`. It used the four
  conditions of `results/sim_eval_v2`:
  - `clean_ref`: test poses 0–9, randomisation off, 1 round (the setup of the `sim_e2e.sh` clean
    run);
  - `noise` (seed 201), `wide` (seed 202) and `hfov62` (seed 203): 10 poses × 3 rounds,
    randomisation on;
  - in every condition, the same `make_schedule.py --seed 2026` pose order (except the earlier
    `clean_ref` rows: the clean run went through poses 0–9 in order), judge and 30 s limit.

  `--pair-with` put the earlier rows in front of the new ones in each condition's `trials.csv`
  and `episodes_gt.csv`, so all arms pair on pose_id × round:
  - `servo` and `bc` (= BC v1): from `results/sim_eval_v2`, i.e. the v2 session, and for
    `clean_ref` the `sim_e2e.sh` clean run;
  - `bc_v2`: from the first new session (`sim_work/v2/eval_v2/jazzy`), for the second.
- **The arms come from different sessions.** A condition fixes the seed, the poses and the
  schedule rule. Its random stream (wheel bias and slip, start jitter, lighting, pixel and lidar
  noise), however, is consumed in timing-dependent order and was shared with the servo in the
  v2 session. So the draws of a given pose × round differ between `servo` / `bc`, `bc_v2` and
  `bc_v2sw`. `clean_ref` has no random draws.
- **Analysis.**
  - Outcomes, scoring, the exact McNemar tests and the Newcombe CIs come from
    `tools/analyze_trials.py` (each condition's `report.md`, baseline servo).
  - The tables below come from `scripts/sim_eval_summary.py table … --vs bc --vs bc_v2`, which
    pairs with `analyze_trials.compare_with_baseline`.
  - The ground-truth columns come from the bags.
  - Stop commanded = the smallest requested |v| in the last 1 s before the end was below
    0.005 m/s. The judge calls ≤ 0.02 m/s still.

## Models

| Arm | Model | sha256 | Validation MAE v / w | Training |
|---|---|---|---:|---|
| `bc` (v1) | `sim_work/runs/bc/policy.onnx` | `7cfb1cbb09dc074bec4888d9c604e57544c6dd700c93171615a767f89783bbc7` | 0.013 m/s / 0.030 rad/s (v1 validation) | 40 v1 demos, 5 epochs |
| `bc_v2` | `sim_work/v2/runs/bc/policy.onnx` | `f6e0b7540b05d0721a847c0285393606ce045d12c1b68ebe871edfb5b77bd180` | 0.009 / 0.024 | 40 v2 demos, 5 epochs, same recipe |
| `bc_v2sw` | `sim_work/v2sw/runs/bc/policy.onnx` | `e018fd406234f879c56fb76bbebafd4e80858b10b23fa979e70bee0cc7ecb42d` | 0.011 / 0.025 | as `bc_v2`, `--stop-weight 30` |

Dataset fingerprint (meta + parquet): `5e8d6f66…` for v2, `e8df5666…` for v1. Policy metas,
metrics, splits and logs are in [training/](training/).

## 1. The re-recorded demonstrations still mostly end on creep

Source: [demos_v2_endings.csv](demos_v2_endings.csv) and [demos_v1_endings.csv](demos_v1_endings.csv),
both from `stop_labels.py bags`.

- **v2: 40/40 demonstrations succeeded, but only 2 requested zero for the whole last second.**
  - In 23 more, the request switched between 0 and about 0.019 m/s: lidar noise (σ 5 mm) moved
    the range back and forth across the servo's 0.27 m stop range.
  - The other 15, all from round 2 (`k_v` 0.8), never requested zero in the last second. Their
    smallest request was 0.014–0.017 m/s.
  - The corrected judge accepts all of these. It calls a request of up to 0.02 m/s still, and
    the servo's heading-scaled creep just outside 0.27 m, `k_v`·(r − 0.25)·(1 − |bearing|/25°),
    is already below that.
- **v1:** no demonstration requested less than 0.021 m/s in its last second (median of the
  largest request: 0.041 m/s).
- **Durations:** the demonstrations got longer, with a median of 15.2 s (v2) against 11.3 s (v1),
  and ended nearer the box, at a lidar range of 0.266–0.276 m against 0.275–0.303 m.

## 2. Dataset stop labels

Source: [dataset_v2_stop_labels.json](dataset_v2_stop_labels.json) and
[dataset_v1_stop_labels.json](dataset_v1_stop_labels.json), from `stop_labels.py dataset`.

The counts below cover the `servo_demo` successes, i.e. every frame BC trains or validates on.
A *stop* frame has action |v| < 0.005 m/s. It is a *full stop* when |w| < 0.05 rad/s as well;
otherwise it is a *turn in place* (the servo searching for or centring the marker).

| Dataset | Episodes | Frames | Stop frames (\|v\| < 0.005) | Full stops | In the last 2 s | In the leading run | Elsewhere | Episodes with a stop in the last 2 s | Label v in the last 2 s, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 40 | 4,749 | 743 (15.6 %) | 19 (0.4 %) | 0 | 329 (322 turns) | 414 (402 turns) | 0/40 | 0.042 m/s |
| v2 | 40 | 6,545 | 939 (14.3 %) | 205 (3.1 %) | 164 (163 full) | 633 (all turns) | 142 (100 turns) | 25/40 | 0.018 m/s |
| v2 train split | 32 | 5,265 | 884 (16.8 %) | 168 (3.2 %) | 130 (all full) | 620 | 134 | 20/32 | 0.018 m/s |
| v2 validation | 8 | 1,280 | 55 (4.3 %) | 37 (2.9 %) | 34 (33 full) | 13 | 8 | 5/8 | 0.019 m/s |

Notes on the table:

- **End vs start.** The episode-end stops the fix was meant to add are 164 frames (2.5 % of
  frames). The run of stop frames that closes an episode is a median 0 frames long (maximum 11).
  In the last 2 s, 636 frames carry creep labels and 164 carry stops.
- **Idle start.** The trim removed 24 idle frames ((v, w) = (0, 0)) from the starts of 10
  episodes. The 633 stop frames left at the starts are all turns in place (v = 0, |w| =
  0.49–0.76 rad/s, mostly the servo's 0.5–0.6 rad/s search rate), not idle frames.

## 3. Offline, before any rollout: v1 and v2 on the same v2 validation frames

Source: [offline_val_v1_v2.json](offline_val_v1_v2.json) and
[offline_val_v2_v2sw.json](offline_val_v2_v2sw.json). Predicted v is in m/s, clipped as
`policy_runner` clips it. Frames come from validation spots 20, 24, 26 and 28, which none of the
models trained on.

| Frames | n | Label v, mean | bc v1: predicted v, mean | bc_v2: mean | bc_v2: share < 0.005 / ≤ 0.02 | bc_v2sw: mean | bc_v2sw: share < 0.005 / ≤ 0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full stops in the last 2 s (docked view) | 33 | 0.000 | 0.024 | 0.013 | 0 % / 100 % | 0.004 | 70 % / 100 % |
| Creep in the last 2 s (docked view) | 126 | 0.020 | 0.028 | 0.018 | 0 % / 75 % | 0.014 | 14 % / 68 % |
| Every stop label | 55 | 0.000 | 0.025 | 0.017 | 16 % / 85 % | 0.013 | 49 % / 85 % |
| Every moving frame | 1,225 | 0.082 | 0.081 | 0.081 | 0.1 % / 11 % | 0.079 | 2 % / 13 % |

**Diagnosis, before any rollout.**

- On the docked view, BC v2 predicts the mean of its labels. Those are about one stop to four
  creeps of ~0.019 m/s, so the mean is ≈ 0.015 m/s; BC v2 predicts 0.013 m/s. The errors involved
  are below 0.025 m/s, i.e. in the quadratic range of the Huber loss (delta 0.1 normalised), so
  the regression settles on the mean.
- Prediction: BC v2 would creep below the judge's 0.02 m/s rather than stop.
- The stop-weighted model brings the docked-view prediction to 0.004 m/s while moving frames are
  unchanged. This is pre-registered prediction 1, and it was met.

## 4. Closed loop (SIMULATION)

`servo` and `bc` rows are from the earlier v2 session (and the clean run). `bc_v2` and `bc_v2sw`
each ran in a session of their own. The noise draws differ between sessions; see Design.

**Primary: the judge's success rule** (in band 0.18–0.32 m, marker within 10°, robot and request
still for 1 s; 30 s limit). b / c and the exact McNemar p are against servo on pose × round
pairs. Times are median (IQR) over successes.

| Condition | Method | Success k/n | Wilson 95% CI | b / c vs servo | McNemar exact p | Time to dock, s | Fails by reason |
|---|---|---:|---:|---:|---:|---:|---|
| clean_ref | servo | 10/10 | 72–100% | — | — | 13.8 (12.2–18.1) | none |
| clean_ref | bc | 0/10 | 0–28% | 10 / 0 | 0.00195 | — | timeout 10 |
| clean_ref | bc_v2 | 10/10 | 72–100% | 0 / 0 | 1 | 14.5 (12.1–18.0) | none |
| clean_ref | bc_v2sw | 10/10 | 72–100% | 0 / 0 | 1 | 13.5 (12.1–16.8) | none |
| noise | servo | 27/30 | 74–97% | — | — | 14.8 (12.7–18.1) | timeout 3 |
| noise | bc | 3/30 | 3–26% | 24 / 0 | 1.19e-07 | 13.2 (13.2–21.7) | timeout 27 |
| noise | bc_v2 | 30/30 | 89–100% | 0 / 3 | 0.25 | 15.1 (12.6–18.7) | none |
| noise | bc_v2sw | 30/30 | 89–100% | 0 / 3 | 0.25 | 13.9 (11.9–18.1) | none |
| wide | servo | 25/30 | 66–93% | — | — | 21.0 (17.8–22.0) | timeout 5 |
| wide | bc | 1/30 | 1–17% | 24 / 0 | 1.19e-07 | 21.8 (21.8–21.8) | timeout 29 |
| wide | bc_v2 | 25/30 | 66–93% | 3 / 3 | 1 | 20.2 (18.8–21.7) | timeout 5 |
| wide | bc_v2sw | 22/30 | 56–86% | 5 / 2 | 0.453 | 19.2 (16.2–22.1) | timeout 8 |
| hfov62 | servo | 28/30 | 79–98% | — | — | 15.1 (12.8–18.7) | timeout 2 |
| hfov62 | bc | 20/30 | 49–81% | 8 / 0 | 0.00781 | 25.0 (21.9–26.9) | timeout 10 |
| hfov62 | bc_v2 | 0/30 | 0–11% | 28 / 0 | 7.45e-09 | — | timeout 30 |
| hfov62 | bc_v2sw | 0/30 | 0–11% | 28 / 0 | 7.45e-09 | — | timeout 30 |

The paired difference against servo (Newcombe 95% CI) is:

- `bc_v2`: +0.0 pp (−27.8 to +27.8) in `clean_ref`, +10.0 pp (−3.1 to +25.6) in `noise`,
  +0.0 pp (−17.9 to +17.9) in `wide` and −93.3 pp (−98.2 to −74.8) in `hfov62`;
- `bc_v2sw`: the same as `bc_v2`, except −10.0 pp (−28.0 to +8.5) in `wide`.

**Secondary: zone reached (stop ignored) and what was requested.** Requested speeds are on
`/cmd_vel_in`, before the gate, as medians (IQR) over trials.

| Condition | Method | Zone reached k/n | b / c vs servo | Time to zone, s | Requested v, last 2 s, m/s | Min requested \|v\|, last 1 s, m/s | Stop commanded | Collisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean_ref | servo | 10/10 | — | 9.2 (8.1–13.2) | 0.020 (0.019–0.020) | 0.000 (0.000–0.000) | 8/10 | 0/10 |
| clean_ref | bc | 10/10 | 0 / 0 | 10.5 (9.0–14.1) | 0.029 (0.027–0.030) | 0.028 (0.026–0.030) | 0/10 | 0/10 |
| clean_ref | bc_v2 | 10/10 | 0 / 0 | 10.2 (8.8–13.8) | 0.019 (0.019–0.020) | 0.017 (0.015–0.017) | 0/10 | 0/10 |
| clean_ref | bc_v2sw | 10/10 | 0 / 0 | 10.3 (9.0–14.0) | 0.019 (0.018–0.019) | 0.012 (0.011–0.015) | 0/10 | 0/10 |
| noise | servo | 30/30 | — | 9.3 (7.9–13.3) | 0.018 (0.017–0.019) | 0.000 (0.000–0.000) | 30/30 | 0/30 |
| noise | bc | 30/30 | 0 / 0 | 10.2 (8.8–14.2) | 0.027 (0.026–0.028) | 0.026 (0.025–0.028) | 0/30 | 0/30 |
| noise | bc_v2 | 30/30 | 0 / 0 | 10.7 (8.8–14.5) | 0.019 (0.019–0.020) | 0.017 (0.016–0.017) | 1/30 | 0/30 |
| noise | bc_v2sw | 30/30 | 0 / 0 | 10.3 (9.2–14.4) | 0.019 (0.018–0.019) | 0.012 (0.011–0.015) | 1/30 | 0/30 |
| wide | servo | 30/30 | — | 14.9 (12.0–17.0) | 0.018 (0.017–0.019) | 0.000 (0.000–0.000) | 27/30 | 0/30 |
| wide | bc | 30/30 | 0 / 0 | 16.6 (12.7–18.9) | 0.028 (0.026–0.030) | 0.027 (0.025–0.028) | 0/30 | 0/30 |
| wide | bc_v2 | 30/30 | 0 / 0 | 16.7 (14.4–18.7) | 0.019 (0.018–0.020) | 0.016 (0.015–0.018) | 0/30 | 0/30 |
| wide | bc_v2sw | 27/30 | 3 / 0 | 16.9 (14.7–19.3) | 0.019 (0.019–0.020) | 0.015 (0.012–0.018) | 0/30 | 0/30 |
| hfov62 | servo | 30/30 | — | 9.3 (7.7–13.6) | 0.018 (0.018–0.019) | 0.000 (0.000–0.000) | 30/30 | 0/30 |
| hfov62 | bc | 30/30 | 0 / 0 | 15.2 (13.8–19.3) | 0.020 (0.019–0.022) | 0.019 (0.018–0.022) | 0/30 | 0/30 |
| hfov62 | bc_v2 | 0/30 | 30 / 0 | — | 0.000 (0.000–0.001) | 0.000 (0.000–0.001) | 29/30 | 0/30 |
| hfov62 | bc_v2sw | 0/30 | 30 / 0 | — | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 30/30 | 0/30 |

**Paired between the BC arms** (judge success; Δ with Newcombe's 95% CI; zone reached paired the
same way).

| Condition | Pair | Pairs | b / c | McNemar exact p | Δ success (95% CI) | Zone b / c |
|---|---|---:|---:|---:|---:|---:|
| clean_ref | bc_v2 vs bc | 10 | 0 / 10 | 0.00195 | +100.0 pp (+60.8 to +100.0) | 0 / 0 |
| noise | bc_v2 vs bc | 30 | 0 / 27 | 1.49e-08 | +90.0 pp (+70.7 to +96.5) | 0 / 0 |
| wide | bc_v2 vs bc | 30 | 0 / 24 | 1.19e-07 | +80.0 pp (+58.5 to +89.7) | 0 / 0 |
| hfov62 | bc_v2 vs bc | 30 | 20 / 0 | 1.91e-06 | −66.7 pp (−80.8 to −45.5) | 30 / 0 |
| clean_ref | bc_v2sw vs bc_v2 | 10 | 0 / 0 | 1 | +0.0 pp (−27.8 to +27.8) | 0 / 0 |
| noise | bc_v2sw vs bc_v2 | 30 | 0 / 0 | 1 | +0.0 pp (−11.4 to +11.4) | 0 / 0 |
| wide | bc_v2sw vs bc_v2 | 30 | 4 / 1 | 0.375 | −10.0 pp (−26.1 to +6.0) | 3 / 0 |
| hfov62 | bc_v2sw vs bc_v2 | 30 | 0 / 0 | 1 | +0.0 pp (−11.4 to +11.4) | 0 / 0 |

`bc_v2sw` against `bc` is in [SUMMARY_table.md](SUMMARY_table.md): +100, +90, +70 and −66.7 pp.
So is the final-error table, with ground-truth squareness and bumper gap.

**Final lidar range, from the judge.**

- Successes: the servo ended at 0.268–0.270 m, `bc_v2` at 0.272–0.273 m and `bc_v2sw` at
  0.276–0.277 m (medians over the conditions with an 80° lens).
- `hfov62`: `bc_v2` stopped at 0.315–0.344 m (median 0.336 m) and `bc_v2sw` at 0.326–0.363 m
  (median 0.349 m), outside the band's 0.32 m edge. The exception is `bc_v2` pose 1 round 3,
  which stopped on the edge (judge range 0.313–0.326 m in its last 10 s): lidar noise toggled
  `in_band`, so the zone never held for 1 s. Neither arm reached the zone.

## What this shows

1. **Re-recording under the corrected judge only partly fixed the data.**
   - Episode-end stop labels went from none to 164 frames (2.5 %), in 25 of 40 demonstrations.
   - Most of the docked views are still labelled with creep of ~0.019 m/s. The judge's 0.02 m/s
     "still" threshold sits above the servo's heading-scaled creep, and lidar noise toggles the
     servo's 0.27 m stop.
   - Round 2 (`k_v` 0.8) never ended on a zero request.
2. **BC v2 learned to creep below the threshold, not to stop.** With the 80° lens, its smallest
   request in the last second had a median of 0.016–0.017 m/s per condition: stop commanded in
   1 of 70 trials. The judge counts that as still, so:
   - BC v2 went from 0/10, 3/30 and 1/30 (BC v1) to 10/10, 30/30 and 25/30, with p ≤ 1.2e-7
     against BC v1 in `noise` and `wide`;
   - it was not significantly different from the servo (b / c = 0 / 0, 0 / 3, 3 / 3).

   This is success at the 0.02 m/s threshold, the same mechanism that gave BC v1 its few
   successes, with the creep now just under the line instead of just over it. BC v2 succeeded on
   noise pose 5, where the servo fails, only because BC v2 does not read the lidar whose noise
   toggles the servo's stop.
3. **BC v2 did learn that the request falls as the marker grows, and the narrower lens exposes
   it.**
   - The 62.2° lens magnifies the image by tan 40° / tan 31.1° = 1.39. A marker at 0.336 m lidar
     range therefore looks about as large as it does at ≈ 0.25 m through the 80° training lens
     (pinhole model, camera 0.036 m ahead of the lidar). That is slightly closer than any
     demonstration ended (0.266–0.276 m).
   - BC v2 commanded a full stop in 29 of 30 `hfov62` trials, but at 0.315–0.344 m, just
     outside the band (one trial on its edge, where lidar noise kept the 1 s hold from
     completing): 0/30.
   - BC v1 scored 20/30 there only because its creep, slowed by the same cue, carried it into
     the band. This confirms the v2 summary: that 20/30 was not lens robustness.
4. **The pre-registered remedy changed the offline numbers, not the closed-loop outcome.**
   - Offline (prediction 1, met): stop-labelled docked frames fell from 0.013 to 0.004 m/s, and
     moving frames were unchanged.
   - In closed loop, the smallest request of the last second fell from 0.016–0.017 to
     0.012–0.015 m/s (medians per condition).
     Prediction 2 (stops commanded in most trials) failed: 1 of 70 with the 80° lens. The judge's
     hold completes, and the episode ends, at about 0.277 m, while the request is still
     0.012–0.019 m/s. That is before the robot reaches the view the stop labels come from.
   - Under the 62.2° lens, `bc_v2sw` stopped in 30/30 trials, even earlier than BC v2 (median
     0.349 m): 0/30.
   - Judge success did not differ from BC v2's (`wide` 22/30 vs 25/30, b / c = 4 / 1, p = 0.375).
   - The stop-short risk named in advance appeared only under the lens mismatch. No `wide`
     failure ended stopped outside the band. The 3 `bc_v2sw` trials that never reached the zone
     ended still driving or creeping above 0.02 m/s (one 0.65 m away, still at 0.11 m/s).
5. **The primary metric cannot tell creep from stopping at this threshold.** Read the "stop
   commanded" column next to it. A test of learned stopping needs demonstrations that end on a
   commanded stop, for example:
   - a judge or demo-recording rule that needs the request below 0.005 m/s;
   - or a servo with hysteresis on its stop range, so lidar noise cannot toggle it.

   That would be a change of protocol, and it was not tried here.

## What this does not show

- **Anything about the real robot.** This is simulation to simulation, with the training
  renderer, no motor latency and an ideal lidar.
- **Same-session pairing.** The four arms ran in three sessions with different noise draws per
  pose × round. The McNemar tests pair slots, not identical disturbances. `clean_ref` is the
  exception: it has no draws.
- **Independent trials.** The 3 rounds of a pose are correlated, so the Wilson CIs and the tests
  are optimistic.
- **Training variance.** There is one training seed per model, so `bc_v2` against `bc_v2sw`
  differences of a few trials are within what a second seed could change.
- **A fair learned-vs-classical comparison.** BC v2's parity with the servo comes from the
  0.02 m/s threshold.
- **Lens robustness for any BC.** BC was never told the HFOV, while the servo and the judge were.

## Other details

- **Timing.**
  - Policy inference p50 was 22–34 ms and p99 48–67 ms in the two new sessions, against
    20–23 ms and 57–64 ms in the v2 session (that session ran the servo node alongside).
  - The safety gate missed 0.05–0.16 % of its deadlines, with a worst single period of
    100–353 ms per condition. Per-session latency and gate tables are in each `report.md`:
    `sim-cpu-onnxruntime` / `session` = the `bc_v2sw` session, `earlier2` = the `bc_v2` session.
  - The `bc_v2` session's `noise` condition was the slowest (infer p50 33.6 ms). Nothing else ran
    during the trials except log polling; a host test of about 1 s ran during the container build
    before `clean_ref`.
- **Training.** The BC v2 training stage logged 6 changed paths but not which ones (this folder
  was created 15 s later). `tools/bag_to_lerobot.py` was unchanged. The remedy was trained after
  `--stop-weight` was added to `train_bc.py`. When the flag is 1, the code takes the original
  loss path, so BC v2's training is reproducible with the current file. `train_bc.py` was last
  saved at 05:26:24, 15 s after the remedy's training started, so the file on disk may differ
  slightly from the code that trained `bc_v2sw`.
- **Provenance.** Each condition's `provenance.txt` holds this session's segment, commit, image
  digest, ONNX Runtime version and model sha256, followed by the provenance of every earlier
  session whose rows were merged.

## Files and how to reproduce

```bash
PY=<venv>/bin/python
bash scripts/sim_e2e.sh --work sim_work/v2 demos train                   # demos (~16 min), dataset, BC v2 (~27 min)
$PY tools/train_bc.py --root sim_work/v2/dataset --repo-id local/jbl_sim --out sim_work/v2sw/runs/bc --epochs 5 \
  --batch-size 32 --device cpu --methods servo_demo --outcomes success --split-by pose_id --val-fraction 0.2 \
  --seed 0 --stop-weight 30
$PY tools/export_onnx.py sim_work/v2sw/runs/bc/model.pt --out sim_work/v2sw/runs/bc/policy.onnx \
  --dataset-root sim_work/v2/dataset --repo-id local/jbl_sim
$PY scripts/stop_labels.py bags sim_work/v2/bags/jazzy/demos_r1 sim_work/v2/bags/jazzy/demos_r2 --out …/demos_v2_endings.csv
$PY scripts/stop_labels.py dataset --root sim_work/v2/dataset --split sim_work/v2/runs/bc/split.json --out …
$PY scripts/stop_labels.py offline --root sim_work/v2/dataset --split sim_work/v2/runs/bc/split.json \
  --model bc_v1=sim_work/runs/bc/policy.onnx --model bc_v2=sim_work/v2/runs/bc/policy.onnx --out …
PYTHON=$PY bash scripts/sim_eval.sh --work sim_work/v2 --results results/sim_eval_v3_bcfix --methods bc_v2 \
  --pair-with results/sim_eval_v2 clean_ref noise wide hfov62                       # ~45 min
PYTHON=$PY bash scripts/sim_eval.sh --work sim_work/v2sw --results results/sim_eval_v3_bcfix --methods bc_v2sw \
  --pair-with results/sim_eval_v2 --pair-with sim_work/v2/eval_v2/jazzy clean_ref noise wide hfov62
$PY scripts/sim_eval_summary.py table --results results/sim_eval_v3_bcfix --conditions clean_ref noise wide hfov62 \
  --vs bc --vs bc_v2 --out results/sim_eval_v3_bcfix/SUMMARY_table.md
```

Contents of this folder:

- Each condition folder (`clean_ref/`, `noise/`, `wide/`, `hfov62/`) holds `trials.csv`,
  `episodes_gt.csv` and `schedule.csv` (all arms, merged), plus `report.md`, `success.png` and
  `provenance.txt`.
- `training/` holds the demonstration CSV and provenance, the dataset episode list, and each
  model's policy meta, metrics, split and training log.
- The bags, datasets and models stay in `sim_work/v2` and `sim_work/v2sw`; they are not in git.
