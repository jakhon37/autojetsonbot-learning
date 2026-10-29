# Pre-registered remedy: stop-weighted BC (`bc_v2sw`)

> **SIMULATION ONLY.** Written 2026-10-29 05:25 UTC, after BC v2 was trained and checked
> offline on its validation episodes. It was written before the remedy model was trained and
> before any closed-loop rollout of BC v2 or the remedy. No test-condition result had been seen
> when the remedy and its weight were fixed.

## Evidence so far (training data and validation only)

- **The corrected judge did not make the servo demonstrations end on a stop.** Only 2 of the
  40 re-recorded demonstrations requested zero for the whole last second. In 23 more, the
  request toggled between 0 and about 0.019 m/s: lidar noise moved the range back and forth
  across the servo's 0.27 m stop range. In the other 15 (all from round 2, with `k_v` 0.8) it
  never reached zero. The judge counts a request of up to 0.02 m/s as still, and the servo's
  heading-scaled creep just outside 0.27 m is about 0.015–0.02 m/s, so these holds pass. Source:
  `demos_v2_endings.csv`.
- **Docked views carry mostly creep labels.** In the training split, the last 2 s of each
  demonstration hold 640 frames. Of these, 130 are labelled a full stop (|v| < 0.005 m/s and
  |w| < 0.05 rad/s) and 510 are creep at a median of 0.018 m/s. In the whole training split,
  only 168 of 5,265 frames (3.2 %) are full stops. Source: `dataset_v2_stop_labels.json`.
- **BC v2 predicts the mean of those labels.** On the 33 full-stop frames of the validation
  docked views, BC v2 predicts 0.013 m/s (mean). None of these predictions is below
  0.005 m/s, and all are at or below the judge's 0.02 m/s. BC v1 predicts 0.024 m/s on the same
  frames. The label mix predicts about 0.8 × 0.018 ≈ 0.015 m/s. Source: `offline_val_v1_v2.json`.

## Hypothesis

BC v2 cannot command a stop because of how its loss treats the labels. In the docked view, the
same image is labelled stop about 1 time in 5 and creep (~0.018 m/s) about 4 times in 5. The
errors involved (< 0.025 m/s) fall in the quadratic range of the Huber loss (delta 0.1 in
normalised units = 0.025 m/s), so the regression settles on their weighted mean, about
0.013 m/s. If BC v2 succeeds in closed loop, it will be because this creep is below the judge's
0.02 m/s threshold, not because it stops.

## Remedy (one change, fixed now)

The training loss is reweighted by class frequency. Every training frame whose label is a full
stop (|v| < 0.005 m/s and |w| < 0.05 rad/s) gets loss weight **W = 30**. All other frames get 1.
W is the inverse frequency in the training split, (5,265 − 168) / 168 = 30.3, rounded, and it is
not tuned further. Implementation: `tools/train_bc.py --stop-weight 30`, a weighted mean of
the per-frame Huber losses. Validation loss and MAE, and with them the checkpoint choice, stay
unweighted.

Everything else is unchanged from BC v2:

- the same dataset (`sim_work/v2/dataset`) and split (pose_id, seed 0, validation spots 20,
  24, 26 and 28);
- 5 epochs, batch 32, lr 3e-4, the same augmentation, ResNet18 from scratch on CPU;
- ONNX export with the same parity check.

The arm is called `bc_v2sw`. For the docked view, the weighted mean is about
(510 × 0.018) / (130 × 30 + 510) ≈ 0.002 m/s, below the 0.005 m/s stop threshold.

## Predictions and how they will be judged

The evaluation uses the same four conditions as BC v2: `clean_ref`, `noise`, `wide` and
`hfov62`, with the same seeds, schedules, judge and rounds. `bc_v2sw` runs in its own session,
and its rows are paired by pose × round with the servo / BC v1 rows (from the earlier v2
session) and with the BC v2 rows.

1. **Offline, on the validation frames** (checked before any rollout): the mean predicted v on
   full-stop docked frames is below 0.005 m/s, and the mean on moving frames is within
   0.01 m/s of BC v2's.
2. **Secondary outcome "stop commanded"** (min requested |v| in the last 1 s < 0.005 m/s):
   `bc_v2sw` commands a stop in most of its trials in every condition. BC v2 rarely does.
3. **Primary outcome (judge success):** reported with exact McNemar tests against servo,
   BC v1 and BC v2. No direction is predicted against BC v2. BC v2's creep can already pass the
   0.02 m/s threshold, and the stop weight risks stopping short of the band. Stopping short
   would show as timeouts with `end_in_zone` = 0 in `episodes_gt.csv`.

The results will be reported whatever they show, and nothing will be retrained or re-weighted
after the test conditions are seen.
