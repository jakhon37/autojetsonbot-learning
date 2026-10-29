# Harder simulation evaluation (v2): servo vs BC

> **SIMULATION ONLY.** Every number here comes from `fake_robot` (the kinematic simulator) in a `ros:jazzy-ros-base` container on a 4-core x86 CPU, 29 October 2026. Nothing was measured on autoJetsonBot, and nothing here predicts how either method does on the real robot.

## Why a second evaluation
The clean comparison in `results/sim_e2e/jazzy` ran with randomisation off and 1 round per pose. The servo scored 10/10 (the ceiling) and BC 0/10 (the floor; every trial timed out because BC never asks to stop). With randomisation off, a second round would replay the same trajectories, so that run cannot show how either method degrades or why BC fails.

## Design
- **Arms.** `servo` is `servo_baseline` with the default gains. `bc` is `policy_runner` running `sim_work/runs/bc/policy.onnx` (sha256 7cfb1cbb…) with ONNX Runtime 1.31.1 on the CPU. This is the clean run's model, not retrained: the later fixes change neither its preprocessing (its dataset stores video frames, already uint8) nor its dataset format or I/O contract. It was trained on 40 `servo_demo` demonstrations recorded with `randomize:=true` at distance 0.70–1.70 m, |lateral| ≤ 0.55 m, |heading| ≤ 45°, with the 80° lens.
- **Schedule.** `tools/make_schedule.py --methods servo,bc --poses 10 --rounds 3 --seed 2026`, interleaved. Both controllers ran in one launch, each answering only to its own method name. That gives 30 paired pose × round slots per condition: 180 trials, all scored, no aborts.
- **Conditions.** `randomize:=true` in all of them.
  - `noise` (seed 201): test poses 0–9 with the default randomisation (brightness gain 0.7–1.3 and offset ±20, pixel noise σ 3, wheel bias σ 2 % and slip σ 3 %, lidar noise σ 5 mm, start jitter ±3 cm / ±3°). These are the magnitudes the demonstrations were recorded with.
  - `wide` (seed 202): as `noise`, from 10 poses outside the demonstrations' distance / lateral / heading box (`config/sim_eval_wide_poses.yaml`: far up to 2.3 m, wide up to 0.8 m, steep up to 60° with the marker out of view, near 0.6 m), chosen from geometry before any trial.
  - `hfov62` (seed 203): as `noise`, with the Pi Camera v2 lens (62.2°, 1.0856 rad). The judge and the servo were given the true HFOV; BC was trained at 80°.
- **Primary outcome.** The unchanged judge rule: band 0.18–0.32 m, |bearing| ≤ 10°, odometry and `/cmd_vel_in` still (≤ 0.02 m/s, ≤ 0.05 rad/s) for 1 s, 30 s limit. Statistics from `tools/analyze_trials.py`: Wilson 95% CI and exact McNemar against servo.
- **Secondary outcomes.** Defined after the clean result but before any v2 trial: zone reached (`in_band` and `aligned` for ≥ 1 s, stop ignored), the requested speeds and fake_robot's ground truth, all computed by `scripts/sim_eval_summary.py` from the bags.

## Results (SIMULATION)
**Primary: the judge's success rule** (in band, marker within 10°, robot and request still for 1 s; 30 s limit). Times are median (IQR) over successes.

| Condition | Method | Success k/n | Wilson 95% CI | b / c vs servo | McNemar exact p | Time to dock, s | Fails by reason |
|---|---|---:|---:|---:|---:|---:|---|
| clean_ref | servo | 10/10 | 72–100% | — | — | 13.8 (12.2–18.1) | none |
| clean_ref | bc | 0/10 | 0–28% | 10 / 0 | 0.00195 | — | timeout 10 |
| noise | servo | 27/30 | 74–97% | — | — | 14.8 (12.7–18.1) | timeout 3 |
| noise | bc | 3/30 | 3–26% | 24 / 0 | 1.19e-07 | 13.2 (13.2–21.7) | timeout 27 |
| wide | servo | 25/30 | 66–93% | — | — | 21.0 (17.8–22.0) | timeout 5 |
| wide | bc | 1/30 | 1–17% | 24 / 0 | 1.19e-07 | 21.8 (21.8–21.8) | timeout 29 |
| hfov62 | servo | 28/30 | 79–98% | — | — | 15.1 (12.8–18.7) | timeout 2 |
| hfov62 | bc | 20/30 | 49–81% | 8 / 0 | 0.00781 | 25.0 (21.9–26.9) | timeout 10 |

**Secondary: docking zone reached, stop ignored** (the judge's in_band and aligned held together for ≥ 1 s at some point, from `/dock_judge/status`; not the success rule). Time to zone and the requested speeds (`/cmd_vel_in`, before the gate) are median (IQR) over trials; stop commanded = the smallest requested |v| in the last 1 s was below 0.005 m/s (the judge's threshold for still is 0.02 m/s); collisions from fake_robot's ground truth.

| Condition | Method | Zone reached k/n | Wilson 95% CI | b / c vs servo | McNemar exact p | Time to zone, s | Requested v, last 2 s, m/s | Min requested \|v\|, last 1 s, m/s | Stop commanded | Collisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_ref | servo | 10/10 | 72–100% | — | — | 9.2 (8.1–13.2) | 0.020 (0.019–0.020) | 0.000 (0.000–0.000) | 8/10 | 0/10 |
| clean_ref | bc | 10/10 | 72–100% | 0 / 0 | 1 | 10.5 (9.0–14.1) | 0.029 (0.027–0.030) | 0.028 (0.026–0.030) | 0/10 | 0/10 |
| noise | servo | 30/30 | 89–100% | — | — | 9.3 (7.9–13.3) | 0.018 (0.017–0.019) | 0.000 (0.000–0.000) | 30/30 | 0/30 |
| noise | bc | 30/30 | 89–100% | 0 / 0 | 1 | 10.2 (8.8–14.2) | 0.027 (0.026–0.028) | 0.026 (0.025–0.028) | 0/30 | 0/30 |
| wide | servo | 30/30 | 89–100% | — | — | 14.9 (12.0–17.0) | 0.018 (0.017–0.019) | 0.000 (0.000–0.000) | 27/30 | 0/30 |
| wide | bc | 30/30 | 89–100% | 0 / 0 | 1 | 16.6 (12.7–18.9) | 0.028 (0.026–0.030) | 0.027 (0.025–0.028) | 0/30 | 0/30 |
| hfov62 | servo | 30/30 | 89–100% | — | — | 9.3 (7.7–13.6) | 0.018 (0.018–0.019) | 0.000 (0.000–0.000) | 30/30 | 0/30 |
| hfov62 | bc | 30/30 | 89–100% | 0 / 0 | 1 | 15.2 (13.8–19.3) | 0.020 (0.019–0.022) | 0.019 (0.018–0.022) | 0/30 | 0/30 |

**Final errors** at the end of each scored trial, median (IQR). Range error and bearing from the judge (trials.csv); GT = fake_robot ground truth (heading off square to the box face, sideways offset from the marker centre line, bumper-to-box gap).

| Condition | Method | Range error, cm | \|Bearing\|, deg | GT \|lateral\|, cm | GT \|heading off square\|, deg | GT bumper gap, cm |
|---|---|---:|---:|---:|---:|---:|
| clean_ref | servo | 2.0 (2.0–2.0) | 3.7 (3.5–3.8) | 10.7 (4.4–12.8) | 16.6 (6.8–21.0) | 20.2 (19.2–22.4) |
| clean_ref | bc | 2.0 (1.3–3.0) | 1.9 (1.2–2.3) | 9.3 (3.1–10.4) | 17.8 (6.1–18.9) | 15.6 (15.5–16.2) |
| noise | servo | 1.9 (1.8–2.0) | 3.6 (3.4–3.8) | 9.6 (2.6–12.2) | 15.9 (4.1–20.2) | 20.2 (19.2–22.7) |
| noise | bc | 1.3 (0.6–2.3) | 2.0 (1.4–3.2) | 10.0 (2.9–12.4) | 18.7 (5.5–23.2) | 15.9 (15.8–16.3) |
| wide | servo | 1.9 (1.7–2.0) | 3.6 (3.4–3.8) | 8.2 (3.5–16.6) | 14.0 (5.6–27.7) | 20.7 (17.2–22.5) |
| wide | bc | 1.5 (1.0–2.4) | 1.8 (0.5–2.8) | 9.4 (4.9–15.2) | 15.5 (8.2–26.8) | 16.8 (15.9–18.0) |
| hfov62 | servo | 1.9 (1.8–2.0) | 3.6 (3.4–3.8) | 10.6 (2.7–13.4) | 18.4 (3.3–21.4) | 19.6 (19.0–22.8) |
| hfov62 | bc | 0.6 (0.3–1.2) | 1.0 (0.4–2.2) | 9.9 (2.9–11.6) | 17.3 (6.0–19.6) | 18.2 (17.0–20.3) |

The paired difference bc − servo (Newcombe 95% CI) is −80.0 pp (−89.2 to −57.9) for `noise`, −80.0 pp (−89.7 to −58.5) for `wide` and −26.7 pp (−44.1 to −8.9) for `hfov62`.

## What this shows and what it does not
- **The primary metric separates the methods, but on stopping, not on reaching the box.** The servo beats BC in every condition (b / c = 24 / 0, 24 / 0 and 8 / 0; exact McNemar p ≤ 0.008). Yet both reached the zone in 30/30 trials everywhere, BC slightly later (median time to zone 10.2 vs 9.3 s, 16.6 vs 14.9 s and 15.2 vs 9.3 s). BC also generalised its approach to all 10 out-of-distribution poses.
- **BC never commanded a stop in any of its 100 trials.** Its smallest request in the last second was always ≥ 0.016 m/s, and every BC success is creep under the 0.02 m/s threshold (0.016–0.0195 m/s). Its 20/30 under the 62.2° lens is not lens robustness: the narrower view makes the marker look closer, so BC creeps at about 0.019 instead of 0.027 m/s. The servo commanded an exact zero in 87 of 90 v2 trials. Even so, 2 of its 10 clean successes were declared while it still asked for 0.017–0.019 m/s, so both methods are judged by the same threshold, and BC's rate swings with it.
- **The servo's 10 v2 failures come from three mechanisms.** Pose 5 (5 of 6 `noise` and `hfov62` trials): lidar noise toggles its request across its 0.27 m stop range, so the hold never completes while the robot stands still. Wide pose 4 (2 of 3): about 39° off square, it pushes against the gate as BC does. Wide pose 9 (3 of 3): it reaches the zone only at 26–27 s. Nothing collided, and both methods end 14–19° off square and 8–11 cm sideways (medians), which the judge does not measure.
- **What it does not show:** anything about the real robot (sim to sim, same renderer as the training images, no motor latency, ideal lidar); out-of-distribution lighting; actuation latency; or a fair learned-vs-classical comparison (BC's 3–67 % follows from demonstrations without stop examples and from the 0.02 m/s threshold). The 30 trials per cell are 10 poses × 3 correlated rounds, so the Wilson CIs, which assume independence, are too narrow; the per-pose tables are in each `report.md`.

## Other details
- **Reproducibility.** The draws are seeded per condition but also depend on timing, so a re-run gives statistically similar, not identical, trials.
- **Timing.** Policy inference p50 20–23 ms, p99 57–64 ms. The gate missed 0.05–0.07 % of its deadlines, with a worst single period of 318–334 ms per session. Two short low-priority host bag reads ran during the `wide` and `hfov62` sessions.
- **Provenance.** The trials ran on commit 5a83553 plus uncommitted changes (53–55 changed paths). The image digest, ONNX Runtime version and model sha256 are in each condition's `provenance.txt`.
- **One fix after the first condition.** The servo's `ros2 run` is now stopped through its process group; after `noise` it was stopped by hand. This does not affect any trial.

## Files and how to reproduce
`PYTHON=<venv>/bin/python bash scripts/sim_eval.sh` runs all three conditions (~35 min each); add `--report-only` to re-analyse. Each condition folder holds `trials.csv`, `schedule.csv`, `report.md`, `success.png`, `episodes_gt.csv` and `provenance.txt`. `clean_ref/` holds the clean trials copied from `results/sim_e2e/jazzy` with ground truth from the clean bags. The bags stay in `sim_work/eval_v2/jazzy/<condition>/bags`.
