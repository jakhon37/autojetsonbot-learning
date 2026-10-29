# Docking trials report

Setting: SIMULATION (fake_robot, ROS 2 jazzy), condition wide.

Source: `trials.csv` (120 valid rows). Baseline: `servo`. Intervals are 95%. A trial is scored when its outcome is success or fail; success also needs zero interventions. Aborted attempts are listed below and never counted.

## Success rate

| Method | Success k/n | Success | Wilson 95% CI |
|---|---:|---:|---:|
| servo | 25/30 | 83.3% | 66.4–92.7% |
| bc | 1/30 | 3.3% | 0.6–16.7% |
| bc_v2 | 25/30 | 83.3% | 66.4–92.7% |
| bc_v2sw | 22/30 | 73.3% | 55.6–85.8% |

![Success rate per method with Wilson 95% CI](success.png)

## Paired comparison with `servo` (McNemar, pairs on pose_id × round)

b = servo succeeded and the method failed; c = the method succeeded and servo failed. p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). Δ = method − servo success rate over the pairs, with a Newcombe (1998, method 10) hybrid score CI for paired proportions.

| Method | Pairs | Both succeed | Both fail | b | c | Exact p | Mid-p | Δ success | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | 30 | 1 | 5 | 24 | 0 | 1.19e-07 | 5.96e-08 | -80.0 pp | -89.7 to -58.5 pp |
| bc_v2 | 30 | 22 | 2 | 3 | 3 | 1 | 1 | +0.0 pp | -17.9 to +17.9 pp |
| bc_v2sw | 30 | 20 | 3 | 5 | 2 | 0.453 | 0.289 | -10.0 pp | -28.0 to +8.5 pp |

Every slot is paired.

## Time to dock and final error

| Method | Time to dock, s (successes) | Final range error, cm | Final marker bearing, deg |
|---|---:|---:|---:|
| servo | 21.0 (17.8–22.0), n=25 | 1.9 (1.7–2.0), n=30 | 3.6 (3.4–3.8), n=30 |
| bc | 21.8 (21.8–21.8), n=1 | 1.5 (1.0–2.4), n=30 | 1.8 (0.5–2.8), n=30 |
| bc_v2 | 20.2 (18.8–21.7), n=25 | 2.2 (2.0–2.6), n=30 | 2.7 (1.2–5.0), n=30 |
| bc_v2sw | 19.2 (16.2–22.1), n=22 | 2.7 (2.3–3.5), n=30 | 3.1 (1.8–4.4), n=30 |

Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it includes the 1 s success hold. Range error = absolute difference between the final lidar range and the 0.25 m target; marker bearing = absolute final bearing of the marker from the image centre (not the robot's heading to the box face, which the judge does not measure); both over all scored trials with a dock_judge reading at the end.

## Outcomes

| Method | Success | Fail: timeout | Fail: stop_obstacle | Fail: intervention | Fail: other | Abort | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| servo | 25 | 5 | 0 | 0 | 0 | 0 | 0 |
| bc | 1 | 29 | 0 | 0 | 0 | 0 | 0 |
| bc_v2 | 25 | 5 | 0 | 0 | 0 | 0 | 0 |
| bc_v2sw | 22 | 8 | 0 | 0 | 0 | 0 | 0 |

## Success by start pose

| Pose | servo | bc | bc_v2 | bc_v2sw |
|---:|---:|---:|---:|---:|
| 0 | 3/3 | 1/3 | 3/3 | 3/3 |
| 1 | 3/3 | 0/3 | 3/3 | 1/3 |
| 2 | 3/3 | 0/3 | 3/3 | 2/3 |
| 3 | 3/3 | 0/3 | 0/3 | 1/3 |
| 4 | 1/3 | 0/3 | 3/3 | 3/3 |
| 5 | 3/3 | 0/3 | 3/3 | 3/3 |
| 6 | 3/3 | 0/3 | 3/3 | 3/3 |
| 7 | 3/3 | 0/3 | 3/3 | 3/3 |
| 8 | 3/3 | 0/3 | 3/3 | 3/3 |
| 9 | 0/3 | 0/3 | 1/3 | 0/3 |

## Schedule check

Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.

## Policy latency (/policy/latency)

| Source | Backend | n | Infer p50 | Infer p99 | Total p50 | Total p99 | Image age p50 | Image age p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim-cpu-onnxruntime | onnxruntime | 6522 | 29.7 | 61.9 | 37.7 | 73.4 | 50.9 | 94.0 |
| earlier2 | onnxruntime | 6387 | 26.0 | 66.8 | 33.4 | 74.8 | 40.3 | 100.3 |

Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + preprocess + infer); image age = tick start minus the frame's capture stamp.

## Safety gate timing (/safety_gate/status)

| Source | Status msgs | Ticks | Missed deadlines | Period p50 | Period p99 | Worst p99 | Jitter p99 | Period max | Callback p99, µs | Restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| session | 1553 | 38731 | 61 (0.157%) | 20.00 | 23.66 | 34.07 | 14.07 | 349.18 | 3666 | 0 |
| earlier2 | 1544 | 38516 | 35 (0.091%) | 20.01 | 23.80 | 24.09 | 4.09 | 353.10 | 3997 | 0 |

Periods in ms; nominal 20.00 ms (rate_hz 50). A tick is a missed deadline when its period exceeds 1.5× nominal. Period p50/p99 are medians over the status windows (each covers the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst window. Ticks and missed deadlines count only what the log covers.

## Aborted attempts

None.

## Data issues

None.
