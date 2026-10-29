# Docking trials report

Setting: SIMULATION (fake_robot, ROS 2 jazzy), condition clean_ref.

Source: `trials.csv` (20 valid rows). Baseline: `servo`. Intervals are 95%. A trial is scored when its outcome is success or fail; success also needs zero interventions. Aborted attempts are listed below and never counted.

## Success rate

| Method | Success k/n | Success | Wilson 95% CI |
|---|---:|---:|---:|
| servo | 10/10 | 100.0% | 72.2–100.0% |
| bc_v4 | 9/10 | 90.0% | 59.6–98.2% |

![Success rate per method with Wilson 95% CI](success.png)

## Paired comparison with `servo` (McNemar, pairs on pose_id × round)

b = servo succeeded and the method failed; c = the method succeeded and servo failed. p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). Δ = method − servo success rate over the pairs, with a Newcombe (1998, method 10) hybrid score CI for paired proportions.

| Method | Pairs | Both succeed | Both fail | b | c | Exact p | Mid-p | Δ success | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc_v4 | 10 | 9 | 0 | 1 | 0 | 1 | 0.5 | -10.0 pp | -40.4 to +18.9 pp |

Every slot is paired.

## Time to dock and final error

| Method | Time to dock, s (successes) | Final range error, cm | Final marker bearing, deg |
|---|---:|---:|---:|
| servo | 14.7 (13.1–19.0), n=10 | 2.0 (2.0–2.0), n=10 | 3.7 (3.4–3.8), n=10 |
| bc_v4 | 18.7 (17.8–23.9), n=9 | 1.5 (1.4–1.6), n=10 | 3.1 (2.8–3.4), n=10 |

Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it includes the 1 s success hold. Range error = absolute difference between the final lidar range and the 0.25 m target; marker bearing = absolute final bearing of the marker from the image centre (not the robot's heading to the box face, which the judge does not measure); both over all scored trials with a dock_judge reading at the end.

## Outcomes

| Method | Success | Fail: timeout | Fail: stop_obstacle | Fail: intervention | Fail: other | Abort | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| servo | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| bc_v4 | 9 | 1 | 0 | 0 | 0 | 0 | 0 |

## Success by start pose

| Pose | servo | bc_v4 |
|---:|---:|---:|
| 0 | 1/1 | 1/1 |
| 1 | 1/1 | 1/1 |
| 2 | 1/1 | 1/1 |
| 3 | 1/1 | 1/1 |
| 4 | 1/1 | 1/1 |
| 5 | 1/1 | 0/1 |
| 6 | 1/1 | 1/1 |
| 7 | 1/1 | 1/1 |
| 8 | 1/1 | 1/1 |
| 9 | 1/1 | 1/1 |

## Schedule check

Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.

## Policy latency (/policy/latency)

| Source | Backend | n | Infer p50 | Infer p99 | Total p50 | Total p99 | Image age p50 | Image age p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim-cpu-onnxruntime | onnxruntime | 2058 | 69.4 | 122.6 | 80.9 | 139.4 | 56.3 | 188.0 |

Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + preprocess + infer); image age = tick start minus the frame's capture stamp.

## Safety gate timing (/safety_gate/status)

| Source | Status msgs | Ticks | Missed deadlines | Period p50 | Period p99 | Worst p99 | Jitter p99 | Period max | Callback p99, µs | Restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| session | 959 | 23863 | 25 (0.105%) | 20.00 | 24.54 | 25.32 | 5.32 | 351.74 | 3363 | 0 |

Periods in ms; nominal 20.00 ms (rate_hz 50). A tick is a missed deadline when its period exceeds 1.5× nominal. Period p50/p99 are medians over the status windows (each covers the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst window. Ticks and missed deadlines count only what the log covers.

## Aborted attempts

None.

## Data issues

None.
