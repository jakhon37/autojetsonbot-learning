# Docking trials report

Setting: SIMULATION (fake_robot, ROS 2 jazzy), condition wide.

Source: `trials.csv` (60 valid rows). Baseline: `servo`. Intervals are 95%. A trial is scored when its outcome is success or fail; success also needs zero interventions. Aborted attempts are listed below and never counted.

## Success rate

| Method | Success k/n | Success | Wilson 95% CI |
|---|---:|---:|---:|
| servo | 25/30 | 83.3% | 66.4–92.7% |
| bc | 1/30 | 3.3% | 0.6–16.7% |

![Success rate per method with Wilson 95% CI](success.png)

## Paired comparison with `servo` (McNemar, pairs on pose_id × round)

b = servo succeeded and the method failed; c = the method succeeded and servo failed. p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). Δ = method − servo success rate over the pairs, with a Newcombe (1998, method 10) hybrid score CI for paired proportions.

| Method | Pairs | Both succeed | Both fail | b | c | Exact p | Mid-p | Δ success | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | 30 | 1 | 5 | 24 | 0 | 1.19e-07 | 5.96e-08 | -80.0 pp | -89.7 to -58.5 pp |

Every slot is paired.

## Time to dock and final error

| Method | Time to dock, s (successes) | Final range error, cm | Final marker bearing, deg |
|---|---:|---:|---:|
| servo | 21.0 (17.8–22.0), n=25 | 1.9 (1.7–2.0), n=30 | 3.6 (3.4–3.8), n=30 |
| bc | 21.8 (21.8–21.8), n=1 | 1.5 (1.0–2.4), n=30 | 1.8 (0.5–2.8), n=30 |

Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it includes the 1 s success hold. Range error = absolute difference between the final lidar range and the 0.25 m target; marker bearing = absolute final bearing of the marker from the image centre (not the robot's heading to the box face, which the judge does not measure); both over all scored trials with a dock_judge reading at the end.

## Outcomes

| Method | Success | Fail: timeout | Fail: stop_obstacle | Fail: intervention | Fail: other | Abort | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| servo | 25 | 5 | 0 | 0 | 0 | 0 | 0 |
| bc | 1 | 29 | 0 | 0 | 0 | 0 | 0 |

## Success by start pose

| Pose | servo | bc |
|---:|---:|---:|
| 0 | 3/3 | 1/3 |
| 1 | 3/3 | 0/3 |
| 2 | 3/3 | 0/3 |
| 3 | 3/3 | 0/3 |
| 4 | 1/3 | 0/3 |
| 5 | 3/3 | 0/3 |
| 6 | 3/3 | 0/3 |
| 7 | 3/3 | 0/3 |
| 8 | 3/3 | 0/3 |
| 9 | 0/3 | 0/3 |

## Schedule check

Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.

## Policy latency (/policy/latency)

| Source | Backend | n | Infer p50 | Infer p99 | Total p50 | Total p99 | Image age p50 | Image age p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim-cpu-onnxruntime | onnxruntime | 8931 | 20.1 | 60.7 | 26.6 | 68.9 | 63.2 | 89.6 |

Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + preprocess + infer); image age = tick start minus the frame's capture stamp.

## Safety gate timing (/safety_gate/status)

| Source | Status msgs | Ticks | Missed deadlines | Period p50 | Period p99 | Worst p99 | Jitter p99 | Period max | Callback p99, µs | Restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| session | 3533 | 88249 | 41 (0.046%) | 20.00 | 23.03 | 23.48 | 3.48 | 318.95 | 2760 | 0 |

Periods in ms; nominal 20.00 ms (rate_hz 50). A tick is a missed deadline when its period exceeds 1.5× nominal. Period p50/p99 are medians over the status windows (each covers the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst window. Ticks and missed deadlines count only what the log covers.

## Aborted attempts

None.

## Data issues

None.
