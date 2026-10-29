# Docking trials report

Setting: SIMULATION (fake_robot, ROS 2 jazzy), condition hfov62.

Source: `trials.csv` (60 valid rows). Baseline: `servo`. Intervals are 95%. A trial is scored when its outcome is success or fail; success also needs zero interventions. Aborted attempts are listed below and never counted.

## Success rate

| Method | Success k/n | Success | Wilson 95% CI |
|---|---:|---:|---:|
| servo | 28/30 | 93.3% | 78.7–98.2% |
| bc | 20/30 | 66.7% | 48.8–80.8% |

![Success rate per method with Wilson 95% CI](success.png)

## Paired comparison with `servo` (McNemar, pairs on pose_id × round)

b = servo succeeded and the method failed; c = the method succeeded and servo failed. p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). Δ = method − servo success rate over the pairs, with a Newcombe (1998, method 10) hybrid score CI for paired proportions.

| Method | Pairs | Both succeed | Both fail | b | c | Exact p | Mid-p | Δ success | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | 30 | 20 | 2 | 8 | 0 | 0.00781 | 0.00391 | -26.7 pp | -44.1 to -8.9 pp |

Every slot is paired.

## Time to dock and final error

| Method | Time to dock, s (successes) | Final range error, cm | Final marker bearing, deg |
|---|---:|---:|---:|
| servo | 15.1 (12.8–18.7), n=28 | 1.9 (1.8–2.0), n=30 | 3.6 (3.4–3.8), n=30 |
| bc | 25.0 (21.9–26.9), n=20 | 0.6 (0.3–1.2), n=30 | 1.0 (0.4–2.2), n=30 |

Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it includes the 1 s success hold. Range error = absolute difference between the final lidar range and the 0.25 m target; marker bearing = absolute final bearing of the marker from the image centre (not the robot's heading to the box face, which the judge does not measure); both over all scored trials with a dock_judge reading at the end.

## Outcomes

| Method | Success | Fail: timeout | Fail: stop_obstacle | Fail: intervention | Fail: other | Abort | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| servo | 28 | 2 | 0 | 0 | 0 | 0 | 0 |
| bc | 20 | 10 | 0 | 0 | 0 | 0 | 0 |

## Success by start pose

| Pose | servo | bc |
|---:|---:|---:|
| 0 | 3/3 | 3/3 |
| 1 | 3/3 | 2/3 |
| 2 | 3/3 | 2/3 |
| 3 | 3/3 | 3/3 |
| 4 | 3/3 | 3/3 |
| 5 | 1/3 | 0/3 |
| 6 | 3/3 | 2/3 |
| 7 | 3/3 | 0/3 |
| 8 | 3/3 | 3/3 |
| 9 | 3/3 | 2/3 |

## Schedule check

Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.

## Policy latency (/policy/latency)

| Source | Backend | n | Infer p50 | Infer p99 | Total p50 | Total p99 | Image age p50 | Image age p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim-cpu-onnxruntime | onnxruntime | 7936 | 23.1 | 63.5 | 29.7 | 73.0 | 57.2 | 100.8 |

Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + preprocess + infer); image age = tick start minus the frame's capture stamp.

## Safety gate timing (/safety_gate/status)

| Source | Status msgs | Ticks | Missed deadlines | Period p50 | Period p99 | Worst p99 | Jitter p99 | Period max | Callback p99, µs | Restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| session | 3084 | 76909 | 56 (0.073%) | 20.00 | 23.83 | 44.20 | 24.20 | 334.40 | 19926 | 0 |

Periods in ms; nominal 20.00 ms (rate_hz 50). A tick is a missed deadline when its period exceeds 1.5× nominal. Period p50/p99 are medians over the status windows (each covers the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst window. Ticks and missed deadlines count only what the log covers.

## Aborted attempts

None.

## Data issues

None.
