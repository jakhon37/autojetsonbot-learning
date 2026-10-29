# Docking trials report

Setting: SIMULATION (fake_robot, ROS 2 foxy).

Source: `trials.csv` (20 valid rows). Baseline: `servo`. Intervals are 95%. A trial is scored when its outcome is success or fail; success also needs zero interventions. Aborted attempts are listed below and never counted.

## Success rate

| Method | Success k/n | Success | Wilson 95% CI |
|---|---:|---:|---:|
| servo | 10/10 | 100.0% | 72.2–100.0% |
| bc | 0/10 | 0.0% | 0.0–27.8% |

![Success rate per method with Wilson 95% CI](success.png)

## Paired comparison with `servo` (McNemar, pairs on pose_id × round)

b = servo succeeded and the method failed; c = the method succeeded and servo failed. p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). Δ = method − servo success rate over the pairs, with a Newcombe (1998, method 10) hybrid score CI for paired proportions.

| Method | Pairs | Both succeed | Both fail | b | c | Exact p | Mid-p | Δ success | Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bc | 10 | 0 | 0 | 10 | 0 | 0.00195 | 0.000977 | -100.0 pp | -100.0 to -60.8 pp |

Every slot is paired.

## Time to dock and final error

| Method | Time to dock, s (successes) | Final range error, cm | Final marker bearing, deg |
|---|---:|---:|---:|
| servo | 13.6 (12.5–18.2), n=10 | 2.0 (2.0–2.0), n=10 | 3.7 (3.4–3.9), n=10 |
| bc | — | 2.0 (1.3–3.0), n=10 | 1.9 (1.1–2.3), n=10 |

Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it includes the 1 s success hold. Range error = absolute difference between the final lidar range and the 0.25 m target; marker bearing = absolute final bearing of the marker from the image centre (not the robot's heading to the box face, which the judge does not measure); both over all scored trials with a dock_judge reading at the end.

## Outcomes

| Method | Success | Fail: timeout | Fail: stop_obstacle | Fail: intervention | Fail: other | Abort | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| servo | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| bc | 0 | 10 | 0 | 0 | 0 | 0 | 0 |

## Success by start pose

| Pose | servo | bc |
|---:|---:|---:|
| 0 | 1/1 | 0/1 |
| 1 | 1/1 | 0/1 |
| 2 | 1/1 | 0/1 |
| 3 | 1/1 | 0/1 |
| 4 | 1/1 | 0/1 |
| 5 | 1/1 | 0/1 |
| 6 | 1/1 | 0/1 |
| 7 | 1/1 | 0/1 |
| 8 | 1/1 | 0/1 |
| 9 | 1/1 | 0/1 |

## Schedule check

Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.

## Policy latency (/policy/latency)

| Source | Backend | n | Infer p50 | Infer p99 | Total p50 | Total p99 | Image age p50 | Image age p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sim-cpu-onnxruntime | onnxruntime | 3001 | 22.2 | 39.7 | 27.2 | 44.9 | 63.5 | 71.5 |

Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + preprocess + infer); image age = tick start minus the frame's capture stamp.

## Safety gate timing (/safety_gate/status)

| Source | Status msgs | Ticks | Missed deadlines | Period p50 | Period p99 | Worst p99 | Jitter p99 | Period max | Callback p99, µs | Restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| servo | 371 | 9244 | 9 (0.097%) | 20.00 | 20.45 | 23.00 | 3.00 | 61.74 | 211 | 0 |
| bc | 659 | 16448 | 7 (0.043%) | 20.00 | 23.54 | 23.61 | 3.61 | 54.01 | 383 | 0 |

Periods in ms; nominal 20.00 ms (rate_hz 50). A tick is a missed deadline when its period exceeds 1.5× nominal. Period p50/p99 are medians over the status windows (each covers the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst window. Ticks and missed deadlines count only what the log covers.

## Aborted attempts

None.

## Data issues

None.
