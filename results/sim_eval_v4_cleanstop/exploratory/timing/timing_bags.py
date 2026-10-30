"""Split policy_runner's recorded latency into preprocess / infer / image age, per session bag.

    python timing_bags.py REPO [--csv out.csv]

Reads the v3 (sim_work/v2, sim_work/v2sw) and v4 (sim_work/v4, sim_work/v4_62) session bags.
Per bag: the /policy/latency fields (preprocess_ms = JPEG decode + preprocess, infer_ms =
backend.infer, total_ms = tick start to publish, image_age_ms at tick start), as n, p50, p90,
p99 (numpy linear percentiles, as tools/analyze_trials.py), the remainder
total - preprocess - infer, the intervals between consecutive /policy/latency messages inside
one policy episode (from the bag's integer-nanosecond receive times), and the policy episodes'
/cmd_vel_in requests with the share that are exact (0, 0) fallback zeros.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SESSIONS = (  # (run, session label, work folder, method of the policy arm)
    ("v3", "bc_v2", "sim_work/v2", "bc_v2"),
    ("v3", "bc_v2sw", "sim_work/v2sw", "bc_v2sw"),
    ("v4", "1: servo + bc_v4", "sim_work/v4", "bc_v4"),
    ("v4", "2: bc_v4_62", "sim_work/v4_62", "bc_v4_62"),
)
CONDITIONS = ("clean_ref", "noise", "wide", "hfov62")
FIELDS = ("preprocess_ms", "infer_ms", "total_ms", "image_age_ms")
TOPICS = ("/policy/latency", "/episode/event", "/cmd_vel_in")


def pct(values, qs=(50, 90, 99)):
    if not len(values):
        return [float("nan")] * len(qs)
    return [float(v) for v in np.percentile(np.asarray(values, dtype=np.float64), qs)]


def read_bag(bag: Path, policy_method: str) -> dict:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    lat = {k: [] for k in FIELDS}
    lat_t = []
    intervals = []
    ticks = zeros = 0
    in_policy = False
    last_ns = None  # integer ns: float seconds would round a 149.999985 ms gap to 150.0003 ms
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = t_ns * 1e-9
            if conn.topic == "/episode/event":
                words = msg.data.split()
                if words and words[0] == "start" and len(words) == 4:
                    in_policy = words[1] == policy_method
                    last_ns = None
                elif words and words[0] in ("end", "intervention"):
                    in_policy = False
                    last_ns = None
                continue
            if conn.topic == "/policy/latency":
                d = json.loads(msg.data)
                for k in FIELDS:
                    lat[k].append(float(d[k]))
                lat_t.append(t)
                if in_policy and last_ns is not None:
                    intervals.append((t_ns - last_ns) / 1e6)
                last_ns = t_ns if in_policy else None
                continue
            if conn.topic == "/cmd_vel_in" and in_policy:
                ticks += 1
                if msg.linear.x == 0.0 and msg.angular.z == 0.0:
                    zeros += 1
    rest = np.asarray(lat["total_ms"]) - np.asarray(lat["preprocess_ms"]) - np.asarray(lat["infer_ms"])
    out = {"n": len(lat["infer_ms"]), "ticks": ticks, "zeros": zeros}
    for k in FIELDS:
        out[k] = pct(lat[k])
        out[k + "_mean"] = float(np.mean(lat[k])) if lat[k] else float("nan")
    out["rest_ms"] = pct(rest)
    out["interval_ms"] = pct(intervals)
    out["interval_gt_150"] = int(np.sum(np.asarray(intervals) > 150.0)) if intervals else 0
    out["n_intervals"] = len(intervals)
    out["infer_gt_100"] = int(np.sum(np.asarray(lat["infer_ms"]) > 100.0))
    out["total_gt_100"] = int(np.sum(np.asarray(lat["total_ms"]) > 100.0))
    out["t_first"] = min(lat_t) if lat_t else None
    out["t_last"] = max(lat_t) if lat_t else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    rows = []
    for run, label, work, method in SESSIONS:
        for cond in CONDITIONS:
            bags = sorted((args.repo / work / "eval_v2/jazzy" / cond / "bags").glob("seg_*"))
            bags = [b for b in bags if b.is_dir()]
            for bag in bags:
                r = read_bag(bag, method)
                r.update(run=run, session=label, method=method, condition=cond, bag=bag.name)
                rows.append(r)
                print(
                    "%-3s %-17s %-9s %-21s n=%4d  pre %5.1f/%5.1f/%5.1f  inf %5.1f/%5.1f/%5.1f  "
                    "tot %5.1f/%5.1f/%5.1f  age %5.1f/%5.1f/%5.1f  rest p50 %.2f p99 %.2f  "
                    "int %5.1f/%5.1f/%5.1f (>150: %d of %d)  zeros %d/%d (%.2f %%)"
                    % (
                        run, label, cond, bag.name, r["n"],
                        *r["preprocess_ms"], *r["infer_ms"], *r["total_ms"], *r["image_age_ms"],
                        r["rest_ms"][0], r["rest_ms"][2], *r["interval_ms"],
                        r["interval_gt_150"], r["n_intervals"], r["zeros"], r["ticks"],
                        100.0 * r["zeros"] / max(r["ticks"], 1),
                    ),
                    flush=True,
                )
    print()
    print("| Run | Session (policy arm) | Condition | Inferences | Preprocess p50 / p99, ms "
          "| Infer p50 / p90 / p99, ms | Total p50 / p99, ms | Image age p50 / p99, ms "
          "| Consecutive inferences > 150 ms apart | Fallback zeros / policy ticks |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            "| %s | %s | %s | %d | %.1f / %.1f | %.1f / %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %d of %d "
            "| %d / %d (%.2f %%) |"
            % (
                r["run"], r["session"], r["condition"], r["n"], r["preprocess_ms"][0], r["preprocess_ms"][2],
                *r["infer_ms"], r["total_ms"][0], r["total_ms"][2], r["image_age_ms"][0], r["image_age_ms"][2],
                r["interval_gt_150"], r["n_intervals"], r["zeros"], r["ticks"], 100.0 * r["zeros"] / max(r["ticks"], 1),
            )
        )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            head = ["run", "session", "method", "condition", "bag", "n"]
            for k in FIELDS + ("rest_ms", "interval_ms"):
                head += [k + "_p50", k + "_p90", k + "_p99"]
            head += ["preprocess_ms_mean", "infer_ms_mean", "infer_gt_100", "total_gt_100",
                     "interval_gt_150", "n_intervals", "policy_ticks", "fallback_zeros"]
            w.writerow(head)
            for r in rows:
                line = [r["run"], r["session"], r["method"], r["condition"], r["bag"], r["n"]]
                for k in FIELDS + ("rest_ms", "interval_ms"):
                    line += ["%.3f" % v for v in r[k]]
                line += ["%.3f" % r["preprocess_ms_mean"], "%.3f" % r["infer_ms_mean"], r["infer_gt_100"],
                         r["total_gt_100"], r["interval_gt_150"], r["n_intervals"], r["ticks"], r["zeros"]]
                w.writerow(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
