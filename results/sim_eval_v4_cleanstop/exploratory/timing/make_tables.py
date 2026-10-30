#!/usr/bin/env python3
"""Markdown tables from harness.py's results (one JSON line per run, run_matrix.sh's labels).

    python3 make_tables.py results.jsonl

Prints (1) a pivot of inference p50 / p99 by background, cpuset and thread count, and (2) every
run in full: preprocess, infer and total p50 / p90 / p99, ticks over the 100 ms period, the CPU
the policy process used, and the CPU affinity of the threads the session created.
"""

import json
import sys

BACKGROUND = {"idle": "none", "sim": "simulator nodes", "busy": "busy loop 1.6 cores", "rep": "simulator nodes (repeat)"}
THREADS = ("default", "1", "2", "3", "4")


def parse(label):
    """idle_free_t2, sim_pin_tdefault_bcv2, rep_sim_pin_tdefault, ... -> (background, cpuset, threads, extra)."""
    parts = label.split("_")
    if parts[0] == "rep":
        parts = ["rep"] + parts[2:]
    background, where, threads = parts[0], parts[1], parts[2][1:]
    return background, where, threads, "_".join(parts[3:])


def threads_label(d):
    if d["threads"] == "default":
        s = "default (%d)" % (d["session_threads_created"] + 1)
    else:
        s = d["threads"]
    return s + (", no spin" if d.get("spin") == "0" else "")


def main(path):
    runs = [json.loads(line) for line in open(path) if line.strip()]
    cells = {}
    for d in runs:
        bg, where, threads, extra = parse(d["label"])
        if extra or bg == "rep":
            continue
        cells[(bg, where, threads)] = d
    print("Inference p50 / p99, ms (400 ticks at 10 Hz each; bc_v4)")
    print()
    print("| Background | Container | " + " | ".join("threads %s" % t for t in THREADS) + " |")
    print("|---|---|" + "---:|" * len(THREADS))
    for bg in ("idle", "sim", "busy"):
        for where in ("free", "pin"):
            row = []
            for t in THREADS:
                d = cells.get((bg, where, t))
                row.append("%.1f / %.1f" % (d["infer_ms"]["p50"], d["infer_ms"]["p99"]) if d else "—")
            print("| %s | %s | %s |" % (BACKGROUND[bg], "no cpuset" if where == "free" else "cpuset 0-2", " | ".join(row)))
    print()
    print("| Run | Background | Container | Model | Intra-op threads | Session threads' affinity | Preprocess p50 / p90 / p99, ms "
          "| Infer p50 / p90 / p99, ms | Total p50 / p90 / p99, ms | Ticks > 100 ms (of 400) | Policy process, cores |")
    print("|---|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for d in runs:
        bg, where, threads, extra = parse(d["label"])
        model = "bc_v2 (v3 frames)" if "bc_v2" in d["model"] else "bc_v4"
        aff = " ".join("{%s}" % ",".join(str(c) for c in t["affinity"]) for t in d["session_threads"]) or "none"
        p, i, t = d["preprocess_ms"], d["infer_ms"], d["total_ms"]
        print("| %s | %s | %s | %s | %s | %s | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | %d | %.2f |" % (
            d["label"], BACKGROUND[bg], "no cpuset" if where == "free" else "cpuset 0-2", model, threads_label(d), aff,
            p["p50"], p["p90"], p["p99"], i["p50"], i["p90"], i["p99"], t["p50"], t["p90"], t["p99"],
            d["ticks_over_100ms"], d["cores_used"]))


if __name__ == "__main__":
    main(sys.argv[1])
