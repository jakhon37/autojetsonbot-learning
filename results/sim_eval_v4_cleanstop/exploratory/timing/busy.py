#!/usr/bin/env python3
"""Calibrated CPU load: --procs processes, each busy for --duty of every 10 ms slice.

    python3 busy.py --procs 2 --duty 0.8 --seconds 600

2 x 0.8 = about 1.6 cores. Each process prints the CPU share it actually got when it stops
(SIGTERM or --seconds), so the load is measured, not assumed.
"""

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time


def worker(duty, seconds, slice_s=0.010):
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    t_start, c_start = time.perf_counter(), time.process_time()
    end = t_start + seconds
    x = 0
    while not stop[0] and time.perf_counter() < end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duty * slice_s:
            x += 1
        rest = slice_s - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
    wall, cpu = time.perf_counter() - t_start, time.process_time() - c_start
    print("busy pid %d: %.1f s wall, %.1f s CPU, %.2f cores" % (os.getpid(), wall, cpu, cpu / wall), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--duty", type=float, default=0.8)
    ap.add_argument("--seconds", type=float, default=600.0)
    args = ap.parse_args()
    procs = [mp.Process(target=worker, args=(args.duty, args.seconds)) for _ in range(args.procs)]
    for p in procs:
        p.start()
    signal.signal(signal.SIGTERM, lambda *_: [p.terminate() for p in procs])
    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
