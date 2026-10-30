#!/usr/bin/env python3
"""CPU use on the host over a short window: all cores, and the given Docker containers.

    python3 hostload.py SECONDS [CONTAINER_ID ...]

Prints one JSON line: host busy cores (root cgroup v1 cpuacct.usage), each container's cores
(its cpuacct.usage), "other" = host busy minus the containers, and /proc/loadavg.
"""

import json
import sys
import time


def host_busy():
    """CPU seconds of every task on the host (root cgroup v1 cpuacct, scheduler-exact)."""
    with open("/sys/fs/cgroup/cpuacct/cpuacct.usage") as f:
        return int(f.read()) / 1e9


def container_ns(cid):
    for path in ("/sys/fs/cgroup/cpuacct/docker/%s/cpuacct.usage" % cid,
                 "/sys/fs/cgroup/cpu,cpuacct/docker/%s/cpuacct.usage" % cid):
        try:
            with open(path) as f:
                return int(f.read())
        except OSError:
            continue
    return None


def main():
    seconds = float(sys.argv[1])
    ids = sys.argv[2:]
    h0, c0, t0 = host_busy(), {c: container_ns(c) for c in ids}, time.time()
    time.sleep(seconds)
    h1, c1, t1 = host_busy(), {c: container_ns(c) for c in ids}, time.time()
    wall = t1 - t0
    host = (h1 - h0) / wall
    cont = {c[:12]: (None if c0[c] is None or c1[c] is None else round((c1[c] - c0[c]) / 1e9 / wall, 3)) for c in ids}
    other = host - sum(v for v in cont.values() if v is not None)
    print(json.dumps({"window_s": round(wall, 1), "host_busy_cores": round(host, 3), "containers": cont,
                      "other_cores": round(other, 3), "loadavg": open("/proc/loadavg").read().strip()}))


if __name__ == "__main__":
    main()
