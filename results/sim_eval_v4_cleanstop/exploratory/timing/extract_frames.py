"""Recorded camera frames (CompressedImage JPEG bytes) from a session bag, for the timing harness.

    python extract_frames.py BAG METHOD N OUT.npz

Takes the first N /camera/image_raw/compressed messages recorded while an episode of METHOD
ran (between its "start" and "end" events), every frame in order. Writes the JPEG bytes
concatenated ("data"), their offsets ("offsets"), the format string and the bag times.
"""

import sys
from pathlib import Path

import numpy as np


def main(bag: str, method: str, n: int, out: str) -> int:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    blobs, times, fmt = [], [], None
    active = False
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_FOXY)) as reader:
        conns = [c for c in reader.connections if c.topic in ("/episode/event", "/camera/image_raw/compressed")]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/episode/event":
                words = msg.data.split()
                if words and words[0] == "start" and len(words) == 4:
                    active = words[1] == method
                elif words and words[0] in ("end", "intervention"):
                    active = False
                continue
            if active:
                blobs.append(bytes(msg.data))
                times.append(t_ns)
                fmt = msg.format
                if len(blobs) >= n:
                    break
    if len(blobs) < n:
        print("only %d frames found" % len(blobs))
        return 1
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    np.savez(out, data=np.frombuffer(b"".join(blobs), dtype=np.uint8), offsets=offsets,
             format=np.array(fmt), times_ns=np.asarray(times, dtype=np.int64))
    print("%d frames, %.1f kB mean, format %r -> %s" % (len(blobs), offsets[-1] / len(blobs) / 1e3, fmt, out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]))
