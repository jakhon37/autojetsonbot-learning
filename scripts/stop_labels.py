#!/usr/bin/env python3
"""Stop examples in demonstrations, datasets and BC models (results/sim_eval_v3_bcfix).

    python scripts/stop_labels.py bags sim_work/v2/bags/jazzy/demos_r1 sim_work/v2/bags/jazzy/demos_r2 \
        --out results/sim_eval_v3_bcfix/demos_v2_endings.csv
    python scripts/stop_labels.py dataset --root sim_work/v2/dataset --split sim_work/v2/runs/bc/split.json
    python scripts/stop_labels.py offline --root sim_work/v2/dataset --split sim_work/v2/runs/bc/split.json \
        --model v1=sim_work/runs/bc/policy.onnx --model v2=sim_work/v2/runs/bc/policy.onnx

A frame or command is a *stop* when its requested |v| is below 0.005 m/s (the threshold
scripts/sim_eval_summary.py uses for "stop commanded"; from v4 on dock_judge calls a request
still at |v| <= 0.005 m/s, request_still_v; v2/v3 called up to 0.02 m/s still). Frames are
classed by position in their episode: stop_end and moving_end are
the stop and moving frames of the last 2 s (the docked view: the judge's 1 s hold and the
approach just before it), stop_start is the run of stop frames that begins the episode,
stop_middle any other stop frame and moving any other moving frame. A stop frame is a *turn*
when its requested |w| is at least 0.05 rad/s (the servo turning in place to find or centre
the marker) and a *full stop* otherwise.

bags     per demonstration in the bags: outcome, duration, the largest requested |v| in the
         last 1.00 s before the end (max_v_last1s, written unrounded, so "exactly 0" can be read
         from the CSV: P1 of results/sim_eval_v4_cleanstop/PREREG.md; a window without any
         request gives nan), how long the request had been a stop (stop_for_s) and
         exactly v = 0 (zero_for_s) when it ended, restarts_in_band: how often the request
         went from a stop back to moving while the judge said in_band (stop toggling near
         the box, as lidar noise caused around the v2/v3 servo's 0.27 m stop; the first 0.3 s
         after the start are skipped), and the judge's last range (final_range_m).
dataset  stop-label counts of the servo_demo successes (the frames train_bc.py trains on with
         --methods servo_demo --outcomes success), overall and per split.
offline  each ONNX model's predicted v on the validation frames, by frame class.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "jbl_nodes"))
sys.path.insert(0, str(REPO / "scripts"))

STOP_V = 0.005  # m/s
STILL_V = 0.02  # m/s, what the v2/v3 judge called a still request (dock_judge stopped_v then)
END_FRAMES = 20  # the last 2 s at the dataset's 10 fps
TURN_W = 0.05  # rad/s, dock_judge stopped_w
CLASSES = ("stop_end", "moving_end", "stop_start", "stop_middle", "moving")


def frame_classes(v: Sequence[float], threshold: float = STOP_V, end_frames: int = END_FRAMES) -> list[str]:
    """Per frame of one episode: stop_end, moving_end, stop_start, stop_middle or moving."""
    stop = np.abs(np.asarray(v, dtype=np.float64)) < threshold
    n = len(stop)
    lead = int(np.argmin(stop)) if n and not stop.all() else n  # frames before the first moving one
    out = []
    for i in range(n):
        if i >= n - end_frames:
            out.append("stop_end" if stop[i] else "moving_end")
        elif i < lead:
            out.append("stop_start")
        else:
            out.append("stop_middle" if stop[i] else "moving")
    return out


def ending(cmd: Sequence[tuple[float, ...]], t1: float, threshold: float = STOP_V) -> dict[str, float]:
    """Requests (t, v[, w]) in the last 1 s before t1 (largest and smallest |v|, share of stops) and
    how long the request had been a stop (|v| < threshold) and exactly v = 0 when the episode ended."""
    last1 = [abs(v) for t, v, *_ in cmd if t >= t1 - 1.0]
    since = zero_since = None
    for t, v, *_ in cmd:
        if abs(v) >= threshold:
            since = None
        elif since is None:
            since = t
        if v != 0.0:
            zero_since = None
        elif zero_since is None:
            zero_since = t
    return {
        "max_v_last1s": max(last1) if last1 else float("nan"),
        "min_v_last1s": min(last1) if last1 else float("nan"),
        "stop_share_last1s": sum(v < threshold for v in last1) / len(last1) if last1 else float("nan"),
        "stop_for_s": (t1 - since) if since is not None else 0.0,
        "zero_for_s": (t1 - zero_since) if zero_since is not None else 0.0,
    }


def restarts_in_band(
    cmd: Sequence[tuple[float, ...]],
    status: Sequence[tuple[float, dict[str, Any]]],
    t_from: float = -np.inf,
    threshold: float = STOP_V,
) -> int:
    """How often, from t_from on, the request went from a stop (|v| < threshold) back to moving
    while the judge's latest status said in_band: 0 for a controller that, once stopped near the
    box, stays stopped. t_from skips the start, where the judge may still report the last pose."""
    n, i, in_band, was_stop = 0, 0, False, False
    for t, v, *_ in cmd:
        while i < len(status) and status[i][0] <= t:
            in_band = bool(status[i][1].get("in_band"))
            i += 1
        stop = abs(v) < threshold
        if was_stop and not stop and in_band and t >= t_from:
            n += 1
        was_stop = stop
    return n


# ---------------------------------------------------------------------------- bags


def cmd_bags(args: argparse.Namespace) -> int:
    import sim_eval_summary as ses

    rows, zero_for_s = [], []  # zero_for_s unrounded, so the >= 1 s count is exact
    for path in args.bags:
        for ep in ses.read_episodes(path):
            t1 = ep.t1 if ep.t1 is not None else ep.t0
            e = ending(ep.cmd, t1)
            zero_for_s.append(e["zero_for_s"])
            # the judge's last range; null (no valid reading) is written as empty, like no status
            final_range = ep.status[-1][1].get("front_range_m") if ep.status else None
            rows.append({
                "bag": path.name, "method": ep.method, "pose_id": ep.pose_id, "round": ep.round,
                "outcome": ep.outcome, "duration_s": "%.3f" % (t1 - ep.t0),
                # repr round-trips the float exactly: P1 reads "exactly 0" from this column
                "max_v_last1s": repr(e["max_v_last1s"]), "min_v_last1s": "%.4f" % e["min_v_last1s"],
                "stop_share_last1s": "%.2f" % e["stop_share_last1s"], "stop_for_s": "%.2f" % e["stop_for_s"],
                "zero_for_s": "%.3f" % e["zero_for_s"],
                "restarts_in_band": restarts_in_band(ep.cmd, ep.status, ep.t0 + ses.TELEPORT_SETTLE_S),
                "final_range_m": "" if final_range is None else final_range,
            })
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["bag"])
            writer.writeheader()
            writer.writerows(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    zero = [r for r in ok if float(r["max_v_last1s"]) < STOP_V]
    creep = [r for r in ok if STOP_V <= float(r["max_v_last1s"]) <= STILL_V]
    print("%d episodes, %d successes; outcomes %s" % (
        len(rows), len(ok), dict(sorted(Counter(r["outcome"] for r in rows).items()))))
    print("successes whose request in the last 1 s was a stop (|v| < %.3f): %d/%d" % (STOP_V, len(zero), len(ok)))
    some = [r for r in ok if float(r["min_v_last1s"]) < STOP_V]
    print("successes with at least one stop request in the last 1 s: %d/%d; share of stop requests in the "
          "last 1 s, median %.2f" % (len(some), len(ok), statistics.median(float(r["stop_share_last1s"]) for r in ok)
                                     if ok else float("nan")))
    print("successes that ended on creep the v2/v3 judge called still (%.3f <= |v| <= %.2f): %d/%d %s" % (
        STOP_V, STILL_V, len(creep), len(ok), ", ".join("%s/%s:%.4f" % (r["pose_id"], r["round"],
                                                                        float(r["max_v_last1s"])) for r in creep)))
    print("successes whose largest requested |v| in the last 1.00 s was exactly 0 (max_v_last1s, unrounded): "
          "%d/%d" % (sum(1 for r in ok if float(r["max_v_last1s"]) == 0.0), len(ok)))
    if ok:
        stops = [float(r["stop_for_s"]) for r in ok]
        print("stop request held at the end: median %.2f s, min %.2f s, max %.2f s" % (
            statistics.median(stops), min(stops), max(stops)))
        zeros = [z for z, r in zip(zero_for_s, rows) if r["outcome"] == "success"]
        print("successes that ended with a request of exactly v = 0 for >= 1 s: %d/%d (held %.3f-%.3f s)" % (
            sum(z >= 1.0 for z in zeros), len(ok), min(zeros), max(zeros)))
    restarts = [r for r in rows if r["restarts_in_band"]]
    print("episodes whose request went from a stop back to moving while in band: %d/%d %s" % (
        len(restarts), len(rows), ", ".join("%s/%s:%d" % (r["pose_id"], r["round"], r["restarts_in_band"])
                                            for r in restarts)))
    durations = [float(r["duration_s"]) for r in ok]
    ranges = [float(r["final_range_m"]) for r in ok if r["final_range_m"] != ""]
    if ok:
        print("successes: duration median %.1f s (%.1f-%.1f); final judge range %s" % (
            statistics.median(durations), min(durations), max(durations),
            "%.3f-%.3f m (median %.3f)" % (min(ranges), max(ranges), statistics.median(ranges)) if ranges else "n/a"))
    return 0


# ---------------------------------------------------------------------------- dataset


def load_frames(root: Path, methods: Sequence[str], outcomes: Sequence[str]) -> tuple[Any, list[dict[str, Any]]]:
    """(DataFrame of episode_index, frame_index, v, w, cls; meta records) for the kept episodes."""
    import pandas as pd

    import jbl_episodes

    info = json.loads((root / "meta" / "info.json").read_text())
    records = jbl_episodes.load_episode_table(root, total_episodes=info["total_episodes"])
    records = jbl_episodes.filter_episodes(records, outcomes=list(outcomes), methods=list(methods))
    keep = {r["episode_index"] for r in records}
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files])
    df = df[df["episode_index"].isin(keep)].sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    df["v"], df["w"] = action[:, 0], action[:, 1]
    df = df.drop(columns=["action"])
    df["cls"] = ""
    for _, idx in df.groupby("episode_index").groups.items():
        df.loc[idx, "cls"] = frame_classes(df.loc[idx, "v"].to_numpy())
    return df, records


def label_stats(df: Any) -> dict[str, Any]:
    stop = df["cls"].str.startswith("stop")
    turn = df["w"].abs() >= TURN_W
    n = int(len(df))
    out: dict[str, Any] = {
        "episodes": int(df["episode_index"].nunique()),
        "frames": n,
        "stop_frames": int(stop.sum()),
        "stop_fraction": round(float(stop.mean()), 4) if n else None,
        "full_stop_frames": int((stop & ~turn).sum()),
        "full_stop_fraction": round(float((stop & ~turn).mean()), 4) if n else None,
    }
    for cls in CLASSES:
        here = df["cls"] == cls
        out[cls] = int(here.sum()) if cls.startswith("moving") else {
            "frames": int(here.sum()), "full_stops": int((here & ~turn).sum()), "turns": int((here & turn).sum())}
    tail = df[df["cls"].isin(["stop_end", "moving_end"])]
    out["end_label_v"] = _describe(tail["v"].abs().to_numpy())
    out["episodes_with_stop_in_last_2s"] = int(tail[tail["cls"] == "stop_end"]["episode_index"].nunique())
    out["final_stop_run_frames_per_episode"] = _describe([
        int(np.argmin(np.abs(g["v"].to_numpy()[::-1]) < STOP_V)) if (np.abs(g["v"].to_numpy()) >= STOP_V).any()
        else len(g) for _, g in df.groupby("episode_index")])
    return out


def _describe(values: Any) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {}
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def _split(path: Path | None) -> dict[str, list[int]]:
    return json.loads(path.read_text()) if path else {}


def cmd_dataset(args: argparse.Namespace) -> int:
    df, records = load_frames(args.root, args.methods, args.outcomes)
    out: dict[str, Any] = {"root": str(args.root), "methods": args.methods, "outcomes": args.outcomes,
                           "stop_v": STOP_V, "all": label_stats(df)}
    idle = [r.get("idle_start_s") for r in records]
    out["idle_start_trimmed"] = (
        None if all(x is None for x in idle)
        else {"episodes_trimmed": sum(1 for x in idle if x), "seconds_total": round(sum(x or 0 for x in idle), 3)}
    )
    for name, eps in _split(args.split).items():
        out[name] = label_stats(df[df["episode_index"].isin(eps)])
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


# ---------------------------------------------------------------------------- offline


def predict_v(model_path: Path, frames: Sequence[np.ndarray]) -> np.ndarray:
    """Predicted v (m/s, clipped as policy_runner does) of an ONNX policy on uint8 RGB frames."""
    import onnxruntime as ort

    from jbl_nodes import preprocess

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name  # the robot's contract: batch 1
    out = [session.run(None, {name: preprocess.to_model_input(f)[None]})[0][0, 0] for f in frames]
    return np.clip(np.asarray(out, dtype=np.float64), -1.0, 1.0) * preprocess.V_MAX


def cmd_offline(args: argparse.Namespace) -> int:
    import train_bc

    df, _ = load_frames(args.root, args.methods, args.outcomes)
    kept = set(int(e) for e in df["episode_index"].unique())
    val = [e for e in _split(args.split)["val"] if e in kept]
    df = df[df["episode_index"].isin(val)].reset_index(drop=True)
    frames_ds = train_bc.open_frames(args.repo_id, args.root, val)
    frames = []
    for i in range(len(frames_ds)):
        item = frames_ds[i]
        ep, fi = int(item["episode_index"]), int(item["frame_index"])
        assert (ep, fi) == (int(df.loc[i, "episode_index"]), int(df.loc[i, "frame_index"])), (ep, fi)
        frames.append(train_bc.frame_rgb_uint8(item[train_bc.IMAGE_KEY]))
    result: dict[str, Any] = {"root": str(args.root), "val_episodes": val, "frames": len(frames), "models": {}}
    for spec in args.model:
        label, _, path = spec.partition("=")
        pred = predict_v(Path(path), frames)
        df["pred_" + label] = pred
        per = {}
        turn = df["w"].abs() >= TURN_W
        masks = {cls: df["cls"] == cls for cls in CLASSES}
        masks["stop_end_full"] = masks["stop_end"] & ~turn  # docked view, labelled a full stop
        masks["stop_any"] = df["cls"].str.startswith("stop")
        masks["moving_any"] = df["cls"].str.startswith("moving")
        for cls, mask in masks.items():
            p, y = pred[mask.to_numpy()], df.loc[mask, "v"].to_numpy()
            per[cls] = {
                "frames": int(mask.sum()),
                "label_v_mean": round(float(y.mean()), 4) if len(y) else None,
                "pred_v_mean": round(float(p.mean()), 4) if len(p) else None,
                "pred_v_median": round(float(np.median(p)), 4) if len(p) else None,
                "pred_below_0.005": round(float((np.abs(p) < STOP_V).mean()), 3) if len(p) else None,
                "pred_below_0.02": round(float((np.abs(p) <= STILL_V).mean()), 3) if len(p) else None,
            }
        result["models"][label] = {"path": path, "by_label": per}
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if args.frames_csv:
        df.to_csv(args.frames_csv, index=False, float_format="%.4f")
    print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bags", help="how each demonstration in the bags ended")
    p.add_argument("bags", type=Path, nargs="+")
    p.add_argument("--out", type=Path, help="per-episode CSV")
    p.set_defaults(func=cmd_bags)
    for name, func, help_text in (("dataset", cmd_dataset, "stop-label counts"),
                                  ("offline", cmd_offline, "ONNX predictions on the validation frames")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--repo-id", default="local/jbl_sim")
        p.add_argument("--split", type=Path, required=name == "offline", help="train_bc split.json")
        p.add_argument("--methods", nargs="+", default=["servo_demo"])
        p.add_argument("--outcomes", nargs="+", default=["success"])
        p.add_argument("--out", type=Path, help="JSON")
        if name == "offline":
            p.add_argument("--model", action="append", required=True, metavar="LABEL=PATH.onnx")
            p.add_argument("--frames-csv", type=Path, help="per-frame labels and predictions")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
