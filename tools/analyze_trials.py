#!/usr/bin/env python3
"""Analyse the paired real-robot docking trials (docs/INTERFACES.md, "Evaluation protocol").

    python tools/analyze_trials.py trials/trials.csv --out-dir reports/eval \
        --schedule trials/schedule.csv \
        --latency jetson=bags/trials_r1 --latency laptop=logs/latency_laptop.jsonl \
        --gate bags/trials_r1

Inputs
    trials.csv   written by trial_logger: method, pose_id, round, outcome, duration_s,
                 final_range_m, final_bearing_deg, interventions, notes
    --schedule   a make_schedule.py CSV (columns method, pose_id, round): lists scheduled
                 slots without a scored trial and scored trials that were never scheduled
    --latency    /policy/latency records      } each [LABEL=]PATH, repeatable. PATH is a rosbag2
    --gate       /safety_gate/status records  } recording (directory with metadata.yaml, a .db3
                 or .mcap file, or a directory holding several) or a JSONL file: one JSON object
                 per line, either the payload itself or {"data": <payload or its JSON text>, ...};
                 blank and "---" lines are skipped. A live JSONL log on Foxy:
                     ros2 topic echo --csv --full-length /policy/latency std_msgs/msg/String > lat.jsonl
                 LABEL (default: the file or bag name) names the source, e.g. jetson / laptop for
                 on-board vs off-board inference.

Scoring rules
    - success and fail rows are scored; abort marks a void attempt (setup error, someone walked
      into the arena, recorder crash) that is re-run in the same slot. Aborts are listed in the
      report and never counted in k/n.
    - Each slot (method, pose_id, round) is scored once, from its first success/fail row in file
      order. Later rows for the slot are reported as duplicates and ignored, so re-running a
      finished trial can never change its result.
    - Success needs outcome success AND zero interventions (an intervention is a fail by the
      success rule); a row claiming both is scored as a fail and reported.
    - Methods are compared with the baseline on the slots where both have a scored trial (pairs
      on pose_id x round): exact and mid-p McNemar on the discordant pairs, and a paired
      bootstrap CI for the difference in success rate. Unpaired slots are listed.

Outputs in --out-dir (default: the directory of trials.csv): report.md and success.png.
Exit status: 0 report written, 2 unreadable or malformed input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

TRIAL_COLUMNS = (
    "method",
    "pose_id",
    "round",
    "outcome",
    "duration_s",
    "final_range_m",
    "final_bearing_deg",
    "interventions",
    "notes",
)
OUTCOMES = ("success", "fail", "abort")
SCORED_OUTCOMES = ("success", "fail")
KNOWN_METHOD_ORDER = ("servo", "bc", "act")
FAIL_REASONS = ("timeout", "stop_obstacle", "intervention")  # dock_judge's "end fail <reason>"
CATEGORIES = ("success", "timeout", "stop_obstacle", "intervention", "other", "abort")
CONFIDENCE = 0.95
TARGET_RANGE_M = 0.25
GATE_RATE_HZ = 50.0
MISSED_DEADLINE_FACTOR = 1.5  # safety_gate_node counts a tick period above 1.5x nominal as missed
LATENCY_TOPIC = "/policy/latency"
GATE_TOPIC = "/safety_gate/status"
LATENCY_FIELDS = ("infer_ms", "total_ms", "image_age_ms")
P99_MIN_SAMPLES = 100
BAG_SUFFIXES = (".db3", ".mcap")

# Chart style: categorical slots of the reference palette, in method order.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

Slot = tuple[int, int]  # (pose_id, round)


class InputError(ValueError):
    """Input that cannot be analysed at all (missing file or columns)."""


# --------------------------------------------------------------------------------------------
# Trials


@dataclass(frozen=True)
class Trial:
    line: int  # line in trials.csv, for the issue list
    method: str
    pose_id: int
    round: int
    outcome: str
    duration_s: float | None
    final_range_m: float | None
    final_bearing_deg: float | None
    interventions: int
    notes: str

    @property
    def slot(self) -> Slot:
        return (self.pose_id, self.round)

    @property
    def scored(self) -> bool:
        return self.outcome in SCORED_OUTCOMES

    @property
    def success(self) -> bool:
        return self.outcome == "success" and self.interventions == 0

    @property
    def category(self) -> str:
        """One of CATEGORIES: success, the fail reason, or abort."""
        if self.outcome == "abort":
            return "abort"
        if self.success:
            return "success"
        if self.interventions > 0:
            return "intervention"
        words = self.notes.split(";")[0].split()
        return words[0] if words and words[0] in FAIL_REASONS else "other"


def _optional_float(text: str, name: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError("%s is not a number: %r" % (name, text)) from None
    return value if np.isfinite(value) else None


def _int(text: str, name: str, default: int | None = None) -> int:
    text = (text or "").strip()
    if not text and default is not None:
        return default
    try:
        value = int(text)
    except ValueError:
        raise ValueError("%s is not an integer: %r" % (name, text)) from None
    if value < 0:
        raise ValueError("%s is negative: %d" % (name, value))
    return value


def parse_trial_row(row: dict[str, str], line: int) -> Trial:
    method = (row.get("method") or "").strip()
    if not method:
        raise ValueError("empty method")
    outcome = (row.get("outcome") or "").strip().lower()
    if outcome not in OUTCOMES:
        raise ValueError("outcome must be one of %s, got %r" % ("/".join(OUTCOMES), row.get("outcome")))
    return Trial(
        line=line,
        method=method,
        pose_id=_int(row.get("pose_id", ""), "pose_id"),
        round=_int(row.get("round", ""), "round"),
        outcome=outcome,
        duration_s=_optional_float(row.get("duration_s", ""), "duration_s"),
        final_range_m=_optional_float(row.get("final_range_m", ""), "final_range_m"),
        final_bearing_deg=_optional_float(row.get("final_bearing_deg", ""), "final_bearing_deg"),
        interventions=_int(row.get("interventions", ""), "interventions", default=0),
        notes=(row.get("notes") or "").strip(),
    )


def load_trials(path: Path) -> tuple[list[Trial], list[str]]:
    """Rows of trials.csv, plus one issue line per row that could not be parsed (and was skipped)."""
    trials: list[Trial] = []
    issues: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in TRIAL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise InputError("%s is missing columns: %s" % (path, ", ".join(missing)))
        for row in reader:
            line = reader.line_num
            if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
                continue
            try:
                trials.append(parse_trial_row(row, line))
            except ValueError as exc:
                issues.append("line %d skipped: %s" % (line, exc))
    return trials, issues


@dataclass
class Scoring:
    scored: dict[tuple[str, int, int], Trial]  # (method, pose_id, round) -> the scored trial
    aborts: list[Trial]
    issues: list[str]


def score_trials(trials: Iterable[Trial]) -> Scoring:
    scored: dict[tuple[str, int, int], Trial] = {}
    aborts: list[Trial] = []
    issues: list[str] = []
    for t in trials:
        if not t.scored:
            aborts.append(t)
            continue
        key = (t.method, t.pose_id, t.round)
        if key in scored:
            issues.append(
                "line %d: duplicate %s pose %d round %d (%s) ignored; line %d (%s) is scored"
                % (t.line, t.method, t.pose_id, t.round, t.outcome, scored[key].line, scored[key].outcome)
            )
            continue
        scored[key] = t
        if t.outcome == "success" and t.interventions > 0:
            issues.append(
                "line %d: %s pose %d round %d logged success with %d intervention(s); scored as a fail"
                % (t.line, t.method, t.pose_id, t.round, t.interventions)
            )
    return Scoring(scored, aborts, issues)


def method_order(present: Iterable[str], baseline: str, requested: Sequence[str] = ()) -> list[str]:
    """Requested methods first (in that order), then baseline, servo, bc, act, then the rest sorted."""
    present = set(present)
    order = list(dict.fromkeys(requested))
    for m in (baseline,) + KNOWN_METHOD_ORDER:
        if m in present and m not in order:
            order.append(m)
    order += sorted(present - set(order))
    return order


# --------------------------------------------------------------------------------------------
# Per-method and paired summaries


@dataclass
class MethodSummary:
    method: str
    k: int
    n: int
    ci: tuple[float, float] | None
    categories: Counter
    interventions: int
    dock_times_s: list[float]
    range_errors_cm: list[float]
    bearing_errors_deg: list[float]

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None


def summarize_method(method: str, scoring: Scoring, target_range_m: float) -> MethodSummary:
    trials = [t for key, t in scoring.scored.items() if key[0] == method]
    k = sum(t.success for t in trials)
    n = len(trials)
    categories = Counter(t.category for t in trials)
    categories["abort"] = sum(1 for t in scoring.aborts if t.method == method)
    return MethodSummary(
        method=method,
        k=k,
        n=n,
        ci=stats.wilson_interval(k, n, CONFIDENCE) if n else None,
        categories=categories,
        interventions=sum(t.interventions for t in trials),
        dock_times_s=[t.duration_s for t in trials if t.success and t.duration_s is not None],
        range_errors_cm=[abs(t.final_range_m - target_range_m) * 100.0 for t in trials if t.final_range_m is not None],
        bearing_errors_deg=[abs(t.final_bearing_deg) for t in trials if t.final_bearing_deg is not None],
    )


@dataclass
class Comparison:
    method: str
    baseline: str
    table: stats.PairedTable | None
    test: stats.McNemarResult | None
    diff: stats.PairedDiff | None
    only_baseline: list[Slot]  # baseline scored, method not
    only_method: list[Slot]  # method scored, baseline not


def compare_with_baseline(method: str, baseline: str, scoring: Scoring, n_boot: int, seed: int) -> Comparison:
    base = {t.slot: t for (m, _, _), t in scoring.scored.items() if m == baseline}
    other = {t.slot: t for (m, _, _), t in scoring.scored.items() if m == method}
    paired = sorted(set(base) & set(other))
    only_baseline = sorted(set(base) - set(other))
    only_method = sorted(set(other) - set(base))
    if not paired:
        return Comparison(method, baseline, None, None, None, only_baseline, only_method)
    y = np.array([base[s].success for s in paired], dtype=np.int64)
    x = np.array([other[s].success for s in paired], dtype=np.int64)
    table = stats.discordant_counts(y, x)
    return Comparison(
        method,
        baseline,
        table,
        stats.mcnemar_exact(table.b, table.c),
        stats.paired_bootstrap_diff(x, y, CONFIDENCE, n_boot=n_boot, seed=seed),
        only_baseline,
        only_method,
    )


def read_schedule(path: Path) -> list[tuple[str, int, int]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("method", "pose_id", "round") if c not in (reader.fieldnames or [])]
        if missing:
            raise InputError("%s is missing columns: %s" % (path, ", ".join(missing)))
        try:
            return [(r["method"].strip(), int(r["pose_id"]), int(r["round"])) for r in reader]
        except (TypeError, ValueError) as exc:
            raise InputError("%s: bad schedule row: %s" % (path, exc)) from None


# --------------------------------------------------------------------------------------------
# /policy/latency and /safety_gate/status logs


def parse_source(text: str) -> tuple[str, Path]:
    """'[LABEL=]PATH' -> (label, path); the label defaults to the file or bag name."""
    label, sep, rest = text.partition("=")
    if sep and label and os.sep not in label:
        return label, Path(rest).expanduser()
    path = Path(text).expanduser()
    return (path.stem if path.suffix else path.name), path


def _payload(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except ValueError:
                return None
    return obj if isinstance(obj, dict) else None


def _jsonl_payloads(path: Path, issues: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bad: list[int] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text == "---":
                continue
            try:
                payload = _payload(json.loads(text))
            except ValueError:
                payload = None
            if payload is None:
                bad.append(lineno)
            else:
                out.append(payload)
    if bad:
        shown = ", ".join(str(i) for i in bad[:5]) + (", ..." if len(bad) > 5 else "")
        issues.append("%s: %d line(s) are not JSON objects and were skipped (lines %s)" % (path, len(bad), shown))
    return out


def _is_bag(path: Path) -> bool:
    return (path / "metadata.yaml").is_file() if path.is_dir() else path.suffix in BAG_SUFFIXES


def find_bags(path: Path) -> list[Path]:
    if _is_bag(path):
        return [path]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        if "metadata.yaml" in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []
        else:
            found.extend(Path(dirpath) / f for f in sorted(filenames) if Path(f).suffix in BAG_SUFFIXES)
    return found


def _bag_payloads(bag: Path, topic: str, issues: list[str]) -> list[dict[str, Any]]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    out: list[dict[str, Any]] = []
    bad = 0
    # Foxy bags carry no message definitions; std_msgs/String is the same in every distro.
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        for connection, _, raw in reader.messages(connections=connections):
            try:
                payload = _payload(json.loads(reader.deserialize(raw, connection.msgtype).data))
            except (ValueError, AttributeError):
                payload = None
            if payload is None:
                bad += 1
            else:
                out.append(payload)
    if bad:
        issues.append("%s: %d %s message(s) are not JSON objects and were skipped" % (bag, bad, topic))
    return out


def load_segments(path: Path, topic: str, issues: list[str]) -> list[list[dict[str, Any]]]:
    """Payload lists, one per recording: one JSONL file, or each bag found under ``path``."""
    if not path.exists():
        raise InputError("no such file or directory: %s" % path)
    if path.is_dir() or path.suffix in BAG_SUFFIXES:
        bags = find_bags(path)
        if not bags:
            raise InputError("no rosbag2 recordings under %s" % path)
        segments = [_bag_payloads(bag, topic, issues) for bag in bags]
    else:
        segments = [_jsonl_payloads(path, issues)]
    if not any(segments):
        issues.append("%s: no %s records" % (path, topic))
    return segments


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if np.isfinite(value) else None


def _p50_p99(values: Sequence[float]) -> tuple[float, float] | None:
    if not values:
        return None
    p50, p99 = np.percentile(np.asarray(values, dtype=np.float64), [50.0, 99.0])
    return float(p50), float(p99)


@dataclass
class LatencySummary:
    source: str
    backend: str
    n: int
    fields: dict[str, tuple[float, float] | None]  # field -> (p50, p99) in ms


def summarize_latency(source: str, segments: Sequence[Sequence[dict[str, Any]]]) -> list[LatencySummary]:
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for record in (r for seg in segments for r in seg):
        by_backend.setdefault(str(record.get("backend") or "unknown"), []).append(record)
    out = []
    for backend in sorted(by_backend):
        records = by_backend[backend]
        fields = {}
        for name in LATENCY_FIELDS:
            values = [v for v in (_number(r.get(name)) for r in records) if v is not None]
            fields[name] = _p50_p99(values)
        out.append(LatencySummary(source, backend, len(records), fields))
    return out


@dataclass
class GateSummary:
    source: str
    n_status: int
    ticks: int  # control ticks covered by the log (counter deltas)
    missed: int  # missed deadlines among them
    restarts: int
    period_p50_ms: float | None  # median over status windows
    period_p99_ms: float | None  # median over status windows
    period_p99_worst_ms: float | None
    period_max_ms: float | None
    cb_p99_worst_us: float | None


def _nested(record: dict[str, Any], group: str, key: str) -> float | None:
    sub = record.get(group)
    return _number(sub.get(key)) if isinstance(sub, dict) else None


def summarize_gate(source: str, segments: Sequence[Sequence[dict[str, Any]]]) -> GateSummary:
    """Timing of safety_gate from its 2 Hz status.

    ticks and missed_deadlines are counters since the node started, so the log's share is the
    sum of their increments; the first status of each recording is only a baseline (what came
    before it is unknown), and a counter that goes down marks a node restart, counted from zero.
    period_ms and cb_latency_us summarise a rolling window of recent ticks in each status.
    """
    ticks = missed = restarts = 0
    for seg in segments:
        prev: tuple[int, int] | None = None
        for record in seg:
            t, m = _number(record.get("ticks")), _number(record.get("missed_deadlines"))
            if t is None or m is None:
                continue
            cur = (int(t), int(m))
            if prev is not None:
                if cur[0] < prev[0] or cur[1] < prev[1]:
                    restarts += 1
                    ticks += cur[0]
                    missed += cur[1]
                else:
                    ticks += cur[0] - prev[0]
                    missed += cur[1] - prev[1]
            prev = cur
    records = [r for seg in segments for r in seg]

    def collect(group: str, key: str) -> list[float]:
        return [v for v in (_nested(r, group, key) for r in records) if v is not None]

    p50s, p99s, maxes = collect("period_ms", "p50"), collect("period_ms", "p99"), collect("period_ms", "max")
    cb99 = collect("cb_latency_us", "p99")
    return GateSummary(
        source=source,
        n_status=len(records),
        ticks=ticks,
        missed=missed,
        restarts=restarts,
        period_p50_ms=float(np.median(p50s)) if p50s else None,
        period_p99_ms=float(np.median(p99s)) if p99s else None,
        period_p99_worst_ms=max(p99s) if p99s else None,
        period_max_ms=max(maxes) if maxes else None,
        cb_p99_worst_us=max(cb99) if cb99 else None,
    )


# --------------------------------------------------------------------------------------------
# Report


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else "%.*f" % (digits, value)


def _pct(value: float | None) -> str:
    return "—" if value is None else "%.1f%%" % (100.0 * value)


def _ci(ci: tuple[float, float] | None) -> str:
    return "—" if ci is None else "%.1f–%.1f%%" % (100.0 * ci[0], 100.0 * ci[1])


def _p(p: float) -> str:
    return "%.3g" % p


def _median_iqr(values: Sequence[float], digits: int = 1) -> str:
    if not values:
        return "—"
    q1, med, q3 = np.percentile(np.asarray(values, dtype=np.float64), [25.0, 50.0, 75.0])
    return "%.*f (%.*f–%.*f), n=%d" % (digits, med, digits, q1, digits, q3, len(values))


def _slots(slots: Sequence[Slot]) -> str:
    return ", ".join("pose %d round %d" % s for s in slots)


def _table(header: Sequence[str], rows: Iterable[Sequence[str]], align: str) -> list[str]:
    """align: one character per column, 'l' or 'r'."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


@dataclass
class Analysis:
    trials_path: Path
    n_rows: int
    baseline: str
    order: list[str]
    scoring: Scoring
    summaries: list[MethodSummary]
    comparisons: list[Comparison]
    issues: list[str]
    target_range_m: float
    n_boot: int
    schedule_missing: list[tuple[str, int, int]] | None = None
    schedule_extra: list[tuple[str, int, int]] | None = None
    latency: list[LatencySummary] = field(default_factory=list)
    gate: list[GateSummary] = field(default_factory=list)
    gate_rate_hz: float = GATE_RATE_HZ
    setting: str | None = None  # e.g. "real robot, on-board TensorRT" or "simulation (fake_robot)"


def analyze(
    trials_path: Path,
    baseline: str = "servo",
    methods: Sequence[str] = (),
    schedule_path: Path | None = None,
    latency_sources: Sequence[str] = (),
    gate_sources: Sequence[str] = (),
    target_range_m: float = TARGET_RANGE_M,
    gate_rate_hz: float = GATE_RATE_HZ,
    n_boot: int = 10000,
    seed: int = 0,
    setting: str | None = None,
) -> Analysis:
    trials, issues = load_trials(trials_path)
    scoring = score_trials(trials)
    issues += scoring.issues
    order = method_order({t.method for t in trials}, baseline, methods)
    summaries = [summarize_method(m, scoring, target_range_m) for m in order]
    comparisons = [compare_with_baseline(m, baseline, scoring, n_boot, seed) for m in order if m != baseline]
    analysis = Analysis(
        trials_path, len(trials), baseline, order, scoring, summaries, comparisons, issues, target_range_m, n_boot,
        gate_rate_hz=gate_rate_hz,
        setting=setting,
    )
    if schedule_path is not None:
        planned = read_schedule(schedule_path)
        planned_set = set(planned)
        analysis.schedule_missing = [s for s in planned if s not in scoring.scored]
        analysis.schedule_extra = sorted(k for k in scoring.scored if k not in planned_set)
    for text in latency_sources:
        label, path = parse_source(text)
        analysis.latency += summarize_latency(label, load_segments(path, LATENCY_TOPIC, issues))
    for text in gate_sources:
        label, path = parse_source(text)
        analysis.gate.append(summarize_gate(label, load_segments(path, GATE_TOPIC, issues)))
    return analysis


def _section_success(a: Analysis, chart_name: str) -> list[str]:
    out = ["## Success rate", ""]
    out += _table(
        ("Method", "Success k/n", "Success", "Wilson 95% CI"),
        ((s.method, "%d/%d" % (s.k, s.n), _pct(s.rate), _ci(s.ci)) for s in a.summaries),
        "lrrr",
    )
    return out + ["", "![Success rate per method with Wilson 95%% CI](%s)" % chart_name, ""]


def _section_paired(a: Analysis) -> list[str]:
    out = ["## Paired comparison with `%s` (McNemar, pairs on pose_id × round)" % a.baseline, ""]
    if not any(s.method == a.baseline and s.n for s in a.summaries):
        return out + ["No scored `%s` trials: no paired comparison is possible." % a.baseline, ""]
    if not a.comparisons:
        return out + ["No other method to compare.", ""]
    out += [
        "b = %s succeeded and the method failed; c = the method succeeded and %s failed. "
        "p values are two-sided (exact binomial on b + c; mid-p per Fagerland et al. 2013). "
        "Δ = method − %s success rate over the pairs, with a paired percentile bootstrap CI "
        "(%d resamples)." % (a.baseline, a.baseline, a.baseline, a.n_boot),
        "",
    ]
    rows = []
    for c in a.comparisons:
        if c.table is None:
            rows.append([c.method, "0"] + ["—"] * 8)
            continue
        rows.append([
            c.method,
            str(c.table.n_pairs),
            str(c.table.both_success),
            str(c.table.both_fail),
            str(c.table.b),
            str(c.table.c),
            _p(c.test.p_exact),
            _p(c.test.p_mid),
            "%+.1f pp" % (100.0 * c.diff.estimate),
            "%+.1f to %+.1f pp" % (100.0 * c.diff.low, 100.0 * c.diff.high),
        ])
    out += _table(
        ("Method", "Pairs", "Both succeed", "Both fail", "b", "c", "Exact p", "Mid-p", "Δ success", "Δ 95% CI"),
        rows,
        "lrrrrrrrrr",
    )
    out.append("")
    unpaired = [c for c in a.comparisons if c.only_baseline or c.only_method]
    if not unpaired:
        return out + ["Every slot is paired.", ""]
    out += ["Unpaired slots (left out of the comparison):", ""]
    for c in unpaired:
        if c.only_baseline:
            out.append("- `%s` missing where `%s` has a trial: %s" % (c.method, a.baseline, _slots(c.only_baseline)))
        if c.only_method:
            out.append("- `%s` missing where `%s` has a trial: %s" % (a.baseline, c.method, _slots(c.only_method)))
    return out + [""]


def _section_timing(a: Analysis) -> list[str]:
    out = ["## Time to dock and final error", ""]
    out += _table(
        ("Method", "Time to dock, s (successes)", "Final range error, cm", "Final bearing error, deg"),
        (
            (s.method, _median_iqr(s.dock_times_s), _median_iqr(s.range_errors_cm), _median_iqr(s.bearing_errors_deg))
            for s in a.summaries
        ),
        "lrrr",
    )
    return out + [
        "",
        "Median (IQR). Time to dock = start event to the judge's end event (duration_s), so it "
        "includes the 1 s success hold. Range error = absolute difference between the final lidar "
        "range and the %.2f m target; bearing error = absolute final marker bearing; both over all "
        "scored trials with a dock_judge reading at the end." % a.target_range_m,
        "",
    ]


def _section_outcomes(a: Analysis) -> list[str]:
    out = ["## Outcomes", ""]
    out += _table(
        ("Method", "Success", "Fail: timeout", "Fail: stop_obstacle", "Fail: intervention", "Fail: other",
         "Abort", "Interventions"),
        ([s.method] + [str(s.categories.get(c, 0)) for c in CATEGORIES] + [str(s.interventions)] for s in a.summaries),
        "lrrrrrrr",
    )
    out += ["", "## Success by start pose", ""]
    poses = sorted({pose for (_, pose, _) in a.scoring.scored})
    if not poses:
        return out + ["No scored trials.", ""]
    rows = []
    for pose in poses:
        row = [str(pose)]
        for m in a.order:
            ts = [t for (mm, p, _), t in a.scoring.scored.items() if mm == m and p == pose]
            row.append("%d/%d" % (sum(t.success for t in ts), len(ts)) if ts else "—")
        rows.append(row)
    return out + _table(["Pose"] + list(a.order), rows, "r" * (1 + len(a.order))) + [""]


def _section_schedule(a: Analysis) -> list[str]:
    if a.schedule_missing is None:
        return []
    out = ["## Schedule check", ""]
    if not a.schedule_missing and not a.schedule_extra:
        return out + ["Every scheduled trial has exactly one scored result, and nothing unscheduled was scored.", ""]
    if a.schedule_missing:
        out.append("- Scheduled but not scored (%d): %s" % (
            len(a.schedule_missing), ", ".join("%s pose %d round %d" % s for s in a.schedule_missing)))
    if a.schedule_extra:
        out.append("- Scored but not scheduled (%d): %s" % (
            len(a.schedule_extra), ", ".join("%s pose %d round %d" % s for s in a.schedule_extra)))
    return out + [""]


def _section_latency(a: Analysis) -> list[str]:
    if not a.latency:
        return []
    out = ["## Policy latency (/policy/latency)", ""]
    rows = []
    for lat in a.latency:
        row = [lat.source, lat.backend, str(lat.n)]
        for name in LATENCY_FIELDS:
            pair = lat.fields[name]
            if pair is None:
                row += ["—", "—"]
            else:
                row += [_fmt(pair[0]), _fmt(pair[1]) + ("*" if lat.n < P99_MIN_SAMPLES else "")]
        rows.append(row)
    out += _table(
        ("Source", "Backend", "n", "Infer p50", "Infer p99", "Total p50", "Total p99", "Image age p50",
         "Image age p99"),
        rows,
        "llrrrrrrr",
    )
    note = (
        "Milliseconds. Infer = backend call; total = tick start to /cmd_vel_in publish (decode + "
        "preprocess + infer); image age = tick start minus the frame's capture stamp."
    )
    if any(lat.n < P99_MIN_SAMPLES for lat in a.latency):
        note += " * p99 from fewer than %d samples." % P99_MIN_SAMPLES
    return out + ["", note, ""]


def _section_gate(a: Analysis) -> list[str]:
    if not a.gate:
        return []
    nominal = 1000.0 / a.gate_rate_hz
    rows = []
    for g in a.gate:
        share = "%.3f%%" % (100.0 * g.missed / g.ticks) if g.ticks else "—"
        jitter = None if g.period_p99_worst_ms is None else g.period_p99_worst_ms - nominal
        rows.append((
            g.source,
            str(g.n_status),
            str(g.ticks),
            "%d (%s)" % (g.missed, share),
            _fmt(g.period_p50_ms, 2),
            _fmt(g.period_p99_ms, 2),
            _fmt(g.period_p99_worst_ms, 2),
            _fmt(jitter, 2),
            _fmt(g.period_max_ms, 2),
            _fmt(g.cb_p99_worst_us, 0),
            str(g.restarts),
        ))
    out = ["## Safety gate timing (/safety_gate/status)", ""]
    out += _table(
        ("Source", "Status msgs", "Ticks", "Missed deadlines", "Period p50", "Period p99", "Worst p99",
         "Jitter p99", "Period max", "Callback p99, µs", "Restarts"),
        rows,
        "lrrrrrrrrrr",
    )
    return out + [
        "",
        "Periods in ms; nominal %.2f ms (rate_hz %g). A tick is a missed deadline when its period "
        "exceeds %.1f× nominal. Period p50/p99 are medians over the status windows (each covers "
        "the last ≤10000 ticks); worst p99, jitter p99 (worst p99 − nominal) and max are the worst "
        "window. Ticks and missed deadlines count only what the log covers."
        % (nominal, a.gate_rate_hz, MISSED_DEADLINE_FACTOR),
        "",
    ]


def _section_aborts_and_issues(a: Analysis) -> list[str]:
    out = ["## Aborted attempts", ""]
    out += [
        "- line %d: %s pose %d round %d%s" % (t.line, t.method, t.pose_id, t.round, " — " + t.notes if t.notes else "")
        for t in a.scoring.aborts
    ] or ["None."]
    out += ["", "## Data issues", ""]
    out += ["- " + issue for issue in a.issues] or ["None."]
    return out + [""]


def render_report(a: Analysis, chart_name: str = "success.png") -> str:
    header = [
        "# Docking trials report",
        "",
        "Setting: %s." % (a.setting or 'not stated (pass --setting, e.g. "real robot" or "simulation")'),
        "",
        "Source: `%s` (%d valid rows). Baseline: `%s`. Intervals are %d%%. A trial is scored when its outcome "
        "is success or fail; success also needs zero interventions. Aborted attempts are listed below "
        "and never counted." % (a.trials_path.name, a.n_rows, a.baseline, round(CONFIDENCE * 100)),
        "",
    ]
    sections = (
        header,
        _section_success(a, chart_name),
        _section_paired(a),
        _section_timing(a),
        _section_outcomes(a),
        _section_schedule(a),
        _section_latency(a),
        _section_gate(a),
        _section_aborts_and_issues(a),
    )
    return "\n".join(line for section in sections for line in section)


def plot_success(summaries: Sequence[MethodSummary], path: Path, setting: str | None = None) -> None:
    """Success rate per method with its Wilson CI, one colour per method in palette order."""
    if len(summaries) > len(SERIES_COLORS):
        raise ValueError("at most %d methods can be charted" % len(SERIES_COLORS))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    fig = Figure(figsize=(6.4, 4.2), dpi=150, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.10, 0.12, 0.86, 0.68), facecolor=BACKGROUND)
    x = np.arange(len(summaries))
    for i, s in enumerate(summaries):
        if not s.n:
            ax.text(i, 2.0, "no scored\ntrials", ha="center", va="bottom", color=TEXT, fontsize=9)
            continue
        rate = 100.0 * s.k / s.n
        low, high = (100.0 * v for v in s.ci)
        ax.bar(i, rate, width=0.42, color=SERIES_COLORS[i], zorder=2)
        ax.errorbar(
            i, rate, yerr=[[rate - low], [high - rate]], fmt="none", ecolor=TEXT_SECONDARY,
            elinewidth=1.4, capsize=7, capthick=1.4, zorder=3,
        )
        ax.text(i, high + 2.5, "%.0f%%\n%d/%d" % (rate, s.k, s.n), ha="center", va="bottom", color=TEXT,
                fontsize=10, linespacing=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.method for s in summaries], color=TEXT, fontsize=11)
    ax.set_xlim(-0.6, len(summaries) - 0.4)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.tick_params(axis="y", colors=TEXT, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    fig.text(0.10, 0.93, "Docking success rate by method", color=TEXT, fontsize=13, fontweight="bold")
    subtitle = "Whiskers: Wilson 95% CI; aborted attempts excluded"
    # The setting gets a line of its own: on one line with the note it ran off the figure.
    lines = [setting, subtitle] if setting else [subtitle]
    fig.text(0.10, 0.905, "\n".join(lines), va="top", color=TEXT_SECONDARY, fontsize=9, linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BACKGROUND)


# --------------------------------------------------------------------------------------------


def _csv_list(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trials", type=Path, help="trials.csv from trial_logger")
    parser.add_argument("--out-dir", type=Path, help="default: the directory of trials.csv")
    parser.add_argument("--baseline", default="servo", help="method the others are paired with (default %(default)s)")
    parser.add_argument("--methods", help="comma-separated display order (default: baseline, servo, bc, act, rest)")
    parser.add_argument("--schedule", type=Path, help="make_schedule.py CSV to check completeness against")
    parser.add_argument(
        "--latency", action="append", default=[], metavar="[LABEL=]PATH", help="/policy/latency bag or JSONL"
    )
    parser.add_argument(
        "--gate", action="append", default=[], metavar="[LABEL=]PATH", help="/safety_gate/status bag or JSONL"
    )
    parser.add_argument(
        "--target-range", type=float, default=TARGET_RANGE_M, help="docking range, m (default %(default)s)"
    )
    parser.add_argument(
        "--gate-rate-hz", type=float, default=GATE_RATE_HZ, help="safety_gate rate_hz (default %(default)s)"
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default %(default)s)")
    parser.add_argument("--setting", help='where the trials ran, shown in the report and chart, e.g. "real robot"')
    args = parser.parse_args(argv)
    if args.gate_rate_hz <= 0 or args.n_boot < 1:
        parser.error("--gate-rate-hz and --n-boot must be positive")

    try:
        analysis = analyze(
            args.trials,
            baseline=args.baseline,
            methods=_csv_list(args.methods),
            schedule_path=args.schedule,
            latency_sources=args.latency,
            gate_sources=args.gate,
            target_range_m=args.target_range,
            gate_rate_hz=args.gate_rate_hz,
            n_boot=args.n_boot,
            seed=args.seed,
            setting=args.setting,
        )
    except (OSError, InputError) as exc:
        print("analyze_trials: %s" % exc, file=sys.stderr)
        return 2
    out_dir = args.out_dir or args.trials.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = out_dir / "success.png"
    plot_success(analysis.summaries, chart, analysis.setting)
    report = out_dir / "report.md"
    report.write_text(render_report(analysis, chart.name))
    for s in analysis.summaries:
        print("%-8s %d/%d  %s  CI %s" % (s.method, s.k, s.n, _pct(s.rate), _ci(s.ci)))
    for c in analysis.comparisons:
        if c.test is not None:
            print("%-8s vs %s: b=%d c=%d exact p=%s mid-p=%s" % (c.method, c.baseline, c.test.b, c.test.c,
                                                             _p(c.test.p_exact), _p(c.test.p_mid)))
    if analysis.issues:
        print("%d data issue(s), see the report" % len(analysis.issues))
    print("wrote %s and %s" % (report, chart))
    return 0


if __name__ == "__main__":
    sys.exit(main())
