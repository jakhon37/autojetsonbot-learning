"""Write one CSV row per finished episode (the per-trial log of docs/INTERFACES.md).

Subscribes
    /episode/event       reliable + transient_local
        start <method> <pose_id> <round>   opens an episode; one that is still open is written
                                           first as outcome "abort", note "no end before next start"
        end <outcome> [reason ...]         closes the open episode: one row, notes = reason;
                                           an end with no open episode (e.g. the judge's and
                                           the operator's for the same trial) is ignored
        intervention [note ...]            counts an operator intervention in the open episode
                                           (``episode intervention <note>``) and adds the note
                                           to the row's notes; an end whose reason starts with
                                           "intervention" counts at least one (dock_judge's
                                           "end fail intervention" can arrive before the
                                           operator's message: they come from different writers)
    /dock_judge/status   the newest status before the end gives final_range_m and
                         final_bearing_deg; if it is older than status_timeout_s (or never
                         came) both are left empty and the notes say so

Columns: method,pose_id,round,outcome,duration_s,final_range_m,final_bearing_deg,interventions,notes
duration_s is end minus start arrival time on this node's clock. The file is appended to and
flushed per row; the header is written only when the file is new or empty.

Parameters
    path                ~/jbl_trials/trials.csv
    status_timeout_s    1.0

Start the logger before the first trial: when it joins late, transient-local history from
several /episode/event writers can arrive out of order and episodes before its start are
not logged.
"""

import csv
import json
import os
import signal
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from jbl_nodes.episode_cli import (
    EVENT_TOPIC,
    EpisodeEvent,
    event_qos,
    format_episode_event,
    parse_episode_event,
)
from jbl_nodes.judge_logic import OUTCOME_FAIL, STATUS_TOPIC, is_intervention

CSV_COLUMNS = (
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
NOTE_NO_END = "no end before next start"
NOTE_NO_STATUS = "no dock_judge status"


class TrialCsv(object):
    """Append-only CSV with the trial columns."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))

    def existing_header(self) -> Optional[List[str]]:
        """The first row of an existing non-empty file, else None."""
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            return None
        with open(self.path, newline="") as f:
            return next(csv.reader(f), None)

    def append(self, row: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())  # a trial row must survive the robot losing power


def _fmt(value: Optional[float], digits: int) -> str:
    return "" if value is None else "%.*f" % (digits, value)


def trial_row(
    start: EpisodeEvent,
    outcome: str,
    duration_s: float,
    final_range_m: Optional[float],
    final_bearing_deg: Optional[float],
    interventions: int,
    notes: List[str],
) -> Dict[str, Any]:
    return {
        "method": start.method,
        "pose_id": start.pose_id,
        "round": start.round,
        "outcome": outcome,
        "duration_s": _fmt(duration_s, 3),
        "final_range_m": _fmt(final_range_m, 4),
        "final_bearing_deg": _fmt(final_bearing_deg, 2),
        "interventions": interventions,
        "notes": "; ".join(n for n in notes if n),
    }


class _OpenEpisode(object):
    def __init__(self, start: EpisodeEvent, t_ns: int) -> None:
        self.start = start
        self.t_ns = t_ns
        self.interventions = 0
        self.notes: List[str] = []  # operator notes from intervention events


class TrialLogger(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("trial_logger", **kwargs)
        self._csv = TrialCsv(str(self.declare_parameter("path", "~/jbl_trials/trials.csv").value))
        self._status_timeout_ns = int(float(self.declare_parameter("status_timeout_s", 1.0).value) * 1e9)
        self._episode: Optional[_OpenEpisode] = None
        self._status: Optional[Dict[str, Any]] = None
        self._status_t_ns: Optional[int] = None

        header = self._csv.existing_header()
        if header is not None and tuple(header) != CSV_COLUMNS:
            self.get_logger().warning(
                "%s has header %s, expected %s; appending anyway"
                % (self._csv.path, header, list(CSV_COLUMNS))
            )
        self.create_subscription(String, EVENT_TOPIC, self._on_event, event_qos())
        self.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
        self.get_logger().info("logging trials to %s" % self._csv.path)

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(status, dict):
            self._status, self._status_t_ns = status, self._now_ns()

    def _on_event(self, msg: String) -> None:
        now_ns = self._now_ns()
        if is_intervention(msg.data):
            if self._episode is None:
                self.get_logger().warning("'%s' outside an episode; not counted" % msg.data)
            else:
                self._episode.interventions += 1
                self._episode.notes.append(" ".join(msg.data.split()[1:]))
            return
        try:
            event = parse_episode_event(msg.data)
        except ValueError as exc:
            self.get_logger().warning("ignoring episode event %r: %s" % (msg.data, exc))
            return
        if event.kind == "start":
            if self._episode is not None:
                self._write("abort", NOTE_NO_END, now_ns)
            self._episode = _OpenEpisode(event, now_ns)
        elif self._episode is None:
            self.get_logger().info("'%s' with no open episode; ignored" % msg.data)
        else:
            if event.outcome == OUTCOME_FAIL and is_intervention(event.reason):
                self._episode.interventions = max(self._episode.interventions, 1)
            self._write(event.outcome, event.reason, now_ns)

    def _write(self, outcome: str, reason: str, now_ns: int) -> None:
        episode, self._episode = self._episode, None
        notes = [reason] + episode.notes
        final_range = final_bearing = None
        status_fresh = self._status_t_ns is not None and now_ns - self._status_t_ns <= self._status_timeout_ns
        if status_fresh:
            final_range = _as_float(self._status.get("front_range_m"))
            final_bearing = _as_float(self._status.get("bearing_deg"))
        else:
            notes.append(NOTE_NO_STATUS)
        row = trial_row(
            episode.start,
            outcome,
            (now_ns - episode.t_ns) * 1e-9,
            final_range,
            final_bearing,
            episode.interventions,
            notes,
        )
        try:
            self._csv.append(row)
        except OSError as exc:
            self.get_logger().error(
                "could not write the row for '%s' to %s: %s; row was %s"
                % (format_episode_event(episode.start), self._csv.path, exc, row)
            )
            return
        self.get_logger().info(
            "'%s' -> %s after %s s" % (format_episode_event(episode.start), outcome, row["duration_s"])
        )


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrialLogger()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ros2 launch forwards Ctrl-C again: finish cleanup
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
