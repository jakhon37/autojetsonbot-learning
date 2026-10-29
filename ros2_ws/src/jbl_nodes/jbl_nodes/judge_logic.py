"""Pure docking logic shared by dock_judge, servo_baseline and their tests (no ROS, no OpenCV).

- front_range: median lidar range in a sector around the forward direction.
- DockJudge: the success / fail state machine of docs/INTERFACES.md ("Docking geometry and
  success"), driven by explicit timestamps so it can be tested without a clock.
- servo_command: the classical baseline's control law, plus the marker-size range estimate it
  falls back on when the lidar has no valid reading. It lives here, not in servo_baseline,
  so it can be tested without rclpy; servo_baseline imports it.
- is_intervention: recognises the "intervention [note ...]" /episode/event payload.

Python 3.8 compatible; numpy only.
"""

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

STATUS_TOPIC = "/dock_judge/status"
GATE_STOP_OBSTACLE = "stop_obstacle"
INTERVENTION_WORD = "intervention"  # /episode/event "intervention [note ...]"

OUTCOME_SUCCESS = "success"
OUTCOME_FAIL = "fail"
REASON_TIMEOUT = "timeout"
REASON_OBSTACLE = "stop_obstacle"
REASON_INTERVENTION = "intervention"

STATUS_KEYS = (
    "t",
    "marker_visible",
    "marker_id",
    "bearing_deg",
    "marker_area_frac",
    "front_range_m",
    "speed_v",
    "speed_w",
    "in_band",
    "aligned",
    "stopped",
    "success",
    "success_hold_s",
)

_ANGLE_EPS = 1e-9  # keeps sector edges inclusive despite float error in angle_min + i * increment


def is_intervention(event_text: str) -> bool:
    """True for an /episode/event payload that reports an operator intervention."""
    words = event_text.split()
    return bool(words) and words[0] == INTERVENTION_WORD


def wrap_angle(angle: Any) -> Any:
    """Wrap radians (scalar or array) into [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def front_range(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    forward_angle: float = 0.0,
    half_angle: float = 0.17,
) -> Optional[float]:
    """Median of the valid readings within +-half_angle of forward_angle, or None if there are none.

    Beam i points at angle_min + i * angle_increment; the increment may be negative (rplidar_ros
    with inverted scans) and the angles may run past +-pi (e.g. 0 .. 2 pi), so the sector is
    found by wrapping each beam's offset from forward_angle. A reading is valid when it is
    finite and inside [range_min, range_max]: the RPLidar reports objects closer than its
    minimum range as 0, inf or a value below range_min, all of which mean "unknown".
    A non-positive or non-finite range_max is treated as "no upper limit".
    """
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return None
    angles = angle_min + angle_increment * np.arange(r.size, dtype=np.float64)
    in_sector = np.abs(wrap_angle(angles - forward_angle)) <= half_angle + _ANGLE_EPS
    upper = range_max if math.isfinite(range_max) and range_max > 0.0 else math.inf
    with np.errstate(invalid="ignore"):  # NaN readings compare False, which is what we want
        valid = in_sector & np.isfinite(r) & (r > 0.0) & (r >= range_min) & (r <= upper)
    if not valid.any():
        return None
    return float(np.median(r[valid]))


class MarkerObservation(NamedTuple):
    """The docking marker as seen in one camera frame."""

    marker_id: int
    bearing_deg: float  # positive = marker left of the image centre
    area_frac: float  # marker area / image area


class JudgeConfig(NamedTuple):
    marker_id: int = 7
    band_min: float = 0.18  # m, lidar front-sector median
    band_max: float = 0.32
    bearing_tol_deg: float = 10.0
    hold_s: float = 1.0
    timeout_s: float = 30.0
    stopped_v: float = 0.02  # m/s, |odom v| and |requested v| at or below count as stopped
    stopped_w: float = 0.05  # rad/s, the same for w
    obstacle_fail_s: float = 2.0  # gate in stop_obstacle for longer than this, while not aligned


def validate_judge_config(cfg: JudgeConfig) -> None:
    if not 0.0 <= cfg.band_min < cfg.band_max:
        raise ValueError("need 0 <= band_min < band_max, got %s, %s" % (cfg.band_min, cfg.band_max))
    if cfg.bearing_tol_deg <= 0.0:
        raise ValueError("bearing_tol_deg must be positive, got %s" % cfg.bearing_tol_deg)
    if cfg.hold_s < 0.0 or cfg.timeout_s <= 0.0 or cfg.obstacle_fail_s < 0.0:
        raise ValueError(
            "need hold_s >= 0, timeout_s > 0, obstacle_fail_s >= 0, got %s, %s, %s"
            % (cfg.hold_s, cfg.timeout_s, cfg.obstacle_fail_s)
        )
    if cfg.stopped_v < 0.0 or cfg.stopped_w < 0.0:
        raise ValueError("stopped_v and stopped_w must be >= 0")


class Verdict(NamedTuple):
    outcome: str  # OUTCOME_SUCCESS or OUTCOME_FAIL
    reason: str  # "" for success, else one of the REASON_* values
    t: float  # when the deciding condition was met
    duration_s: float  # t - episode start


class DockJudge(object):
    """Decides each docking episode once, from a stream of perception and odometry samples.

    Call start_episode() / end_episode() / intervention() on /episode/event and update() on
    every judge tick. Inputs that are stale or unknown are passed as None and count as "not
    satisfied". The three success conditions are tracked live, inside and outside episodes,
    so status() always shows how long the robot has been docked; an episode only counts the
    part of a hold that happened after its start.

    Within an episode, the first of these to happen decides it (ties go to success):
      success  in_band, aligned and stopped continuously for hold_s; stopped means odometry
               and the controller's latest fresh request (/cmd_vel_in) are both still, so a
               robot that only the gate's slow zone has slowed below stopped_v does not count
      fail     timeout_s since the start
      fail     gate state stop_obstacle for more than obstacle_fail_s while not aligned
      fail     an operator intervention
    """

    def __init__(self, config: JudgeConfig = JudgeConfig()) -> None:
        validate_judge_config(config)
        self._cfg = config
        self._start: Optional[float] = None  # set while an episode runs
        self._verdict: Optional[Verdict] = None
        self._hold_since: Optional[float] = None
        self._obstacle_since: Optional[float] = None
        self._last_t: Optional[float] = None
        self._marker: Optional[MarkerObservation] = None
        self._front_range: Optional[float] = None
        self._v = 0.0
        self._w = 0.0
        self._in_band = False
        self._aligned = False
        self._stopped = False
        self._hold = 0.0

    @property
    def config(self) -> JudgeConfig:
        return self._cfg

    @property
    def episode_active(self) -> bool:
        """An episode has started, has not ended and is not decided yet."""
        return self._start is not None and self._verdict is None

    @property
    def verdict(self) -> Optional[Verdict]:
        """The current (or last ended) episode's verdict, if it was decided."""
        return self._verdict

    def start_episode(self, now: float) -> None:
        """Begin judging; a running, undecided episode is abandoned without a verdict."""
        self._start = now
        self._verdict = None
        self._hold_since = None
        self._obstacle_since = None

    def end_episode(self) -> None:
        """An external end: stop judging. Idempotent; the verdict (if any) is kept."""
        self._start = None

    def intervention(self, now: float) -> Optional[Verdict]:
        """An operator intervened: the running episode fails. None if nothing was running."""
        if not self.episode_active:
            return None
        return self._decide(OUTCOME_FAIL, REASON_INTERVENTION, now)

    def update(
        self,
        now: float,
        marker: Optional[MarkerObservation],
        front_range_m: Optional[float],
        v: Optional[float],
        w: Optional[float],
        gate_state: Optional[str],
        cmd: Optional[Tuple[float, float]] = None,
    ) -> Optional[Verdict]:
        """Feed one sample; returns the verdict on the tick that decides an episode, else None.

        cmd is the controller's latest fresh request (/cmd_vel_in v, w), None when there is none
        (the gate then outputs zero, so none counts as still).
        """
        cfg = self._cfg
        if self._last_t is not None and now < self._last_t:
            # The clock jumped back (sim time restarted): durations measured so far are void.
            self._hold_since = None
            self._obstacle_since = None
            if self._start is not None:
                self._start = now
        self._last_t = now

        self._marker = marker
        self._front_range = front_range_m
        if v is not None and w is not None:
            self._v, self._w = float(v), float(w)
        self._in_band = front_range_m is not None and cfg.band_min <= front_range_m <= cfg.band_max
        self._aligned = marker is not None and abs(marker.bearing_deg) <= cfg.bearing_tol_deg
        self._stopped = (
            v is not None
            and w is not None
            and abs(v) <= cfg.stopped_v
            and abs(w) <= cfg.stopped_w
            and (cmd is None or (abs(cmd[0]) <= cfg.stopped_v and abs(cmd[1]) <= cfg.stopped_w))
        )

        if self._in_band and self._aligned and self._stopped:
            if self._hold_since is None:
                self._hold_since = now
            self._hold = now - self._hold_since
        else:
            self._hold_since = None
            self._hold = 0.0

        if gate_state == GATE_STOP_OBSTACLE and not self._aligned:
            if self._obstacle_since is None:
                self._obstacle_since = now
        else:
            self._obstacle_since = None

        if not self.episode_active:
            return None
        candidates: List[Tuple[float, str, str]] = []
        if self._hold_since is not None and now - self._hold_since >= cfg.hold_s:
            candidates.append((self._hold_since + cfg.hold_s, OUTCOME_SUCCESS, ""))
        if self._obstacle_since is not None and now - self._obstacle_since > cfg.obstacle_fail_s:
            candidates.append((self._obstacle_since + cfg.obstacle_fail_s, OUTCOME_FAIL, REASON_OBSTACLE))
        if now - self._start >= cfg.timeout_s:
            candidates.append((self._start + cfg.timeout_s, OUTCOME_FAIL, REASON_TIMEOUT))
        if not candidates:
            return None
        t, outcome, reason = min(candidates, key=lambda c: c[0])
        return self._decide(outcome, reason, t)

    def _decide(self, outcome: str, reason: str, t: float) -> Verdict:
        self._verdict = Verdict(outcome, reason, t, t - self._start)
        return self._verdict

    def status(self) -> Dict[str, Any]:
        """The /dock_judge/status JSON object for the latest update().

        success is live: the conditions have held for at least hold_s right now. speed_v and
        speed_w are the latest odometry the judge received (0.0 before any); when that sample
        is stale, stopped is false.
        """
        m = self._marker
        return {
            "t": float(self._last_t if self._last_t is not None else 0.0),
            "marker_visible": m is not None,
            "marker_id": int(m.marker_id) if m is not None else None,
            "bearing_deg": round(float(m.bearing_deg), 3) if m is not None else None,
            "marker_area_frac": round(float(m.area_frac), 6) if m is not None else None,
            "front_range_m": round(float(self._front_range), 4) if self._front_range is not None else None,
            "speed_v": round(self._v, 4),
            "speed_w": round(self._w, 4),
            "in_band": self._in_band,
            "aligned": self._aligned,
            "stopped": self._stopped,
            "success": self._hold_since is not None and self._hold >= self._cfg.hold_s,
            "success_hold_s": round(self._hold, 3),
        }


# ------------------------------------------------------------------ classical servo baseline


class ServoParams(NamedTuple):
    target_range: float = 0.25  # m, lidar front range to stop at
    range_tol: float = 0.02  # m; at or below target_range + range_tol forward motion stops
    k_v: float = 1.0  # (m/s) per m of range error
    k_w: float = 1.2  # (rad/s) per rad of bearing
    v_max: float = 0.15  # m/s
    w_max: float = 0.8  # rad/s
    search_w: float = 0.5  # rad/s, rotation in place while the marker is not visible
    slow_bearing_deg: float = 25.0  # v falls linearly to 0 as |bearing| grows to this
    bearing_deadband_deg: float = 4.0  # no rotation within this; well inside the judge's 10 deg


def validate_servo_params(p: ServoParams) -> None:
    if p.target_range <= 0.0 or p.range_tol < 0.0:
        raise ValueError("need target_range > 0 and range_tol >= 0")
    if min(p.k_v, p.k_w, p.v_max, p.w_max) <= 0.0:
        raise ValueError("k_v, k_w, v_max and w_max must be positive")
    if p.search_w < 0.0 or p.bearing_deadband_deg < 0.0 or p.slow_bearing_deg <= p.bearing_deadband_deg:
        raise ValueError("need search_w >= 0 and 0 <= bearing_deadband_deg < slow_bearing_deg")


def servo_command(
    bearing_deg: Optional[float],
    range_m: Optional[float],
    params: ServoParams = ServoParams(),
    search_sign: float = 1.0,
) -> Tuple[float, float]:
    """Classical docking law -> (v m/s, w rad/s).

    bearing_deg   marker bearing (positive = left of centre), None if the marker is not visible
    range_m       best estimate of the lidar front range, None if unknown
    search_sign   direction to search in when the marker is not visible (+1 = turn left)

    Marker visible: w = k_w * bearing (left of centre -> positive yaw, turning towards it),
    zero inside the deadband; v = clip(k_v * (range - target), 0, v_max), scaled by
    (1 - |bearing| / slow_bearing_deg) so the robot turns before it drives, and zero once the
    range is at or below target + range_tol (never reverses). So (0, 0) needs the range at or
    below target + range_tol and the bearing inside the deadband, not just anywhere in the
    judge's band: above target + range_tol it still creeps forward at k_v * (range - target).
    Marker not visible: rotate in place at search_w.
    """
    p = params
    if bearing_deg is None:
        return 0.0, math.copysign(p.search_w, search_sign)
    if abs(bearing_deg) <= p.bearing_deadband_deg:
        w = 0.0
    else:
        w = min(max(p.k_w * math.radians(bearing_deg), -p.w_max), p.w_max)
    if range_m is None or range_m <= p.target_range + p.range_tol:
        v = 0.0
    else:
        heading_scale = max(0.0, 1.0 - abs(bearing_deg) / p.slow_bearing_deg)
        v = min(p.k_v * (range_m - p.target_range), p.v_max) * heading_scale
    return v, w


def range_from_marker_area(
    area_frac: float,
    image_width: int,
    image_height: int,
    hfov_rad: float,
    marker_side_m: float = 0.10,
    camera_ahead_of_lidar_m: float = 0.033,
) -> float:
    """Lidar-equivalent front range from the marker's apparent size (pinhole, fronto-parallel).

    The marker's side in pixels is sqrt(area); its distance from the camera is
    side_m * f / side_px with f = (W / 2) / tan(HFOV / 2). The lidar sits camera_ahead_of_lidar_m
    behind the camera (URDF: lidar x = 0.064 m, camera at the bumper, x ~ 0.097 m). A marker
    seen at an angle looks smaller, so this overestimates the range off-axis: fine for a
    fallback that only has to say "close" or "far" when the lidar cannot.
    """
    if area_frac <= 0.0:
        raise ValueError("area_frac must be positive, got %s" % area_frac)
    focal_px = 0.5 * image_width / math.tan(0.5 * hfov_rad)
    side_px = math.sqrt(area_frac * image_width * image_height)
    return marker_side_m * focal_px / side_px + camera_ahead_of_lidar_m
