"""Pure docking logic: front-sector range, the dock judge state machine and the servo law."""

import json
import math

import numpy as np
import pytest

from jbl_nodes.judge_logic import (
    OUTCOME_FAIL,
    OUTCOME_SUCCESS,
    REASON_INTERVENTION,
    REASON_OBSTACLE,
    REASON_TIMEOUT,
    STATUS_KEYS,
    DockJudge,
    JudgeConfig,
    MarkerObservation,
    ServoParams,
    front_range,
    is_intervention,
    range_from_marker_area,
    servo_command,
    validate_servo_params,
)

N = 360
INC = 2.0 * math.pi / N


def scan(values_by_deg, default=5.0, angle_min=-math.pi, increment=INC, n=N):
    """ranges where the beam whose angle is nearest to each key (deg) gets that value."""
    r = np.full(n, default, dtype=np.float64)
    angles = angle_min + increment * np.arange(n)
    for deg, value in values_by_deg.items():
        offset = (angles - math.radians(deg) + math.pi) % (2 * math.pi) - math.pi
        r[int(np.argmin(np.abs(offset)))] = value
    return r


# ---------------------------------------------------------------- front_range


def test_front_range_is_the_sector_median():
    r = scan({-8: 0.30, -4: 0.26, 0: 0.25, 4: 0.24, 8: 0.20, 12: 0.10, -15: 0.12}, default=3.0)
    # +-0.17 rad = +-9.7 deg: the 0.10 at 12 deg and 0.12 at -15 deg are outside the sector;
    # the beams between the listed ones are 3.0 and count too (19 beams in all).
    got = front_range(r, -math.pi, INC, 0.15, 12.0, 0.0, 0.17)
    sector = [3.0] * 14 + [0.30, 0.26, 0.25, 0.24, 0.20]
    assert got == pytest.approx(float(np.median(sector)))
    only = np.full(N, np.inf)
    only[[179, 180, 181]] = [0.30, 0.25, 0.20]
    assert front_range(only, -math.pi, INC, 0.15, 12.0) == pytest.approx(0.25)


def test_front_range_ignores_invalid_readings():
    r = np.full(N, np.inf)
    idx = [176, 177, 178, 179, 180, 181, 182, 183]  # -4 .. +3 deg
    r[idx] = [0.0, np.nan, -1.0, 0.10, 0.22, 0.24, 13.0, np.inf]
    assert front_range(r, -math.pi, INC, 0.15, 12.0) == pytest.approx(0.23)
    r[[180, 181]] = 0.0
    assert front_range(r, -math.pi, INC, 0.15, 12.0) is None
    assert front_range([], -math.pi, INC, 0.15, 12.0) is None
    # A driver that leaves range_max at 0 or inf: no upper limit.
    r[182] = np.inf
    r[180] = 20.0
    assert front_range(r, -math.pi, INC, 0.15, 0.0) == pytest.approx(20.0)
    assert front_range(r, -math.pi, INC, 0.15, math.inf) == pytest.approx(20.0)


def test_front_range_wraps_around_two_pi():
    # angle_min = 0, angles 0 .. 2 pi: the forward sector straddles both ends of the array.
    # Only a sector that includes both ends has the median 0.26.
    r = np.full(N, 4.0)
    r[0] = 0.26  # 0 deg
    r[1:6] = 0.25  # 1 .. 5 deg
    r[-5:] = 0.27  # 355 .. 359 deg = -5 .. -1 deg
    assert front_range(r, 0.0, INC, 0.15, 12.0, 0.0, math.radians(5.5)) == pytest.approx(0.26)
    # Facing backwards on a -pi .. pi scan also straddles the ends.
    r = np.full(N, 4.0)
    r[0] = 0.6  # -180 deg
    r[1:5] = 0.4  # -179 .. -176 deg
    r[-4:] = 0.8  # 176 .. 179 deg
    for forward in (math.pi, -math.pi, 3.0 * math.pi):
        assert front_range(r, -math.pi, INC, 0.15, 12.0, forward, math.radians(4.5)) == pytest.approx(0.6)


def test_front_range_negative_increment():
    # rplidar_ros with inverted scans: angle_min = +pi, angle_increment < 0.
    r = scan({0: 0.25, 3: 0.24, -3: 0.26, 90: 0.1}, default=np.inf, angle_min=math.pi, increment=-INC)
    assert front_range(r, math.pi, -INC, 0.15, 12.0) == pytest.approx(0.25)
    # forward_angle selects the sector in the scan's own angles regardless of beam order
    left = scan({90: 0.4}, default=np.inf, angle_min=math.pi, increment=-INC)
    assert front_range(left, math.pi, -INC, 0.15, 12.0, math.radians(90), 0.05) == pytest.approx(0.4)
    assert front_range(left, math.pi, -INC, 0.15, 12.0, math.radians(-90), 0.05) is None


def test_front_range_sector_edges_are_inclusive():
    r = np.full(N, np.inf)
    r[190] = 0.3  # exactly +10 deg
    half = 10 * INC
    assert front_range(r, -math.pi, INC, 0.15, 12.0, 0.0, half) == pytest.approx(0.3)
    assert front_range(r, -math.pi, INC, 0.15, 12.0, 0.0, half - 1e-6) is None


# ---------------------------------------------------------------- DockJudge

ALIGNED = MarkerObservation(7, 2.0, 0.1)
DOCKED = dict(marker=ALIGNED, front_range_m=0.25, v=0.0, w=0.0, gate_state="stop_cmd_timeout")
DT = 0.1


def run(judge, t0, t1, **inputs):
    """Tick the judge at 10 Hz over [t0, t1); return [(t, verdict)] for decisive ticks."""
    sample = dict(DOCKED)
    sample.update(inputs)
    out = []
    for i in range(int(round((t1 - t0) / DT))):
        t = t0 + i * DT
        verdict = judge.update(t, **sample)
        if verdict is not None:
            out.append((t, verdict))
    return out


def test_success_after_hold_time_exactly_once():
    judge = DockJudge()
    judge.start_episode(100.0)
    decided = run(judge, 100.0, 103.0)
    assert len(decided) == 1, "a verdict is returned exactly once"
    t, verdict = decided[0]
    assert t == pytest.approx(101.0)
    assert (verdict.outcome, verdict.reason) == (OUTCOME_SUCCESS, "")
    assert verdict.duration_s == pytest.approx(1.0)
    assert not judge.episode_active and judge.verdict == verdict
    status = judge.status()
    assert status["success"] and status["success_hold_s"] == pytest.approx(2.9)


def test_hold_restarts_when_any_condition_breaks():
    judge = DockJudge()
    judge.start_episode(0.0)
    assert run(judge, 0.0, 0.9) == []
    assert judge.update(0.9, ALIGNED, 0.25, 0.03, 0.0, None) is None  # moving: hold breaks
    assert judge.status()["success_hold_s"] == 0.0 and not judge.status()["stopped"]
    decided = run(judge, 1.0, 2.5)
    assert decided[0][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "change, ok",
    [
        (dict(front_range_m=0.18), True),
        (dict(front_range_m=0.32), True),
        (dict(front_range_m=0.179), False),
        (dict(front_range_m=0.321), False),
        (dict(front_range_m=None), False),
        (dict(marker=MarkerObservation(7, 10.0, 0.1)), True),
        (dict(marker=MarkerObservation(7, -10.0, 0.1)), True),
        (dict(marker=MarkerObservation(7, 10.01, 0.1)), False),
        (dict(marker=MarkerObservation(7, -10.01, 0.1)), False),
        (dict(marker=None), False),
        (dict(v=0.02, w=-0.05), True),
        (dict(v=-0.021), False),
        (dict(w=0.051), False),
        (dict(v=None, w=None), False),
    ],
)
def test_success_conditions_and_thresholds(change, ok):
    judge = DockJudge()
    judge.start_episode(0.0)
    decided = run(judge, 0.0, 2.0, **change)
    assert bool(decided) == ok
    status = judge.status()
    assert status["success"] == ok
    if ok:
        assert decided[0][1].outcome == OUTCOME_SUCCESS


def test_timeout_fails_at_30_s():
    judge = DockJudge()
    judge.start_episode(5.0)
    decided = run(judge, 5.0, 40.0, front_range_m=0.6)
    assert len(decided) == 1
    t, verdict = decided[0]
    assert t == pytest.approx(35.0)
    assert (verdict.outcome, verdict.reason, verdict.duration_s) == (
        OUTCOME_FAIL,
        REASON_TIMEOUT,
        pytest.approx(30.0),
    )


def test_success_completing_before_timeout_wins_within_one_tick():
    judge = DockJudge(JudgeConfig(timeout_s=2.05, hold_s=1.0))
    judge.start_episode(0.0)
    assert run(judge, 0.0, 1.0, front_range_m=0.6) == []
    # docked from t=1.0: the hold completes at 2.0, the timeout at 2.05; the tick at 2.1 sees both
    assert judge.update(1.0, **DOCKED) is None
    assert judge.update(1.95, **DOCKED) is None
    verdict = judge.update(2.1, **DOCKED)
    assert verdict.outcome == OUTCOME_SUCCESS and verdict.t == pytest.approx(2.0)


def test_pushed_into_obstacle_fails_after_more_than_two_seconds_unaligned():
    judge = DockJudge()
    judge.start_episode(0.0)
    unaligned = dict(DOCKED, marker=MarkerObservation(7, 25.0, 0.1), gate_state="stop_obstacle")
    assert run(judge, 0.0, 2.0, **unaligned) == []  # 0.0 .. 1.9
    assert judge.update(2.0, **unaligned) is None  # exactly 2.0 s is not "> 2 s"
    verdict = judge.update(2.1, **unaligned)
    assert (verdict.outcome, verdict.reason) == (OUTCOME_FAIL, REASON_OBSTACLE)
    assert verdict.t == pytest.approx(2.0)


def test_obstacle_while_aligned_or_marker_lost_briefly():
    judge = DockJudge()
    judge.start_episode(0.0)
    # Aligned and stopped right against the box (in band): the gate saying stop_obstacle is fine.
    decided = run(judge, 0.0, 1.5, gate_state="stop_obstacle")
    assert decided[0][1].outcome == OUTCOME_SUCCESS

    judge.start_episode(10.0)
    lost = dict(marker=None, gate_state="stop_obstacle", front_range_m=0.17)
    assert run(judge, 10.0, 11.5, **lost) == []
    # Realigning resets the obstacle timer; losing the marker again starts it from zero.
    assert run(judge, 11.5, 11.7, gate_state="stop_obstacle", front_range_m=0.17) == []
    assert run(judge, 11.7, 13.7, **lost) == []
    decided = run(judge, 13.7, 14.2, **lost)
    assert decided[0][1].reason == REASON_OBSTACLE and decided[0][1].t == pytest.approx(13.7)


def test_intervention_fails_the_running_episode_only():
    judge = DockJudge()
    assert judge.intervention(0.0) is None
    judge.start_episode(1.0)
    run(judge, 1.0, 1.5, front_range_m=0.5)
    verdict = judge.intervention(1.55)
    assert (verdict.outcome, verdict.reason) == (OUTCOME_FAIL, REASON_INTERVENTION)
    assert verdict.duration_s == pytest.approx(0.55)
    assert judge.intervention(1.6) is None
    assert run(judge, 1.6, 5.0) == [], "a decided episode is never decided again"


def test_no_verdict_outside_an_episode_and_start_resets_the_hold():
    judge = DockJudge()
    assert run(judge, 0.0, 3.0) == []
    status = judge.status()
    assert status["success"] and status["success_hold_s"] == pytest.approx(2.9), "live outside episodes"
    judge.start_episode(3.0)
    decided = run(judge, 3.0, 5.0)
    assert decided[0][0] == pytest.approx(4.0), "the hold before the start does not count"

    judge.start_episode(10.0)
    judge.end_episode()
    assert not judge.episode_active
    assert run(judge, 10.0, 50.0, front_range_m=1.0) == [], "an ended episode never times out"
    judge.end_episode()  # idempotent


def test_restart_abandons_an_undecided_episode():
    judge = DockJudge(JudgeConfig(timeout_s=5.0))
    judge.start_episode(0.0)
    run(judge, 0.0, 4.0, front_range_m=1.0)
    judge.start_episode(4.0)
    decided = run(judge, 4.0, 10.0, front_range_m=1.0)
    assert [round(t, 6) for t, _ in decided] == [9.0]


def test_clock_jumping_back_restarts_the_timers():
    judge = DockJudge(JudgeConfig(timeout_s=5.0))
    judge.start_episode(100.0)
    run(judge, 100.0, 104.0, front_range_m=1.0)
    assert run(judge, 0.0, 4.9, front_range_m=1.0) == []
    assert run(judge, 4.9, 5.1, front_range_m=1.0)[0][1].reason == REASON_TIMEOUT


def test_status_matches_the_schema():
    judge = DockJudge()
    judge.update(12.5, None, None, None, None, None)
    status = judge.status()
    assert tuple(status) == STATUS_KEYS
    assert status == {
        "t": 12.5,
        "marker_visible": False,
        "marker_id": None,
        "bearing_deg": None,
        "marker_area_frac": None,
        "front_range_m": None,
        "speed_v": 0.0,
        "speed_w": 0.0,
        "in_band": False,
        "aligned": False,
        "stopped": False,
        "success": False,
        "success_hold_s": 0.0,
    }
    judge.update(12.6, MarkerObservation(7, -3.21234, 0.0912345), 0.2512345, 0.011, -0.02, "ok")
    status = json.loads(json.dumps(judge.status()))
    assert status["marker_visible"] and status["marker_id"] == 7
    assert status["bearing_deg"] == pytest.approx(-3.212) and status["front_range_m"] == pytest.approx(0.2512)
    assert status["marker_area_frac"] == pytest.approx(0.091234, abs=1e-6)
    assert (status["speed_v"], status["speed_w"]) == (0.011, -0.02)
    assert status["in_band"] and status["aligned"] and status["stopped"] and not status["success"]
    types = {k: type(v) for k, v in status.items()}
    assert types["t"] is float and types["success_hold_s"] is float and types["marker_id"] is int
    # stale odometry: the last speeds are still reported, but they no longer prove "stopped"
    judge.update(12.7, ALIGNED, 0.25, None, None, None)
    status = judge.status()
    assert (status["speed_v"], status["speed_w"], status["stopped"]) == (0.011, -0.02, False)


def test_invalid_config_is_rejected():
    for bad in (
        JudgeConfig(band_min=0.3, band_max=0.2),
        JudgeConfig(timeout_s=0.0),
        JudgeConfig(hold_s=-1.0),
        JudgeConfig(bearing_tol_deg=0.0),
    ):
        with pytest.raises(ValueError):
            DockJudge(bad)


def test_is_intervention():
    assert is_intervention("intervention") and is_intervention("intervention  pushed back by hand")
    assert not is_intervention("") and not is_intervention("end fail intervention")
    assert not is_intervention("interventions")


# ---------------------------------------------------------------- servo law

P = ServoParams()


def test_servo_turns_towards_the_marker():
    v, w = servo_command(15.0, 1.0, P)  # marker left of centre
    assert w > 0.0 and w == pytest.approx(P.k_w * math.radians(15.0))
    v, w = servo_command(-15.0, 1.0, P)
    assert w < 0.0
    assert servo_command(P.bearing_deadband_deg, 1.0, P)[1] == 0.0
    assert servo_command(-P.bearing_deadband_deg - 0.1, 1.0, P)[1] < 0.0
    assert servo_command(40.0, 1.0, P)[1] == pytest.approx(P.w_max)
    assert servo_command(-40.0, 1.0, P)[1] == pytest.approx(-P.w_max)


def test_servo_forward_speed():
    assert servo_command(0.0, 2.0, P) == (P.v_max, 0.0)
    v, _ = servo_command(0.0, 0.35, P)
    assert v == pytest.approx(P.k_v * 0.10)
    # turning before driving: speed falls with |bearing| and is zero beyond slow_bearing_deg
    v_half = servo_command(0.5 * P.slow_bearing_deg, 2.0, P)[0]
    assert v_half == pytest.approx(0.5 * P.v_max)
    assert servo_command(-P.slow_bearing_deg, 2.0, P)[0] == 0.0
    assert servo_command(35.0, 2.0, P)[0] == 0.0
    # never reverses, and does not move forward when the range is unknown
    assert servo_command(0.0, 0.10, P)[0] == 0.0
    assert servo_command(0.0, None, P)[0] == 0.0


def test_servo_stops_in_band_when_aligned():
    for r in (P.target_range + P.range_tol, P.target_range, 0.2, 0.16):
        for b in (0.0, P.bearing_deadband_deg, -P.bearing_deadband_deg):
            assert servo_command(b, r, P) == (0.0, 0.0)
    v, w = servo_command(8.0, 0.25, P)  # in band, not aligned: rotate in place
    assert v == 0.0 and w > 0.0
    assert servo_command(0.0, P.target_range + P.range_tol + 0.01, P)[0] > 0.0


def test_servo_searches_when_the_marker_is_not_visible():
    assert servo_command(None, 1.0, P) == (0.0, P.search_w)
    assert servo_command(None, None, P, search_sign=-1.0) == (0.0, -P.search_w)


def test_servo_params_validation():
    validate_servo_params(P)
    for bad in (ServoParams(k_v=0.0), ServoParams(v_max=-0.1), ServoParams(slow_bearing_deg=3.0)):
        with pytest.raises(ValueError):
            validate_servo_params(bad)


def test_range_from_marker_area_inverts_the_pinhole_model():
    hfov, w, h = 1.396, 640, 480
    f = 0.5 * w / math.tan(0.5 * hfov)
    for d_cam in (0.15, 0.217, 0.5, 1.5):
        side_px = 0.10 * f / d_cam
        area = side_px**2 / (w * h)
        assert range_from_marker_area(area, w, h, hfov) == pytest.approx(d_cam + 0.033)
    assert range_from_marker_area(0.05, 320, 240, hfov) == pytest.approx(
        range_from_marker_area(0.05, 640, 480, hfov)
    )
    with pytest.raises(ValueError):
        range_from_marker_area(0.0, w, h, hfov)
