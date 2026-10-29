// Copyright 2026 Jakhongir Nodirov
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "jbl_safety/safety_gate.hpp"

namespace jbl_safety
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// A 360-beam RPLidar A1 style scan, clear to 3 m in every direction.
struct Scan
{
  double angle_min = -kPi;
  double angle_increment = 2.0 * kPi / 360.0;
  double range_min = 0.15;
  double range_max = 12.0;
  std::vector<float> ranges = std::vector<float>(360, 3.0f);

  // Sets the beam whose angle is closest to `angle`, whatever the scan's start and direction.
  Scan & at(double angle, float range)
  {
    std::size_t best = 0;
    double best_error = 1e9;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      const double beam = angle_min + static_cast<double>(i) * angle_increment;
      const double error = std::abs(std::remainder(beam - angle, 2.0 * kPi));
      if (error < best_error) {
        best_error = error;
        best = i;
      }
    }
    ranges[best] = range;
    return *this;
  }

  // Sets every beam within `half_width` of `centre`.
  Scan & span(double centre, double half_width, float range)
  {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      const double beam = angle_min + static_cast<double>(i) * angle_increment;
      if (std::abs(std::remainder(beam - centre, 2.0 * kPi)) <= half_width + 1e-9) {
        ranges[i] = range;
      }
    }
    return *this;
  }
};

bool feed(SafetyGate & gate, const Scan & scan, double time)
{
  return gate.ingest_scan(
    scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max, scan.ranges, time);
}

// Fresh scan and command at t=0, evaluated at t=0.05.
GateOutput run(
  const SafetyGateParams & params, const Scan & scan, double v = 0.2, double w = 0.5)
{
  SafetyGate gate(params);
  EXPECT_TRUE(feed(gate, scan, 0.0));
  gate.ingest_command(v, w, 0.0);
  return gate.compute(0.05);
}

bool forward_blocked(const SafetyGateParams & params, const Scan & scan)
{
  return run(params, scan).v == 0.0;
}

TEST(SafetyGate, ClearScanForwardsCommandUnchanged)
{
  const GateOutput out = run(SafetyGateParams(), Scan());
  EXPECT_DOUBLE_EQ(out.v, 0.2);
  EXPECT_DOUBLE_EQ(out.w, 0.5);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_DOUBLE_EQ(out.scale, 1.0);
  ASSERT_TRUE(out.front_min.has_value());
  EXPECT_FLOAT_EQ(*out.front_min, 3.0f);
}

TEST(SafetyGate, SlowZoneScalesForwardSpeedLinearly)
{
  const SafetyGateParams p;
  const double mid = 0.5 * (p.stop_distance + p.slow_distance);  // 0.31 m
  const GateOutput out = run(p, Scan().at(0.0, static_cast<float>(mid)));
  EXPECT_EQ(out.state, GateState::kSlow);
  EXPECT_NEAR(out.scale, 0.5, 1e-6);
  EXPECT_NEAR(out.v, 0.1, 1e-6);
  EXPECT_DOUBLE_EQ(out.w, 0.5);

  const double quarter = p.stop_distance + 0.25 * (p.slow_distance - p.stop_distance);
  EXPECT_NEAR(run(p, Scan().at(0.1, static_cast<float>(quarter))).scale, 0.25, 1e-6);
  EXPECT_EQ(run(p, Scan().at(0.0, 0.45f)).state, GateState::kOk);
}

TEST(SafetyGate, StopsForwardAtOrBelowStopDistanceButKeepsRotation)
{
  const GateOutput at_stop = run(SafetyGateParams(), Scan().at(0.0, 0.17f));
  EXPECT_EQ(at_stop.state, GateState::kStopObstacle);
  EXPECT_DOUBLE_EQ(at_stop.scale, 0.0);
  EXPECT_DOUBLE_EQ(at_stop.v, 0.0);
  EXPECT_DOUBLE_EQ(at_stop.w, 0.5);

  Scan near;
  near.range_min = 0.05;
  near.at(0.0, 0.12f);
  const GateOutput close = run(SafetyGateParams(), near);
  EXPECT_EQ(close.state, GateState::kStopObstacle);
  EXPECT_DOUBLE_EQ(close.v, 0.0);
  ASSERT_TRUE(close.front_min.has_value());
  EXPECT_NEAR(*close.front_min, 0.12, 1e-6);
}

TEST(SafetyGate, ReverseAndPureRotationAreNeverBlocked)
{
  Scan wall;
  wall.range_min = 0.05;
  wall.span(0.0, 0.35, 0.10f);

  const GateOutput reverse = run(SafetyGateParams(), wall, -0.1, 0.0);
  EXPECT_EQ(reverse.state, GateState::kStopObstacle);
  EXPECT_DOUBLE_EQ(reverse.v, -0.1);

  const GateOutput rotate = run(SafetyGateParams(), wall, 0.0, -1.2);
  EXPECT_DOUBLE_EQ(rotate.v, 0.0);
  EXPECT_DOUBLE_EQ(rotate.w, -1.2);

  const GateOutput reverse_arc = run(SafetyGateParams(), wall, -0.2, 0.7);
  EXPECT_DOUBLE_EQ(reverse_arc.v, -0.2);
  EXPECT_DOUBLE_EQ(reverse_arc.w, 0.7);
}

TEST(SafetyGate, OnlyBeamsInsideTheSectorCount)
{
  SafetyGateParams p;  // half angle 0.35 rad = 20.05 deg
  p.footprint_half_width = 0.0;  // the sector alone
  EXPECT_TRUE(forward_blocked(p, Scan().at(20.0 * kDeg, 0.16f)));
  EXPECT_TRUE(forward_blocked(p, Scan().at(-20.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(21.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(-21.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(kPi, 0.16f)));
}

TEST(SafetyGate, SectorWrapsAroundPlusMinusPi)
{
  for (const double forward : {kPi, -kPi}) {
    SafetyGateParams p;
    p.forward_angle = forward;
    p.footprint_half_width = 0.0;  // the sector alone
    SCOPED_TRACE(forward);
    // Scan starts at -pi: the sector spans the first and the last beams.
    EXPECT_TRUE(forward_blocked(p, Scan().at(-kPi, 0.16f)));
    EXPECT_TRUE(forward_blocked(p, Scan().at(-kPi + 15.0 * kDeg, 0.16f)));
    EXPECT_TRUE(forward_blocked(p, Scan().at(kPi - 15.0 * kDeg, 0.16f)));
    EXPECT_FALSE(forward_blocked(p, Scan().at(kPi - 25.0 * kDeg, 0.16f)));
    EXPECT_FALSE(forward_blocked(p, Scan().at(0.0, 0.16f)));
  }
}

TEST(SafetyGate, ForwardAngleAnywhereWithScansFromZeroToTwoPi)
{
  for (const double forward : {-kPi / 2.0, kPi / 2.0, 3.0, -3.0, 0.0}) {
    SafetyGateParams p;
    p.forward_angle = forward;
    p.footprint_half_width = 0.0;  // the sector alone
    SCOPED_TRACE(forward);
    Scan from_zero;
    from_zero.angle_min = 0.0;  // some drivers publish [0, 2pi)
    EXPECT_TRUE(forward_blocked(p, Scan(from_zero).at(forward, 0.16f)));
    EXPECT_TRUE(forward_blocked(p, Scan(from_zero).at(forward + 0.3, 0.16f)));
    EXPECT_FALSE(forward_blocked(p, Scan(from_zero).at(forward + 0.4, 0.16f)));
    EXPECT_FALSE(forward_blocked(p, Scan(from_zero).at(forward + kPi, 0.16f)));
  }
}

TEST(SafetyGate, ForwardCorridorCatchesObstaclesAheadOfAChassisCorner)
{
  const SafetyGateParams p;  // corridor +-0.15 m
  // Just outside the sector, still ahead of the chassis: counted at its forward distance.
  GateOutput out = run(p, Scan().at(21.0 * kDeg, 0.16f));
  EXPECT_EQ(out.state, GateState::kStopObstacle);
  ASSERT_TRUE(out.front_min.has_value());
  EXPECT_NEAR(*out.front_min, 0.16 * std::cos(21.0 * kDeg), 1e-6);
  // A box corner 0.15 m ahead and 0.12 m to the side of the lidar.
  const double corner = std::atan2(0.12, 0.15);
  EXPECT_TRUE(forward_blocked(p, Scan().at(corner, static_cast<float>(std::hypot(0.15, 0.12)))));
  EXPECT_TRUE(forward_blocked(p, Scan().at(-corner, static_cast<float>(std::hypot(0.15, 0.12)))));
  // Farther ahead it slows like an in-sector reading at the same forward distance.
  out = run(p, Scan().at(21.0 * kDeg, 0.3324f));
  EXPECT_EQ(out.state, GateState::kSlow);
  EXPECT_NEAR(out.scale, (0.3324 * std::cos(21.0 * kDeg) - 0.17) / (0.45 - 0.17), 1e-6);
  // Beside the corridor (|y| = 0.155 m) or behind the lidar: ignored.
  EXPECT_FALSE(forward_blocked(p, Scan().at(75.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(-75.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(100.0 * kDeg, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan().at(kPi, 0.16f)));
  // Invalid readings outside the sector never count, even after a close reading.
  SafetyGate gate(p);
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.30f), 0.0));
  ASSERT_TRUE(feed(gate, Scan().span(30.0 * kDeg, 5.0 * kDeg, 0.0f), 0.1));
  gate.ingest_command(0.2, 0.0, 0.1);
  EXPECT_EQ(gate.compute(0.1).state, GateState::kOk);
}

TEST(SafetyGate, NegativeAngleIncrement)
{
  Scan reversed;
  reversed.angle_min = kPi;
  reversed.angle_increment = -2.0 * kPi / 360.0;

  SafetyGateParams p;
  EXPECT_TRUE(forward_blocked(p, Scan(reversed).at(0.0, 0.16f)));
  EXPECT_TRUE(forward_blocked(p, Scan(reversed).at(-0.3, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan(reversed).at(kPi / 2.0, 0.16f)));

  p.forward_angle = kPi / 2.0;
  EXPECT_TRUE(forward_blocked(p, Scan(reversed).at(kPi / 2.0, 0.16f)));
  EXPECT_FALSE(forward_blocked(p, Scan(reversed).at(0.0, 0.16f)));

  p.forward_angle = kPi;  // the first beams of this scan
  EXPECT_TRUE(forward_blocked(p, Scan(reversed).at(kPi - 10.0 * kDeg, 0.16f)));
  EXPECT_TRUE(forward_blocked(p, Scan(reversed).at(-kPi + 10.0 * kDeg, 0.16f)));
}

TEST(SafetyGate, InvalidReadingsAfterCloseReadingBlock)
{
  for (const float invalid : {0.0f, kInf, kNaN, 0.10f, -1.0f}) {
    SCOPED_TRACE(invalid);
    SafetyGate gate;
    gate.ingest_command(0.2, 0.3, 0.0);
    ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.30f), 0.0));
    EXPECT_EQ(gate.compute(0.0).state, GateState::kSlow);

    // The obstacle moved into the dead zone: every sector beam is now invalid.
    ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, invalid), 0.1));
    gate.ingest_command(0.2, 0.3, 0.1);
    const GateOutput out = gate.compute(0.1);
    EXPECT_EQ(out.state, GateState::kStopObstacle);
    EXPECT_DOUBLE_EQ(out.v, 0.0);
    EXPECT_DOUBLE_EQ(out.w, 0.3);
    ASSERT_TRUE(out.front_min.has_value());
    EXPECT_DOUBLE_EQ(*out.front_min, 0.0);
  }
}

TEST(SafetyGate, InvalidBeamsNextToCloseReadingBlock)
{
  // Edges of a box still return valid ranges (just past the dead zone) while its centre is
  // inside it.
  SafetyGate gate;
  gate.ingest_command(0.2, 0.0, 0.0);
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.19f).span(0.0, 2.5 * kDeg, 0.0f), 0.0));
  EXPECT_EQ(gate.compute(0.0).state, GateState::kStopObstacle);
}

TEST(SafetyGate, FewerInvalidBeamsThanTheMinimumDoNotBlock)
{
  const auto state = [](const SafetyGateParams & p, const Scan & scan) {
      SafetyGate gate(p);
      gate.ingest_command(0.2, 0.0, 0.0);
      EXPECT_TRUE(feed(gate, scan, 0.0));
      return gate.compute(0.0).state;
    };
  const Scan close = Scan().span(0.0, 0.35, 0.19f);
  const SafetyGateParams p;  // min_invalid_beams 3
  EXPECT_EQ(state(p, Scan(close).at(0.0, kInf)), GateState::kSlow);
  EXPECT_EQ(state(p, Scan(close).at(0.0, kInf).at(10.0 * kDeg, 0.0f)), GateState::kSlow);
  EXPECT_EQ(
    state(p, Scan(close).at(0.0, kInf).at(10.0 * kDeg, 0.0f).at(-15.0 * kDeg, kNaN)),
    GateState::kStopObstacle);

  SafetyGateParams single = p;
  single.min_invalid_beams = 1;
  EXPECT_EQ(state(single, Scan(close).at(0.0, kInf)), GateState::kStopObstacle);

  // A sector narrower than min_invalid_beams beams: all of its beams invalid is enough.
  SafetyGateParams narrow = p;
  narrow.sector_half_angle = 0.01;  // only the beam at 0
  narrow.footprint_half_width = 0.0;
  SafetyGate gate(narrow);
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.30f), 0.0));
  ASSERT_TRUE(feed(gate, Scan().at(0.0, kInf), 0.1));
  gate.ingest_command(0.2, 0.0, 0.1);
  EXPECT_EQ(gate.compute(0.1).state, GateState::kStopObstacle);
}

TEST(SafetyGate, ScatteredEmptyBinsDoNotBlockOrLatch)
{
  // rplidar_ros with angle_compensate publishes every empty 1-degree bin as +inf: at 7 Hz
  // about every fourth bin. Box face 0.20 m wide, background at 3 m.
  const auto scan = [](double d) {
      Scan s;
      for (int k = -25; k <= 25; ++k) {
        const double a = k * kDeg;
        const double r =
          (k % 4 == 0) ? kInf : (std::abs(d * std::tan(a)) <= 0.10 ? d / std::cos(a) : 3.0);
        s.at(a, static_cast<float>(r));
      }
      return s;
    };
  SafetyGate gate;
  double t = 0.0;
  for (int cm = 44; cm >= 25; --cm, t += 0.14) {  // approach from 0.44 m to 0.25 m
    ASSERT_TRUE(feed(gate, scan(0.01 * cm), t));
    gate.ingest_command(0.15, 0.0, t);
    ASSERT_EQ(gate.compute(t).state, GateState::kSlow) << "d=" << 0.01 * cm;
  }
  for (int i = 0; i < 10; ++i, t += 0.14) {  // the box moved away to 1.5 m
    ASSERT_TRUE(feed(gate, scan(1.5), t));
    gate.ingest_command(0.15, 0.0, t);
    ASSERT_EQ(gate.compute(t).state, GateState::kOk) << "i=" << i;
  }
}

TEST(SafetyGate, InvalidReadingsWithoutRecentCloseReadingAreIgnored)
{
  SafetyGate gate;
  gate.ingest_command(0.2, 0.0, 0.0);
  ASSERT_TRUE(feed(gate, Scan(), 0.0));

  // Dropouts (no return) next to a far wall.
  ASSERT_TRUE(feed(gate, Scan().at(0.0, kInf).at(0.1, 0.0f), 0.1));
  GateOutput out = gate.compute(0.1);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_DOUBLE_EQ(out.v, 0.2);
  ASSERT_TRUE(out.front_min.has_value());
  EXPECT_FLOAT_EQ(*out.front_min, 3.0f);

  // Nothing usable in the sector at all: no information, not an obstacle.
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, kInf), 0.2));
  gate.ingest_command(0.2, 0.0, 0.2);
  out = gate.compute(0.2);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_FALSE(out.front_min.has_value());
  EXPECT_DOUBLE_EQ(out.scale, 1.0);
  EXPECT_DOUBLE_EQ(out.v, 0.2);
}

TEST(SafetyGate, CloseReadingMemoryExpiresAfterHalfASecond)
{
  SafetyGate gate;
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.30f), 0.0));
  // A far reading does not refresh the memory; at t=0.5 the close reading is just recent enough.
  ASSERT_TRUE(feed(gate, Scan(), 0.3));
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, kInf), SafetyGate::kCloseMemory));
  gate.ingest_command(0.2, 0.0, SafetyGate::kCloseMemory);
  EXPECT_EQ(gate.compute(SafetyGate::kCloseMemory).state, GateState::kStopObstacle);

  SafetyGate later;
  ASSERT_TRUE(feed(later, Scan().span(0.0, 0.35, 0.30f), 0.0));
  ASSERT_TRUE(feed(later, Scan(), 0.3));
  ASSERT_TRUE(feed(later, Scan().span(0.0, 0.35, kInf), 0.6));
  later.ingest_command(0.2, 0.0, 0.6);
  const GateOutput out = later.compute(0.6);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_FALSE(out.front_min.has_value());
  EXPECT_DOUBLE_EQ(out.v, 0.2);
}

TEST(SafetyGate, ObstacleStayingInDeadZoneStaysBlocked)
{
  SafetyGate gate;
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.16f), 0.0));
  for (int i = 1; i <= 30; ++i) {  // 3 s of 10 Hz scans with the box inside 0.15 m
    const double t = 0.1 * i;
    ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.0f), t));
    gate.ingest_command(0.2, 0.0, t);
    ASSERT_EQ(gate.compute(t).state, GateState::kStopObstacle) << "t=" << t;
  }
  // Backing away until the sector reads valid and far releases the gate.
  ASSERT_TRUE(feed(gate, Scan(), 3.1));
  gate.ingest_command(0.2, 0.0, 3.1);
  EXPECT_EQ(gate.compute(3.1).state, GateState::kOk);
}

TEST(SafetyGate, TreatInvalidAsCloseCanBeDisabled)
{
  SafetyGateParams p;
  p.treat_invalid_as_close = false;
  SafetyGate gate(p);
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.30f), 0.0));
  ASSERT_TRUE(feed(gate, Scan().span(0.0, 0.35, 0.0f), 0.1));
  gate.ingest_command(0.2, 0.0, 0.1);
  const GateOutput out = gate.compute(0.1);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_FALSE(out.front_min.has_value());
  EXPECT_DOUBLE_EQ(out.v, 0.2);
}

TEST(SafetyGate, ReadingsBeyondRangeMaxCountAsClear)
{
  Scan scan;
  scan.span(0.0, 0.35, 20.0f);
  const GateOutput out = run(SafetyGateParams(), scan);
  EXPECT_EQ(out.state, GateState::kOk);
  EXPECT_FALSE(out.front_min.has_value());
}

TEST(SafetyGate, ScanTimeoutBlocksForwardOnly)
{
  SafetyGate gate;
  ASSERT_TRUE(feed(gate, Scan(), 0.0));
  gate.ingest_command(0.2, 0.5, 0.45);
  EXPECT_EQ(gate.compute(0.5).state, GateState::kOk);  // exactly scan_timeout old: still fresh

  GateOutput out = gate.compute(0.55);
  EXPECT_EQ(out.state, GateState::kStopScanTimeout);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.5);
  EXPECT_DOUBLE_EQ(out.scale, 0.0);
  EXPECT_FALSE(out.front_min.has_value());

  gate.ingest_command(-0.15, -0.4, 0.55);
  out = gate.compute(0.6);
  EXPECT_DOUBLE_EQ(out.v, -0.15);
  EXPECT_DOUBLE_EQ(out.w, -0.4);

  SafetyGate never_scanned;
  never_scanned.ingest_command(0.2, 0.5, 0.0);
  out = never_scanned.compute(0.0);
  EXPECT_EQ(out.state, GateState::kStopScanTimeout);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.5);
}

TEST(SafetyGate, CommandTimeoutPublishesZero)
{
  SafetyGate gate;
  gate.ingest_command(0.2, 0.5, 0.0);
  ASSERT_TRUE(feed(gate, Scan(), 0.30));
  EXPECT_DOUBLE_EQ(gate.compute(0.30).v, 0.2);  // exactly cmd_timeout old: still fresh

  ASSERT_TRUE(feed(gate, Scan(), 0.31));
  const GateOutput out = gate.compute(0.31);
  EXPECT_EQ(out.state, GateState::kStopCmdTimeout);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.0);
  EXPECT_DOUBLE_EQ(out.scale, 1.0);  // the sector is still reported

  SafetyGate never_commanded;
  ASSERT_TRUE(feed(never_commanded, Scan(), 0.0));
  EXPECT_EQ(never_commanded.compute(0.0).state, GateState::kStopCmdTimeout);
  EXPECT_DOUBLE_EQ(never_commanded.compute(0.0).w, 0.0);
}

TEST(SafetyGate, ScanTimeoutTakesPrecedenceOverCommandTimeout)
{
  SafetyGate gate;
  const GateOutput out = gate.compute(0.0);
  EXPECT_EQ(out.state, GateState::kStopScanTimeout);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.0);
}

TEST(SafetyGate, ClampsToSpeedLimitsBeforeScaling)
{
  const SafetyGateParams p;
  GateOutput out = run(p, Scan(), 1.0, 5.0);
  EXPECT_DOUBLE_EQ(out.v, 0.25);
  EXPECT_DOUBLE_EQ(out.w, 1.5);

  out = run(p, Scan(), -1.0, -5.0);
  EXPECT_DOUBLE_EQ(out.v, -0.25);
  EXPECT_DOUBLE_EQ(out.w, -1.5);

  out = run(p, Scan().at(0.0, 0.31f), 1.0, 0.0);  // scale 0.5
  EXPECT_NEAR(out.v, 0.125, 1e-6);
}

TEST(SafetyGate, NonFiniteCommandStopsImmediately)
{
  SafetyGate gate;
  ASSERT_TRUE(feed(gate, Scan(), 0.0));
  EXPECT_TRUE(gate.ingest_command(0.2, 0.5, 0.0));
  EXPECT_FALSE(gate.ingest_command(std::nan(""), 0.5, 0.05));
  GateOutput out = gate.compute(0.05);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.0);

  EXPECT_FALSE(gate.ingest_command(0.1, std::numeric_limits<double>::infinity(), 0.06));
  out = gate.compute(0.06);
  EXPECT_DOUBLE_EQ(out.v, 0.0);
  EXPECT_DOUBLE_EQ(out.w, 0.0);
}

TEST(SafetyGate, UnusableScansAreRejectedAndTimeOut)
{
  SafetyGate gate;
  ASSERT_TRUE(feed(gate, Scan(), 0.0));
  gate.ingest_command(0.2, 0.0, 0.55);

  Scan zero_increment;
  zero_increment.angle_increment = 0.0;
  Scan nan_start;
  nan_start.angle_min = std::nan("");
  Scan empty;
  empty.ranges.clear();
  Scan bad_limits;
  bad_limits.range_max = 0.1;
  Scan rear_only;  // covers only [1.5, 2.5] rad, forward is 0
  rear_only.angle_min = 1.5;
  rear_only.ranges.assign(58, 3.0f);

  for (const Scan * scan : {&zero_increment, &nan_start, &empty, &bad_limits, &rear_only}) {
    EXPECT_FALSE(feed(gate, *scan, 0.4));
  }
  EXPECT_EQ(gate.compute(0.55).state, GateState::kStopScanTimeout);
}

TEST(SafetyGate, ValidateRejectsInconsistentParameters)
{
  EXPECT_NO_THROW(validate(SafetyGateParams()));
  const auto expect_invalid = [](auto mutate) {
      SafetyGateParams p;
      mutate(p);
      EXPECT_THROW(validate(p), std::invalid_argument);
      EXPECT_THROW(SafetyGate{p}, std::invalid_argument);
    };
  expect_invalid([](SafetyGateParams & p) {p.rate_hz = 0.0;});
  expect_invalid([](SafetyGateParams & p) {p.forward_angle = std::nan("");});
  expect_invalid([](SafetyGateParams & p) {p.sector_half_angle = 0.0;});
  expect_invalid([](SafetyGateParams & p) {p.sector_half_angle = 4.0;});
  expect_invalid([](SafetyGateParams & p) {p.stop_distance = -0.1;});
  expect_invalid([](SafetyGateParams & p) {p.slow_distance = p.stop_distance;});
  expect_invalid([](SafetyGateParams & p) {p.min_invalid_beams = 0;});
  expect_invalid([](SafetyGateParams & p) {p.footprint_half_width = -0.01;});
  expect_invalid([](SafetyGateParams & p) {p.footprint_half_width = std::nan("");});
  expect_invalid([](SafetyGateParams & p) {p.cmd_timeout = 0.0;});
  expect_invalid([](SafetyGateParams & p) {p.scan_timeout = -1.0;});
  expect_invalid([](SafetyGateParams & p) {p.max_linear = 0.0;});
  expect_invalid(
    [](SafetyGateParams & p) {p.max_angular = std::numeric_limits<double>::infinity();});
}

TEST(SafetyGate, StateNamesMatchStatusSchema)
{
  EXPECT_STREQ(to_string(GateState::kOk), "ok");
  EXPECT_STREQ(to_string(GateState::kSlow), "slow");
  EXPECT_STREQ(to_string(GateState::kStopObstacle), "stop_obstacle");
  EXPECT_STREQ(to_string(GateState::kStopCmdTimeout), "stop_cmd_timeout");
  EXPECT_STREQ(to_string(GateState::kStopScanTimeout), "stop_scan_timeout");
}

TEST(Percentile, MatchesNumpyLinearInterpolation)
{
  std::vector<double> values;
  for (int i = 100; i >= 1; --i) {
    values.push_back(i);
  }
  // numpy.percentile(np.arange(1, 101), [0, 50, 99, 100]) -> [1, 50.5, 99.01, 100]
  EXPECT_DOUBLE_EQ(percentile_inplace(values, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(percentile_inplace(values, 50.0), 50.5);
  EXPECT_NEAR(percentile_inplace(values, 99.0), 99.01, 1e-9);
  EXPECT_DOUBLE_EQ(percentile_inplace(values, 100.0), 100.0);

  std::vector<double> single{4.2};
  EXPECT_DOUBLE_EQ(percentile_inplace(single, 99.0), 4.2);
  std::vector<double> empty;
  EXPECT_TRUE(std::isnan(percentile_inplace(empty, 50.0)));
  EXPECT_THROW(percentile_inplace(single, 101.0), std::invalid_argument);
  EXPECT_THROW(percentile_inplace(single, std::nan("")), std::invalid_argument);
}

TEST(RollingStats, KeepsOnlyTheMostRecentSamples)
{
  RollingStats stats(10);
  for (int i = 1; i <= 25; ++i) {
    stats.add(i);
  }
  EXPECT_EQ(stats.size(), 10u);
  EXPECT_EQ(stats.total(), 25u);
  const StatsSummary summary = stats.summary();  // window is 16..25
  EXPECT_EQ(summary.count, 10u);
  EXPECT_DOUBLE_EQ(summary.p50, 20.5);
  EXPECT_NEAR(summary.p99, 24.91, 1e-9);
  EXPECT_DOUBLE_EQ(summary.max, 25.0);
  EXPECT_DOUBLE_EQ(stats.percentile(0.0), 16.0);

  stats.add(std::nan(""));  // ignored
  EXPECT_EQ(stats.total(), 25u);

  stats.clear();
  EXPECT_EQ(stats.size(), 0u);
  EXPECT_TRUE(std::isnan(stats.summary().p50));
  EXPECT_THROW(RollingStats(0), std::invalid_argument);
}

TEST(RollingStats, DefaultWindowIsTenThousandSamples)
{
  RollingStats stats;
  EXPECT_EQ(stats.capacity(), 10000u);
  // A single 1000 ms outlier among 20 ms periods: visible in max, not in p99 once it is <1%.
  stats.add(1000.0);
  for (int i = 0; i < 12000; ++i) {
    stats.add(20.0 + 0.001 * (i % 100));
  }
  const StatsSummary summary = stats.summary();
  EXPECT_EQ(summary.count, 10000u);
  EXPECT_LT(summary.max, 20.1);  // the outlier has left the window
  EXPECT_NEAR(summary.p50, 20.05, 0.001);
}

TEST(StatusJson, MatchesInterfaceSchema)
{
  GateStatus status;
  status.output.state = GateState::kSlow;
  status.output.front_min = 0.31;
  status.output.scale = 0.5;
  status.period_ms = summarize({19.0, 20.0, 21.0});
  status.cb_latency_us = summarize({40.0, 60.0});
  status.missed_deadlines = 2;
  status.ticks = 1234;
  EXPECT_EQ(
    format_status_json(status),
    "{\"state\": \"slow\", \"front_min_m\": 0.31, \"scale\": 0.5, "
    "\"period_ms\": {\"p50\": 20, \"p99\": 20.98, \"max\": 21}, \"missed_deadlines\": 2, "
    "\"ticks\": 1234, \"cb_latency_us\": {\"p50\": 50, \"p99\": 59.8}}");
}

TEST(StatusJson, MissingValuesAreNull)
{
  GateStatus status;  // no scan, no samples yet
  EXPECT_EQ(
    format_status_json(status),
    "{\"state\": \"stop_scan_timeout\", \"front_min_m\": null, \"scale\": 0, "
    "\"period_ms\": {\"p50\": null, \"p99\": null, \"max\": null}, \"missed_deadlines\": 0, "
    "\"ticks\": 0, \"cb_latency_us\": {\"p50\": null, \"p99\": null}}");
}

}  // namespace
}  // namespace jbl_safety
