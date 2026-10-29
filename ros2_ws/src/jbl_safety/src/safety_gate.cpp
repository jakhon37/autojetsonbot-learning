// Copyright 2026 Jakhongir Nodirov
// SPDX-License-Identifier: Apache-2.0

#include "jbl_safety/safety_gate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jbl_safety
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Beam angles are rebuilt from float32 angle_min/angle_increment, so a beam that lies exactly
// on the sector edge may land a few microradians outside it.
constexpr double kAngleTolerance = 1e-6;

// Ranges arrive as float32: 0.17f is 1.8e-9 m above 0.17. Without this a reading "at"
// stop_distance would leave a nanometre-per-second creep and report "slow".
constexpr double kRangeTolerance = 1e-6;

double wrap_angle(double angle)
{
  return std::remainder(angle, 2.0 * kPi);
}

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

bool positive_finite(double value)
{
  return std::isfinite(value) && value > 0.0;
}

std::string json_number(double value)
{
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out.precision(6);
  out << value;
  return out.str();
}

std::string json_number(const std::optional<double> & value)
{
  return value ? json_number(*value) : std::string("null");
}

}  // namespace

void validate(const SafetyGateParams & p)
{
  require(positive_finite(p.rate_hz), "rate_hz must be a positive number");
  require(std::isfinite(p.forward_angle), "forward_angle must be finite");
  require(
    positive_finite(p.sector_half_angle) && p.sector_half_angle <= kPi,
    "sector_half_angle must be in (0, pi]");
  require(
    std::isfinite(p.stop_distance) && p.stop_distance >= 0.0,
    "stop_distance must be a non-negative number");
  require(
    std::isfinite(p.slow_distance) && p.slow_distance > p.stop_distance,
    "slow_distance must be greater than stop_distance");
  require(p.min_invalid_beams >= 1, "min_invalid_beams must be at least 1");
  require(
    std::isfinite(p.footprint_half_width) && p.footprint_half_width >= 0.0,
    "footprint_half_width must be a non-negative number");
  require(positive_finite(p.cmd_timeout), "cmd_timeout must be a positive number");
  require(positive_finite(p.scan_timeout), "scan_timeout must be a positive number");
  require(positive_finite(p.max_linear), "max_linear must be a positive number");
  require(positive_finite(p.max_angular), "max_angular must be a positive number");
}

const char * to_string(GateState state)
{
  switch (state) {
    case GateState::kOk:
      return "ok";
    case GateState::kSlow:
      return "slow";
    case GateState::kStopObstacle:
      return "stop_obstacle";
    case GateState::kStopCmdTimeout:
      return "stop_cmd_timeout";
    case GateState::kStopScanTimeout:
      return "stop_scan_timeout";
  }
  return "unknown";
}

SafetyGate::SafetyGate(const SafetyGateParams & params)
: params_(params)
{
  validate(params_);
}

bool SafetyGate::ingest_scan(
  double angle_min, double angle_increment, double range_min, double range_max,
  const std::vector<float> & ranges, double time)
{
  if (ranges.empty() || !std::isfinite(angle_min) || !std::isfinite(angle_increment) ||
    angle_increment == 0.0)
  {
    return false;
  }
  const double lower = (std::isfinite(range_min) && range_min > 0.0) ? range_min : 0.0;
  const double upper =
    std::isfinite(range_max) ? range_max : std::numeric_limits<double>::infinity();
  if (upper <= lower) {
    return false;
  }

  // Per-beam angular distance to the forward direction handles any forward_angle, scans that
  // wrap through +-pi, and both rotation directions without special cases. At 360 beams and
  // 10 Hz this costs microseconds.
  std::size_t sector_beams = 0;
  std::size_t invalid_beams = 0;
  std::optional<double> sector_min;  // nearest valid sector range
  std::optional<double> corridor_min;  // nearest forward distance of a valid corridor return
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double angle = angle_min + static_cast<double>(i) * angle_increment;
    const double relative = wrap_angle(angle - params_.forward_angle);
    const double range = ranges[i];
    const bool valid = std::isfinite(range) && range > 0.0 && range >= lower;
    if (std::abs(relative) > params_.sector_half_angle + kAngleTolerance) {
      // Outside the sector: a valid return inside the robot's forward swept corridor still
      // counts, at its forward distance, so an obstacle ahead of a chassis corner is not lost.
      // Invalid readings out here are never counted (no-return beams are common).
      if (valid && range <= upper) {
        const double x = range * std::cos(relative);
        const double y = range * std::sin(relative);
        if (x > 0.0 && std::abs(y) <= params_.footprint_half_width) {
          corridor_min = corridor_min ? std::min(*corridor_min, x) : x;
        }
      }
      continue;
    }
    ++sector_beams;
    if (!valid) {
      ++invalid_beams;
    } else if (range <= upper) {
      sector_min = sector_min ? std::min(*sector_min, range) : range;
    }
    // A finite reading beyond range_max says nothing is within range: treat as clear.
  }
  if (sector_beams == 0) {
    return false;
  }

  std::optional<double> valid_min = sector_min;
  if (corridor_min) {
    valid_min = valid_min ? std::min(*valid_min, *corridor_min) : corridor_min;
  }
  if (valid_min && *valid_min < params_.slow_distance) {
    last_close_time_ = time;
  }
  // The RPLidar reports objects closer than its minimum range as 0, inf or < range_min, which
  // is indistinguishable from "no return" in a single scan. A close reading shortly before
  // disambiguates it. rplidar_ros (angle_compensate) also publishes every empty 1-degree bin as
  // +inf, so most scans hold scattered invalid beams: they count only when there are at least
  // min_invalid_beams of them and no valid sector reading lies beyond the edge of the dead
  // zone (otherwise the lidar is looking past whatever is invalid).
  const std::size_t needed =
    std::min(static_cast<std::size_t>(params_.min_invalid_beams), sector_beams);
  const double near_limit = std::max(params_.stop_distance, lower) + kDeadZoneMargin;
  const bool invalid_blocks = params_.treat_invalid_as_close && invalid_beams >= needed &&
    (!sector_min || *sector_min <= near_limit) &&
    last_close_time_ && time - *last_close_time_ <= kCloseMemory;
  if (invalid_blocks) {
    last_close_time_ = time;
    scan_front_min_ = 0.0;
  } else {
    scan_front_min_ = valid_min;
  }
  has_scan_ = true;
  scan_time_ = time;
  return true;
}

bool SafetyGate::ingest_command(double v, double w, double time)
{
  const bool finite = std::isfinite(v) && std::isfinite(w);
  cmd_v_ = finite ? v : 0.0;
  cmd_w_ = finite ? w : 0.0;
  cmd_time_ = time;
  has_cmd_ = true;
  return finite;
}

double SafetyGate::speed_scale(double front_distance) const
{
  if (front_distance <= params_.stop_distance + kRangeTolerance) {
    return 0.0;
  }
  if (front_distance >= params_.slow_distance - kRangeTolerance) {
    return 1.0;
  }
  return (front_distance - params_.stop_distance) /
         (params_.slow_distance - params_.stop_distance);
}

GateOutput SafetyGate::compute(double now) const
{
  GateOutput out;
  const bool scan_fresh = has_scan_ && now - scan_time_ <= params_.scan_timeout;
  const bool cmd_fresh = has_cmd_ && now - cmd_time_ <= params_.cmd_timeout;

  if (scan_fresh) {
    out.front_min = scan_front_min_;
    out.scale = out.front_min ? speed_scale(*out.front_min) : 1.0;
  } else {
    out.scale = 0.0;
  }

  if (!scan_fresh) {
    out.state = GateState::kStopScanTimeout;
  } else if (!cmd_fresh) {
    out.state = GateState::kStopCmdTimeout;
  } else if (out.scale <= 0.0) {
    out.state = GateState::kStopObstacle;
  } else if (out.scale < 1.0) {
    out.state = GateState::kSlow;
  } else {
    out.state = GateState::kOk;
  }

  if (cmd_fresh) {
    out.v = std::clamp(cmd_v_, -params_.max_linear, params_.max_linear);
    out.w = std::clamp(cmd_w_, -params_.max_angular, params_.max_angular);
    if (out.v > 0.0) {
      out.v *= out.scale;
    }
  }
  return out;
}

double percentile_inplace(std::vector<double> & values, double p)
{
  if (!(p >= 0.0 && p <= 100.0)) {
    throw std::invalid_argument("percentile must be in [0, 100]");
  }
  if (values.empty()) {
    return kNaN;
  }
  const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
  const auto lower_index = static_cast<std::size_t>(std::floor(rank));
  const double fraction = rank - static_cast<double>(lower_index);
  const auto lower_it = values.begin() + static_cast<std::ptrdiff_t>(lower_index);
  std::nth_element(values.begin(), lower_it, values.end());
  const double lower = *lower_it;
  if (fraction == 0.0 || lower_index + 1 >= values.size()) {
    return lower;
  }
  // After nth_element everything right of lower_it is >= lower, so the next rank is their min.
  const double upper = *std::min_element(lower_it + 1, values.end());
  return lower + fraction * (upper - lower);
}

StatsSummary summarize(std::vector<double> samples)
{
  StatsSummary summary;
  summary.count = samples.size();
  if (samples.empty()) {
    summary.p50 = summary.p99 = summary.max = kNaN;
    return summary;
  }
  summary.max = *std::max_element(samples.begin(), samples.end());
  summary.p50 = percentile_inplace(samples, 50.0);
  summary.p99 = percentile_inplace(samples, 99.0);
  return summary;
}

RollingStats::RollingStats(std::size_t capacity)
: capacity_(capacity)
{
  if (capacity_ == 0) {
    throw std::invalid_argument("RollingStats capacity must be positive");
  }
  window_.reserve(capacity_);
}

void RollingStats::add(double sample)
{
  if (!std::isfinite(sample)) {
    return;
  }
  ++total_;
  if (window_.size() < capacity_) {
    window_.push_back(sample);
    return;
  }
  window_[next_] = sample;
  next_ = (next_ + 1) % capacity_;
}

void RollingStats::clear()
{
  window_.clear();
  next_ = 0;
  total_ = 0;
}

double RollingStats::percentile(double p) const
{
  std::vector<double> copy = window_;
  return percentile_inplace(copy, p);
}

std::string format_status_json(const GateStatus & status)
{
  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << "{\"state\": \"" << to_string(status.output.state) << "\"";
  json << ", \"front_min_m\": " << json_number(status.output.front_min);
  json << ", \"scale\": " << json_number(status.output.scale);
  json << ", \"period_ms\": {\"p50\": " << json_number(status.period_ms.p50);
  json << ", \"p99\": " << json_number(status.period_ms.p99);
  json << ", \"max\": " << json_number(status.period_ms.max) << "}";
  json << ", \"missed_deadlines\": " << status.missed_deadlines;
  json << ", \"ticks\": " << status.ticks;
  json << ", \"cb_latency_us\": {\"p50\": " << json_number(status.cb_latency_us.p50);
  json << ", \"p99\": " << json_number(status.cb_latency_us.p99) << "}}";
  return json.str();
}

}  // namespace jbl_safety
