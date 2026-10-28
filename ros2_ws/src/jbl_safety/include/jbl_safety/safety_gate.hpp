// Copyright 2026 Jakhongir Nodirov
// SPDX-License-Identifier: Apache-2.0
//
// ROS-free core of the autoJetsonBot safety gate: lidar front-sector check, command timeouts,
// speed clamps, and bounded statistics for the status report. Times are seconds on any
// monotonic clock chosen by the caller.

#ifndef JBL_SAFETY__SAFETY_GATE_HPP_
#define JBL_SAFETY__SAFETY_GATE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace jbl_safety
{

/// Mirrors the "Safety gate" parameter table in docs/INTERFACES.md (same names and defaults).
struct SafetyGateParams
{
  double rate_hz = 50.0;
  double forward_angle = 0.0;
  double sector_half_angle = 0.35;
  double stop_distance = 0.17;
  double slow_distance = 0.45;
  bool treat_invalid_as_close = true;
  double cmd_timeout = 0.3;
  double scan_timeout = 0.5;
  double max_linear = 0.25;
  double max_angular = 1.5;
};

/// Throws std::invalid_argument naming the first parameter that is out of range.
void validate(const SafetyGateParams & params);

enum class GateState
{
  kOk,
  kSlow,
  kStopObstacle,
  kStopCmdTimeout,
  kStopScanTimeout,
};

/// Name used in the status JSON: "ok", "slow", "stop_obstacle", "stop_cmd_timeout",
/// "stop_scan_timeout".
const char * to_string(GateState state);

struct GateOutput
{
  double v = 0.0;
  double w = 0.0;
  GateState state = GateState::kStopScanTimeout;
  /// Front distance the decision was based on: the minimum valid sector range, 0.0 when invalid
  /// readings are counted as blocked, nullopt when the sector holds no usable reading or the
  /// scan is stale.
  std::optional<double> front_min;
  /// Factor applied to forward linear speed (1 = free, 0 = blocked).
  double scale = 0.0;
};

/// Decides what may reach the motors given the latest scan and command.
///
/// State precedence: stop_scan_timeout (sensor fault) > stop_cmd_timeout > stop_obstacle > slow
/// > ok. The obstacle states describe the front sector independently of the command, so a
/// stopped robot in front of a wall reports stop_obstacle. Only forward motion (v > 0) is ever
/// scaled or blocked; reverse and rotation always pass (after clamping) while a command is fresh.
class SafetyGate
{
public:
  /// How long a close sector reading (valid and below slow_distance) keeps invalid sector
  /// readings counted as blocked. A scan blocked this way refreshes the memory, so an obstacle
  /// that has moved into the lidar dead zone stays blocked for as long as it remains there.
  static constexpr double kCloseMemory = 0.5;

  explicit SafetyGate(const SafetyGateParams & params = SafetyGateParams());

  const SafetyGateParams & params() const {return params_;}

  /// Evaluates the front sector of a scan received at `time`. Returns false, leaving the
  /// previous scan in effect (so it eventually times out), when the scan geometry is unusable:
  /// non-finite or zero increment, no ranges, inconsistent range limits, or no beam inside the
  /// sector.
  bool ingest_scan(
    double angle_min, double angle_increment, double range_min, double range_max,
    const std::vector<float> & ranges, double time);

  /// Stores the requested motion. A non-finite component makes the stored command zero (the
  /// robot stops at once instead of coasting until cmd_timeout) and returns false.
  bool ingest_command(double v, double w, double time);

  GateOutput compute(double now) const;

private:
  double speed_scale(double front_distance) const;

  SafetyGateParams params_;

  bool has_scan_ = false;
  double scan_time_ = 0.0;
  std::optional<double> scan_front_min_;
  std::optional<double> last_close_time_;

  bool has_cmd_ = false;
  double cmd_time_ = 0.0;
  double cmd_v_ = 0.0;
  double cmd_w_ = 0.0;
};

/// p-th percentile (p in [0, 100]) with linear interpolation between closest ranks, the
/// default method of numpy.percentile. Reorders `values`. Returns NaN for an empty input.
double percentile_inplace(std::vector<double> & values, double p);

/// Percentiles and max of a sample set; NaN while there are no samples.
struct StatsSummary
{
  std::size_t count = 0;
  double p50 = std::numeric_limits<double>::quiet_NaN();
  double p99 = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
};

StatsSummary summarize(std::vector<double> samples);

/// Keeps the most recent `capacity` samples so that memory and the cost of a percentile query
/// stay bounded on a node that runs for hours.
class RollingStats
{
public:
  static constexpr std::size_t kDefaultCapacity = 10000;

  explicit RollingStats(std::size_t capacity = kDefaultCapacity);

  void add(double sample);
  void clear();

  std::size_t size() const {return window_.size();}
  std::size_t capacity() const {return capacity_;}
  /// Number of samples ever added, including those that have left the window.
  std::uint64_t total() const {return total_;}
  /// Samples currently in the window, in storage (not chronological) order.
  const std::vector<double> & window() const {return window_;}

  double percentile(double p) const;
  StatsSummary summary() const {return summarize(window_);}

private:
  std::size_t capacity_;
  std::vector<double> window_;
  std::size_t next_ = 0;
  std::uint64_t total_ = 0;
};

struct GateStatus
{
  GateOutput output;
  StatsSummary period_ms;
  StatsSummary cb_latency_us;
  std::uint64_t missed_deadlines = 0;
  std::uint64_t ticks = 0;
};

/// Serialises the /safety_gate/status payload defined in docs/INTERFACES.md. Non-finite and
/// missing values become JSON null.
std::string format_status_json(const GateStatus & status);

}  // namespace jbl_safety

#endif  // JBL_SAFETY__SAFETY_GATE_HPP_
