// Copyright 2026 Jakhongir Nodirov
// SPDX-License-Identifier: Apache-2.0
//
// safety_gate_node: the only publisher of /cmd_vel. Forwards /cmd_vel_in at a fixed rate after
// the lidar front-sector check, timeouts and clamps of jbl_safety::SafetyGate, and reports its
// own timing on /safety_gate/status.

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "jbl_safety/safety_gate.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/string.hpp"

namespace jbl_safety
{
namespace
{

using SteadyClock = std::chrono::steady_clock;

double seconds_since_epoch(SteadyClock::time_point t)
{
  return std::chrono::duration<double>(t.time_since_epoch()).count();
}

template<typename Rep, typename Period>
double to_ms(std::chrono::duration<Rep, Period> d)
{
  return std::chrono::duration<double, std::milli>(d).count();
}

template<typename Rep, typename Period>
double to_us(std::chrono::duration<Rep, Period> d)
{
  return std::chrono::duration<double, std::micro>(d).count();
}

constexpr double kMissedDeadlineFactor = 1.5;
constexpr std::chrono::milliseconds kStatusPeriod{500};
constexpr int kThrottleMs = 2000;
constexpr int kScanWarningMs = 5000;

}  // namespace

class SafetyGateNode : public rclcpp::Node
{
public:
  explicit SafetyGateNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("safety_gate", options),
    gate_(declare_gate_parameters()),
    nominal_period_ms_(1000.0 / gate_.params().rate_hz),
    throttle_clock_(RCL_STEADY_TIME)
  {
    last_output_ = gate_.compute(seconds_since_epoch(SteadyClock::now()));

    // Inputs and the control timer live in separate mutually exclusive groups so that a burst
    // of scans or commands never delays a control tick; the status timer has its own group so
    // that computing percentiles never does either.
    input_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    control_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    status_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions input_options;
    input_options.callback_group = input_group_;
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {on_scan(*msg);},
      input_options);
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel_in", rclcpp::QoS(10).reliable(),
      [this](geometry_msgs::msg::Twist::ConstSharedPtr msg) {on_command(*msg);},
      input_options);

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", rclcpp::QoS(10).reliable());
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "safety_gate/status", rclcpp::QoS(10).reliable());

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / gate_.params().rate_hz));
    control_timer_ = create_wall_timer(period, [this]() {on_tick();}, control_group_);
    status_timer_ = create_wall_timer(kStatusPeriod, [this]() {publish_status();}, status_group_);

    const SafetyGateParams & p = gate_.params();
    RCLCPP_INFO(
      get_logger(),
      "Gating %s -> %s at %.1f Hz: sector %.2f +- %.2f rad, stop %.3f m, slow %.3f m, "
      "invalid-as-close %s, timeouts cmd %.2f s / scan %.2f s, limits %.2f m/s %.2f rad/s",
      cmd_sub_->get_topic_name(), cmd_pub_->get_topic_name(), p.rate_hz, p.forward_angle,
      p.sector_half_angle, p.stop_distance, p.slow_distance,
      p.treat_invalid_as_close ? "on" : "off", p.cmd_timeout, p.scan_timeout, p.max_linear,
      p.max_angular);
  }

  /// Publishes a zero twist. Call only once no callback can run any more (executor stopped),
  /// otherwise a later tick could overwrite it.
  void publish_stop()
  {
    cmd_pub_->publish(geometry_msgs::msg::Twist());
  }

private:
  template<typename T>
  T declare_read_only(const std::string & name, const T & default_value, const char * description)
  {
    // Read-only: the gate's thresholds are part of the experiment protocol, so they are fixed
    // for the lifetime of the node and `ros2 param set` fails loudly instead of being ignored.
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = description;
    descriptor.read_only = true;
    return declare_parameter<T>(name, default_value, descriptor);
  }

  SafetyGateParams declare_gate_parameters()
  {
    SafetyGateParams p;
    p.rate_hz = declare_read_only("rate_hz", p.rate_hz, "output timer rate (Hz)");
    p.forward_angle = declare_read_only(
      "forward_angle", p.forward_angle, "scan angle (rad) that points forward");
    p.sector_half_angle = declare_read_only(
      "sector_half_angle", p.sector_half_angle, "half width (rad) of the obstacle sector");
    p.stop_distance = declare_read_only(
      "stop_distance", p.stop_distance, "range (m) at or below which forward motion is zeroed");
    p.slow_distance = declare_read_only(
      "slow_distance", p.slow_distance, "range (m) below which forward speed is scaled down");
    p.treat_invalid_as_close = declare_read_only(
      "treat_invalid_as_close", p.treat_invalid_as_close,
      "count 0/inf/nan/<range_min sector readings after a recent close reading as blocked");
    p.cmd_timeout = declare_read_only(
      "cmd_timeout", p.cmd_timeout, "seconds without a command before publishing zero");
    p.scan_timeout = declare_read_only(
      "scan_timeout", p.scan_timeout, "seconds without a scan before forward motion is blocked");
    p.max_linear = declare_read_only("max_linear", p.max_linear, "linear speed clamp (m/s)");
    p.max_angular = declare_read_only(
      "max_angular", p.max_angular, "angular speed clamp (rad/s)");
    return p;  // SafetyGate's constructor validates
  }

  void on_scan(const sensor_msgs::msg::LaserScan & scan)
  {
    // Receive time on the steady clock: timeouts must not depend on the lidar driver's stamps
    // or on sim time.
    const double now = seconds_since_epoch(SteadyClock::now());
    bool accepted = false;
    {
      std::lock_guard<std::mutex> lock(gate_mutex_);
      accepted = gate_.ingest_scan(
        scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max, scan.ranges, now);
    }
    if (!accepted) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), throttle_clock_, kThrottleMs,
        "Ignoring scan (frame '%s', %zu ranges, angle_min %.3f, increment %.5f): unusable "
        "geometry or no beam within %.2f rad of forward_angle %.2f",
        scan.header.frame_id.c_str(), scan.ranges.size(), scan.angle_min, scan.angle_increment,
        gate_.params().sector_half_angle, gate_.params().forward_angle);
    }
  }

  void on_command(const geometry_msgs::msg::Twist & cmd)
  {
    const double now = seconds_since_epoch(SteadyClock::now());
    bool finite = false;
    {
      std::lock_guard<std::mutex> lock(gate_mutex_);
      finite = gate_.ingest_command(cmd.linear.x, cmd.angular.z, now);
    }
    if (!finite) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), throttle_clock_, kThrottleMs,
        "Non-finite command on %s (v=%f, w=%f): stopping", cmd_sub_->get_topic_name(),
        cmd.linear.x, cmd.angular.z);
    }
  }

  void on_tick()
  {
    const SteadyClock::time_point start = SteadyClock::now();
    GateOutput out;
    {
      std::lock_guard<std::mutex> lock(gate_mutex_);
      out = gate_.compute(seconds_since_epoch(start));
    }
    geometry_msgs::msg::Twist twist;
    twist.linear.x = out.v;
    twist.angular.z = out.w;
    cmd_pub_->publish(twist);
    const SteadyClock::time_point end = SteadyClock::now();

    GateState previous_state;
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      if (last_tick_start_) {
        const double period_ms = to_ms(start - *last_tick_start_);
        period_ms_.add(period_ms);
        if (period_ms > kMissedDeadlineFactor * nominal_period_ms_) {
          ++missed_deadlines_;
        }
      }
      last_tick_start_ = start;
      cb_latency_us_.add(to_us(end - start));
      ++ticks_;
      previous_state = last_output_.state;
      last_output_ = out;
    }
    log_state(previous_state, out);
  }

  void log_state(GateState previous, const GateOutput & out)
  {
    // Throttled rather than on transition only, so a lidar that never comes up keeps being
    // reported (the gate starts in this state).
    if (out.state == GateState::kStopScanTimeout) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), throttle_clock_, kScanWarningMs,
        "No usable scan on %s for more than %.2f s: forward motion blocked",
        scan_sub_->get_topic_name(), gate_.params().scan_timeout);
    } else if (previous == GateState::kStopScanTimeout) {
      RCLCPP_INFO(get_logger(), "Scans resumed (%s)", to_string(out.state));
    } else if (out.state != previous) {
      RCLCPP_DEBUG(
        get_logger(), "%s -> %s (front_min %.3f m, scale %.2f)", to_string(previous),
        to_string(out.state), out.front_min.value_or(-1.0), out.scale);
    }
  }

  void publish_status()
  {
    GateStatus status;
    std::vector<double> periods;
    std::vector<double> latencies;
    {
      // Copy under the lock and sort outside it so the control tick never waits on the
      // percentile computation.
      std::lock_guard<std::mutex> lock(stats_mutex_);
      periods = period_ms_.window();
      latencies = cb_latency_us_.window();
      status.output = last_output_;
      status.missed_deadlines = missed_deadlines_;
      status.ticks = ticks_;
    }
    status.period_ms = summarize(std::move(periods));
    status.cb_latency_us = summarize(std::move(latencies));

    std_msgs::msg::String msg;
    msg.data = format_status_json(status);
    status_pub_->publish(msg);
  }

  // Guards gate_ (written by the input group, read by the control group).
  std::mutex gate_mutex_;
  SafetyGate gate_;

  // Guards everything the control tick reports to the status timer.
  std::mutex stats_mutex_;
  RollingStats period_ms_;
  RollingStats cb_latency_us_;
  std::optional<SteadyClock::time_point> last_tick_start_;
  std::uint64_t missed_deadlines_ = 0;
  std::uint64_t ticks_ = 0;
  GateOutput last_output_;

  const double nominal_period_ms_;
  // Steady so that throttling works without /clock; rclcpp::Clock::now() on a steady clock is
  // stateless, so the input and control groups may share it.
  rclcpp::Clock throttle_clock_;

  rclcpp::CallbackGroup::SharedPtr input_group_;
  rclcpp::CallbackGroup::SharedPtr control_group_;
  rclcpp::CallbackGroup::SharedPtr status_group_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

}  // namespace jbl_safety

int main(int argc, char ** argv)
{
  // rclcpp's own SIGINT/SIGTERM handler shuts the context down before anything can publish
  // (Foxy has no pre-shutdown hook), so the final zero twist would be dropped. Block both
  // signals in every thread instead (DDS and executor threads inherit this mask) and take them
  // synchronously below while the context is still valid.
  sigset_t termination_signals;
  sigemptyset(&termination_signals);
  sigaddset(&termination_signals, SIGINT);
  sigaddset(&termination_signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &termination_signals, nullptr);

  rclcpp::init(argc, argv);
  rclcpp::uninstall_signal_handlers();

  std::shared_ptr<jbl_safety::SafetyGateNode> node;
  try {
    node = std::make_shared<jbl_safety::SafetyGateNode>();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("safety_gate"), "Failed to start: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  std::atomic<bool> spin_failed{false};
  std::thread spinner([&executor, &spin_failed, &node]() {
      try {
        executor.spin();
      } catch (const std::exception & e) {
        RCLCPP_FATAL(node->get_logger(), "Executor stopped: %s", e.what());
        spin_failed = true;
        kill(getpid(), SIGTERM);  // wake main() so it still stops the robot
      }
    });

  int signal_number = 0;
  sigwait(&termination_signals, &signal_number);
  executor.cancel();
  spinner.join();

  // Repeat the stop so that a best-effort subscriber or a lost first sample still sees it, and
  // linger briefly so DDS can deliver before the participant is destroyed.
  for (int i = 0; i < 3; ++i) {
    node->publish_stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  RCLCPP_INFO(node->get_logger(), "Stopped (signal %d): published zero twist", signal_number);

  rclcpp::shutdown();
  return spin_failed ? 1 : 0;
}
