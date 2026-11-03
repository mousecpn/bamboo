// Joint Impedance control

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include <Eigen/Dense>

#include "bamboo_messages.h"
#include "controllers/joint_impedance_controller.h"
#include "interpolators/joint_interpolator.h"
#include "safety/contact_math.h"

// Global flag for signal handling
std::atomic<bool> global_shutdown{false};

// Global exception handling
std::mutex exception_mutex;
std::exception_ptr thread_exception_ptr = nullptr;

void setThreadException(std::exception_ptr ex) {
  std::lock_guard<std::mutex> lock(exception_mutex);
  if (!thread_exception_ptr) {
    thread_exception_ptr = ex;
    global_shutdown = true;
  }
}

std::exception_ptr getThreadException() {
  std::lock_guard<std::mutex> lock(exception_mutex);
  return thread_exception_ptr;
}

namespace {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using SteadyClock = std::chrono::steady_clock;

static_assert(std::atomic<double>::is_always_lock_free,
              "Streaming control requires lock-free double atomics");

constexpr std::array<double, 7> kFr3JointLower = {
    -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
constexpr std::array<double, 7> kFr3JointUpper = {
    2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
constexpr std::array<double, 7> kPandaJointLower = {
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
constexpr std::array<double, 7> kPandaJointUpper = {
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
constexpr double kJointMargin = 0.08;
constexpr double kHardMaxStreamVelocity = 0.50;
constexpr double kHardMaxStreamAcceleration = 2.0;
constexpr int kHardMaxWatchdogMs = 500;
constexpr double kMaxTrackingError = 0.03;
// Hand guiding.
constexpr double kHardMaxGuideVelocity = 1.0;
constexpr double kGuideVelocityFraction = 0.5; // of Franka's position-dependent
                                               // joint velocity limits
constexpr double kGuideStopTimeout = 2.0;
constexpr std::array<double, 7> kGuideTorqueLimits = {60.0, 60.0, 60.0, 60.0,
                                                      30.0, 15.0, 15.0};
using Mat67 = Eigen::Matrix<double, 6, 7>;

int64_t steadyNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             SteadyClock::now().time_since_epoch())
      .count();
}

// Why Franka would refuse to start guiding in this mode, or empty when it can.
// Franka accepts a motion only while idle; libfranka's own rejection names the
// mode but not what the operator has to do about it.
std::string GuideStartProblem(franka::RobotMode mode) {
  switch (mode) {
  case franka::RobotMode::kIdle:
    return "";
  case franka::RobotMode::kUserStopped:
    return "Franka is not in Execution mode (robot mode \"User stopped\"): "
           "white lights mean Desk is in Programming mode, otherwise the user "
           "stop is pressed. Switch Desk to Execution and release the user "
           "stop, as for teleoperation, then start guiding again";
  case franka::RobotMode::kGuiding:
    return "Franka's Pilot guiding buttons are held. Let go of them, then "
           "start guiding again";
  case franka::RobotMode::kReflex:
    return "Franka is still in a reflex after error recovery. Check Desk for "
           "the error, then start guiding again";
  case franka::RobotMode::kAutomaticErrorRecovery:
    return "Franka is recovering from an error. Wait for it to finish, then "
           "start guiding again";
  case franka::RobotMode::kMove:
    return "Franka is already executing a motion. Stop it, then start "
           "guiding again";
  case franka::RobotMode::kOther:
    break;
  }
  return "Franka is not ready to move (robot mode \"Other\"). In Desk, "
         "unlock the joints and activate FCI, then start guiding again";
}

// Bamboo's collision thresholds, as main() sets them at startup. Lowering the
// Cartesian force ones makes Franka itself stop the arm at a smaller force.
constexpr double kDefaultCollisionThreshold = 100.0;
// guard_trip value for a stream that yielded because a client asked.
constexpr int kYieldRequested = 4;

// Contact limits for streaming: set while no loop runs, fixed for the life of
// each stream, and inert (the stream behaves exactly as before) until set.
struct ContactLimits {
  bool enabled = false;
  bool shadow = false;    // report the push cap without applying it
  double max_force = 0.0; // N the stream may push with; 0: no cap
  Eigen::Vector3d contact_point = Eigen::Vector3d::Zero(); // EE frame
  double max_point_speed = 0.0; // m/s of the contact point; 0: no limit
  bamboo::safety::ForceGuardParams guard; // off while guard.force is 0
  double yield_time = 0.2;      // s of damping before a yielded stream ends
  double collision_force = 0.0; // N for Franka's own reflex; 0: Bamboo's
};

} // namespace

// Signal handler for graceful shutdown
void signalHandler(int signal) {
  if (signal == SIGINT || signal == SIGTERM || signal == SIGHUP) {
    global_shutdown = true;
  }
}

// Server implementation
class BambooControlServer {
private:
  franka::Robot *robot_;
  franka::Model *model_;
  franka::Gripper *gripper_;
  bamboo::controllers::JointImpedanceController *controller_;
  bamboo::interpolators::JointInterpolator *interpolator_;

  std::atomic<bool> control_running_{false};
  std::atomic<bool> joint_limit_hit_{false};
  const std::array<double, 7> joint_lower_;
  const std::array<double, 7> joint_upper_;

  // Persistent streaming control state. All callback-shared values are atomic
  // so the 1 kHz real-time loop never waits on the ZMQ thread.
  std::array<std::atomic<double>, 7> stream_target_velocity_;
  std::array<std::atomic<double>, 7> state_q_;
  std::array<std::atomic<double>, 7> state_dq_;
  std::array<std::atomic<double>, 7> state_tau_;
  std::array<std::atomic<double>, 16> state_pose_;
  std::atomic<double> state_time_{0.0};
  std::atomic<int64_t> stream_last_update_ns_{0};
  std::atomic<int64_t> stream_stop_request_ns_{0};
  std::atomic<bool> stream_running_{false};
  std::atomic<bool> stream_ready_{false};
  std::atomic<bool> stream_stop_requested_{false};
  std::atomic<bool> stream_failed_{false};
  std::thread stream_thread_;
  std::mutex stream_error_mutex_;
  std::string stream_error_message_;
  int stream_watchdog_ms_{250};
  double stream_max_velocity_{0.35};
  double stream_max_acceleration_{1.5};

  // Franka's external wrench/torque estimates and the configured payload,
  // cached alongside the rest of the state.
  std::array<std::atomic<double>, 6> state_wrench_;
  std::array<std::atomic<double>, 7> state_tau_ext_;
  std::atomic<double> state_m_ee_{0.0};
  std::atomic<double> state_m_load_{0.0};
  std::atomic<double> state_m_total_{0.0};
  std::array<std::atomic<double>, 3> state_com_ee_;
  std::array<std::atomic<double>, 3> state_com_load_;
  std::array<std::atomic<double>, 3> state_com_total_;

  // Persistent hand-guiding loop, same ownership pattern as streaming: the
  // parameters are written before the thread starts, and everything the ZMQ
  // thread changes afterwards is atomic.
  std::atomic<bool> guide_running_{false};
  std::atomic<bool> guide_ready_{false};
  std::atomic<bool> guide_stop_requested_{false};
  std::atomic<bool> guide_failed_{false};
  std::thread guide_thread_;
  std::mutex guide_error_mutex_;
  std::string guide_error_message_;
  bamboo::safety::GuideParams guide_params_;
  Eigen::Vector3d guide_contact_point_ = Eigen::Vector3d::Zero(); // EE frame
  int guide_watchdog_ms_{250};
  double guide_max_velocity_{0.35};
  std::atomic<double> guide_level_{0.0};
  std::atomic<bool> guide_level_valid_{true};
  std::atomic<bool> guide_has_direction_{false};
  std::array<std::atomic<double>, 3> guide_direction_; // EE frame, unit
  std::atomic<uint64_t> guide_last_seq_{0};
  std::atomic<int64_t> guide_update_ns_{0};
  std::atomic<uint64_t> guide_resume_requests_{0};
  std::atomic<int> guide_phase_{0};
  std::atomic<int> guide_reason_{0};
  std::atomic<double> guide_filtered_level_{0.0};
  std::atomic<double> guide_lead_{0.0};
  std::atomic<bool> guide_ratchet_{false};

  // Contact limits: contact_limits_ is written only by the request thread
  // while no loop runs; the stream thread copies it when it starts.
  ContactLimits contact_limits_;
  bool collision_lowered_ = false;
  std::atomic<double> contact_scale_{1.0};
  std::atomic<double> contact_push_{0.0};
  std::atomic<int> guard_trip_{0};
  std::atomic<bool> stream_yield_requested_{false};
  std::mutex yield_reason_mutex_;
  std::string yield_reason_;

  // Control parameters
  const int traj_rate_ = 500; // Hz
  const double max_time = 1.0;
  const bool log_err_ = true;

  Eigen::Matrix<double, 7, 1> q_current_;
  Eigen::Matrix<double, 7, 1> q_goal_;

  // For acceleration computation via finite differencing
  Eigen::Matrix<double, 7, 1> velocity_cmd_prev_;
  Eigen::Matrix<double, 7, 1> a_cmd_latest_;

  // Low-pass filter frequency for acceleration
  const double diff_low_pass_freq_ = 30.0; // Hz

public:
  BambooControlServer(franka::Robot *robot, franka::Model *model,
                      franka::Gripper *gripper,
                      bamboo::controllers::JointImpedanceController *controller,
                      bamboo::interpolators::JointInterpolator *interpolator,
                      const std::array<double, 7> &joint_lower,
                      const std::array<double, 7> &joint_upper)
      : robot_(robot), model_(model), gripper_(gripper),
        controller_(controller), interpolator_(interpolator),
        joint_lower_(joint_lower), joint_upper_(joint_upper) {

    // Get initial robot state
    franka::RobotState init_state = robot_->readOnce();
    cacheRobotState(init_state);
    q_current_ = Eigen::VectorXd::Map(init_state.q.data(), 7);
    q_goal_ = q_current_;

    // Initialize velocity and acceleration tracking
    velocity_cmd_prev_.setZero();
    a_cmd_latest_.setZero();
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }
    for (auto &value : guide_direction_) {
      value.store(0.0);
    }

    std::cout << "Initial joint positions: " << q_current_.transpose()
              << std::endl;
  }

  ~BambooControlServer() {
    try {
      StopStreaming();
    } catch (const std::exception &e) {
      std::cerr << "[STREAM] Error during shutdown: " << e.what() << std::endl;
    }
    try {
      StopGuide();
    } catch (const std::exception &e) {
      std::cerr << "[GUIDE] Error during shutdown: " << e.what() << std::endl;
    }
  }

  bamboo_msgs::RobotState GetRobotState() {
    try {
      // readOnce cannot run concurrently with robot.control(). During a stream,
      // return the lock-free state cache populated by the control callback.
      if (!stream_running_.load() && !guide_running_.load()) {
        cacheRobotState(robot_->readOnce());
      }

      bamboo_msgs::RobotState state_msg;

      // Add joint positions, velocities, and torques
      state_msg.q.resize(7);
      state_msg.dq.resize(7);
      state_msg.tau_J.resize(7);
      for (size_t i = 0; i < 7; ++i) {
        state_msg.q[i] = state_q_[i].load();
        state_msg.dq[i] = state_dq_[i].load();
        state_msg.tau_J[i] = state_tau_[i].load();
      }

      // Add end-effector pose (4x4 transformation matrix: O_T_EE)
      state_msg.O_T_EE.resize(16);
      for (size_t i = 0; i < 16; ++i) {
        state_msg.O_T_EE[i] = state_pose_[i].load();
      }

      // Add timing information
      state_msg.time_sec = state_time_.load();

      state_msg.O_F_ext_hat_K.resize(6);
      for (size_t i = 0; i < 6; ++i) {
        state_msg.O_F_ext_hat_K[i] = state_wrench_[i].load();
      }
      state_msg.tau_ext_hat_filtered.resize(7);
      for (size_t i = 0; i < 7; ++i) {
        state_msg.tau_ext_hat_filtered[i] = state_tau_ext_[i].load();
      }
      state_msg.loop_mode = guide_running_.load()     ? 3
                            : stream_running_.load()  ? 2
                            : control_running_.load() ? 1
                                                      : 0;
      state_msg.guide_phase = guide_phase_.load();
      state_msg.guide_reason = guide_reason_.load();
      state_msg.guide_level = guide_filtered_level_.load();
      state_msg.guide_lead = guide_lead_.load();
      state_msg.guide_ratchet = guide_ratchet_.load();
      state_msg.contact_scale = contact_scale_.load();
      state_msg.contact_push = contact_push_.load();
      state_msg.guard_trip = guard_trip_.load();

      return state_msg;
    } catch (const franka::Exception &e) {
      throw std::runtime_error(std::string("Franka exception: ") + e.what());
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Exception: ") + e.what());
    }
  }

  void StartStreaming(int watchdog_ms, double max_velocity,
                      double max_acceleration) {
    if (stream_running_.load()) {
      // Reconnecting clients always inherit a stopped stream.
      SetStreamVelocity(std::vector<double>(7, 0.0));
      return;
    }
    if (control_running_.load()) {
      throw std::runtime_error("Another control loop is already running");
    }
    if (stream_thread_.joinable()) {
      stream_thread_.join();
    }

    stream_watchdog_ms_ = std::clamp(watchdog_ms, 50, kHardMaxWatchdogMs);
    stream_max_velocity_ =
        std::clamp(max_velocity, 0.01, kHardMaxStreamVelocity);
    stream_max_acceleration_ =
        std::clamp(max_acceleration, 0.05, kHardMaxStreamAcceleration);

    const franka::RobotState initial_state = robot_->readOnce();
    cacheRobotState(initial_state);
    q_current_ = Eigen::Map<const Vector7d>(initial_state.q.data());
    q_goal_ = q_current_;
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
    stream_stop_request_ns_.store(0);
    stream_stop_requested_.store(false);
    stream_failed_.store(false);
    stream_ready_.store(false);
    setStreamError("");
    stream_yield_requested_.store(false);
    guard_trip_.store(0);
    contact_scale_.store(1.0);
    contact_push_.store(0.0);
    if (contact_limits_.enabled && contact_limits_.collision_force > 0.0 &&
        !collision_lowered_) {
      // Guide mode restores the default thresholds; put the lower ones back.
      applyCollisionForce(contact_limits_.collision_force);
      collision_lowered_ = true;
    }
    control_running_.store(true);
    stream_running_.store(true);
    stream_thread_ = std::thread(&BambooControlServer::streamControlLoop, this);

    const auto deadline = SteadyClock::now() + std::chrono::seconds(2);
    while (!stream_ready_.load() && stream_running_.load() &&
           SteadyClock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!stream_ready_.load()) {
      stream_stop_requested_.store(true);
      if (stream_thread_.joinable()) {
        stream_thread_.join();
      }
      const std::string message = getStreamError();
      throw std::runtime_error(message.empty() ? "Stream failed to start"
                                               : message);
    }
    std::cout << "[STREAM] Started (watchdog=" << stream_watchdog_ms_
              << " ms, max_velocity=" << stream_max_velocity_
              << " rad/s, max_acceleration=" << stream_max_acceleration_
              << " rad/s^2)" << std::endl;
    if (contact_limits_.enabled) {
      std::cout << "[STREAM] " << describeContactLimits(contact_limits_)
                << std::endl;
    }
  }

  void SetStreamVelocity(const std::vector<double> &velocity) {
    if (!stream_running_.load() || !stream_ready_.load()) {
      const std::string error = stream_failed_.load() ? getStreamError() : "";
      throw std::runtime_error(error.empty()
                                   ? "Streaming control is not running"
                                   : "Streaming control stopped: " + error);
    }
    if (velocity.size() != 7) {
      throw std::runtime_error("Velocity must contain seven values");
    }
    for (size_t i = 0; i < velocity.size(); ++i) {
      if (!std::isfinite(velocity[i])) {
        throw std::runtime_error("Velocity values must be finite");
      }
      if (std::abs(velocity[i]) > stream_max_velocity_ + 1e-9) {
        throw std::runtime_error("Velocity exceeds negotiated stream limit");
      }
    }
    for (size_t i = 0; i < velocity.size(); ++i) {
      stream_target_velocity_[i].store(velocity[i]);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
  }

  void StopStreaming() {
    if (!stream_running_.load() && !stream_thread_.joinable()) {
      return;
    }
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
    stream_stop_request_ns_.store(steadyNowNanoseconds());
    stream_stop_requested_.store(true);
    if (stream_thread_.joinable()) {
      stream_thread_.join();
    }
    if (stream_failed_.load()) {
      throw std::runtime_error(getStreamError());
    }
    std::cout << "[STREAM] Stopped" << std::endl;
  }

  ContactLimits SetContactLimits(ContactLimits limits) {
    if (control_running_.load()) {
      throw std::runtime_error(
          "Stop the active control loop before setting contact limits");
    }
    const auto within = [](double value, double low, double high) {
      return std::isfinite(value) && value >= low && value <= high;
    };
    if (!within(limits.max_force, 0.0, 100.0)) {
      throw std::runtime_error("max_force_n must be within 0 to 100 N");
    }
    if (!limits.contact_point.allFinite() ||
        limits.contact_point.norm() > 0.5) {
      throw std::runtime_error(
          "contact_point must be within 0.5 m of the end effector");
    }
    if (!within(limits.max_point_speed, 0.0, 1.0)) {
      throw std::runtime_error("max_point_speed must be within 0 to 1 m/s");
    }
    if (!within(limits.guard.force, 0.0, 100.0)) {
      throw std::runtime_error("guard_force_n must be within 0 to 100 N");
    }
    if (!within(limits.guard.impulse, 0.001, 1.0)) {
      throw std::runtime_error(
          "guard_impulse_ns must be within 0.001 to 1 N s");
    }
    if (!within(limits.guard.hard_force, 0.0, 150.0) ||
        (limits.guard.hard_force > 0.0 &&
         limits.guard.hard_force < limits.guard.force)) {
      throw std::runtime_error("guard_hard_force_n must be 0, or between "
                               "guard_force_n and 150 N");
    }
    if (!within(limits.yield_time, 0.05, 1.0)) {
      throw std::runtime_error("yield_time must be within 0.05 to 1 s");
    }
    if (!(limits.collision_force == 0.0 ||
          within(limits.collision_force, 5.0, 100.0))) {
      throw std::runtime_error(
          "collision_force_n must be 0 or within 5 to 100 N");
    }
    if (limits.collision_force > 0.0) {
      applyCollisionForce(limits.collision_force);
      collision_lowered_ = true;
    } else if (collision_lowered_) {
      applyCollisionForce(0.0);
      collision_lowered_ = false;
    }
    limits.enabled = true;
    contact_limits_ = limits;
    std::cout << "[CONTACT] " << describeContactLimits(contact_limits_)
              << std::endl;
    return contact_limits_;
  }

  void ClearContactLimits() {
    if (control_running_.load()) {
      throw std::runtime_error(
          "Stop the active control loop before clearing contact limits");
    }
    if (collision_lowered_) {
      applyCollisionForce(0.0);
      collision_lowered_ = false;
    }
    contact_limits_ = ContactLimits();
    std::cout << "[CONTACT] Limits cleared" << std::endl;
  }

  ContactLimits GetContactLimits() const { return contact_limits_; }

  // Stop pushing and end the stream, e.g. when the client's gel guard trips.
  void YieldStream(const std::string &reason) {
    if (!stream_running_.load() || !stream_ready_.load()) {
      throw std::runtime_error("Streaming control is not running");
    }
    {
      std::lock_guard<std::mutex> lock(yield_reason_mutex_);
      yield_reason_ = reason;
    }
    stream_yield_requested_.store(true);
  }

  static std::string describeContactLimits(const ContactLimits &limits) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << "Contact limits: push cap "
         << (limits.max_force > 0.0 ? std::to_string(limits.max_force) + " N"
                                    : std::string("off"))
         << (limits.shadow ? " (shadow: reported, not applied)" : "")
         << ", point speed "
         << (limits.max_point_speed > 0.0
                 ? std::to_string(limits.max_point_speed) + " m/s"
                 : std::string("off"))
         << ", guard "
         << (limits.guard.force > 0.0
                 ? std::to_string(limits.guard.force) + " N"
                 : std::string("off"))
         << ", Franka reflex "
         << (limits.collision_force > 0.0
                 ? std::to_string(limits.collision_force) + " N"
                 : std::string("default"));
    return text.str();
  }

  struct GuideOptions {
    int watchdog_ms = 250;
    double max_joint_velocity = 0.35;
    std::vector<double> joint_damping; // empty: default
    std::vector<double> contact_point; // EE frame; empty: flange
    double wall_damping = -1.0;        // negative: default
    double idle_hold_s = -1.0;         // negative: default, 0: off
    // Refuse to start when Franka's external force estimate exceeds this:
    // an uncompensated payload (or a hand already on the arm) would make the
    // floating arm sink. 0 disables the check.
    double max_start_force = 5.0;
  };

  void StartGuide(const GuideOptions &options) {
    if (guide_running_.load()) {
      throw std::runtime_error("Guide mode is already running");
    }
    if (control_running_.load()) {
      throw std::runtime_error("Another control loop is already running");
    }
    if (guide_thread_.joinable()) {
      guide_thread_.join();
    }

    bamboo::safety::GuideParams params;
    if (!options.joint_damping.empty()) {
      if (options.joint_damping.size() != 7) {
        throw std::runtime_error("joint_damping must contain seven values");
      }
      for (int i = 0; i < 7; ++i) {
        const double value = options.joint_damping[i];
        // Keep free-mode damping well inside the hold damping.
        if (!std::isfinite(value) || value < 0.0 ||
            value > 0.5 * params.hold_damping[i]) {
          throw std::runtime_error("joint_damping value out of range");
        }
        params.damping[i] = value;
      }
    }
    if (options.wall_damping >= 0.0) {
      if (!std::isfinite(options.wall_damping) ||
          options.wall_damping > 150.0) {
        throw std::runtime_error("wall_damping must be within 0 to 150 N s/m");
      }
      params.wall_damping = options.wall_damping;
    }
    Eigen::Vector3d contact = Eigen::Vector3d::Zero();
    if (!options.contact_point.empty()) {
      if (options.contact_point.size() != 3) {
        throw std::runtime_error("contact_point must contain three values");
      }
      contact =
          Eigen::Vector3d(options.contact_point[0], options.contact_point[1],
                          options.contact_point[2]);
      if (!contact.allFinite() || contact.norm() > 0.5) {
        throw std::runtime_error(
            "contact_point must be within 0.5 m of the end effector");
      }
    }
    if (!std::isfinite(options.max_joint_velocity)) {
      throw std::runtime_error("max_joint_velocity must be finite");
    }
    if (options.idle_hold_s >= 0.0) {
      if (!std::isfinite(options.idle_hold_s) || options.idle_hold_s > 10.0) {
        throw std::runtime_error("idle_hold_s must be within 0 to 10 s");
      }
      params.idle_time = options.idle_hold_s;
    }
    if (!std::isfinite(options.max_start_force) ||
        options.max_start_force < 0.0 || options.max_start_force > 30.0) {
      throw std::runtime_error("max_start_force must be within 0 to 30 N");
    }

    guide_params_ = params;
    guide_contact_point_ = contact;
    guide_watchdog_ms_ =
        std::clamp(options.watchdog_ms, 50, kHardMaxWatchdogMs);
    guide_max_velocity_ =
        std::clamp(options.max_joint_velocity, 0.05, kHardMaxGuideVelocity);
    guide_level_.store(0.0);
    guide_level_valid_.store(true);
    guide_has_direction_.store(false);
    guide_last_seq_.store(0);
    guide_phase_.store(0);
    guide_reason_.store(0);
    guide_filtered_level_.store(0.0);
    guide_lead_.store(0.0);
    guide_ratchet_.store(false);
    // Starting counts as the first update, so the watchdog runs from here.
    guide_update_ns_.store(steadyNowNanoseconds());
    guide_stop_requested_.store(false);
    guide_failed_.store(false);
    guide_ready_.store(false);
    {
      std::lock_guard<std::mutex> lock(guide_error_mutex_);
      guide_error_message_.clear();
    }
    franka::RobotState start_state = robot_->readOnce();
    if (start_state.robot_mode == franka::RobotMode::kReflex) {
      std::cout << "[GUIDE] Clearing a Franka reflex before guiding"
                << std::endl;
      robot_->automaticErrorRecovery();
      start_state = robot_->readOnce();
    }
    cacheRobotState(start_state);
    const std::string mode_problem = GuideStartProblem(start_state.robot_mode);
    if (!mode_problem.empty()) {
      throw std::runtime_error(mode_problem);
    }
    if (collision_lowered_) {
      // A guiding hand is an external force too: lowered thresholds would
      // make Franka stop the arm under it.
      applyCollisionForce(0.0);
      collision_lowered_ = false;
      std::cout << "[GUIDE] Restored Bamboo's collision thresholds for guiding"
                << std::endl;
    }
    if (bamboo::safety::WrenchEstimateValid(start_state.O_F_ext_hat_K)) {
      const double force = Eigen::Vector3d(start_state.O_F_ext_hat_K[0],
                                           start_state.O_F_ext_hat_K[1],
                                           start_state.O_F_ext_hat_K[2])
                               .norm();
      std::cout << "[GUIDE] External force estimate at start: " << force << " N"
                << std::endl;
      if (options.max_start_force > 0.0 && force > options.max_start_force) {
        throw std::runtime_error(
            "External force estimate is " + std::to_string(force) +
            " N before guiding starts (limit " +
            std::to_string(options.max_start_force) +
            " N). Let go of the arm, or set the payload with set_load if "
            "the gripper's weight is not compensated");
      }
    } else if (options.max_start_force > 0.0) {
      std::cout << "[GUIDE] External force estimate unavailable at start "
                   "(near a singularity); start check skipped"
                << std::endl;
    }
    control_running_.store(true);
    guide_running_.store(true);
    guide_thread_ = std::thread(&BambooControlServer::guideControlLoop, this);

    const auto deadline = SteadyClock::now() + std::chrono::seconds(2);
    while (!guide_ready_.load() && guide_running_.load() &&
           SteadyClock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!guide_ready_.load()) {
      guide_stop_requested_.store(true);
      if (guide_thread_.joinable()) {
        guide_thread_.join();
      }
      const std::string error = getGuideError();
      throw std::runtime_error(error.empty() ? "Guide mode did not start"
                                             : error);
    }
    std::cout << "[GUIDE] Started (watchdog=" << guide_watchdog_ms_
              << " ms, max_velocity=" << guide_max_velocity_ << " rad/s)"
              << std::endl;
  }

  struct GuideStatus {
    bool stale = false;
    int phase = 0;
    int reason = 0;
    double level = 0.0;
  };

  GuideStatus GuideUpdate(double level, bool valid,
                          const std::vector<double> &direction, uint64_t seq) {
    requireGuideRunning();
    GuideStatus status;
    // A repeated or older sequence number is not news: it must not keep the
    // watchdog alive, or a sender stuck on one gel frame would hide a stall.
    if (seq <= guide_last_seq_.load()) {
      status.stale = true;
    } else {
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      bool has_direction = false;
      if (direction.size() == 3) {
        d = Eigen::Vector3d(direction[0], direction[1], direction[2]);
        has_direction = d.allFinite() && d.norm() > 1e-6;
        if (has_direction) {
          d.normalize();
        }
      }
      for (int i = 0; i < 3; ++i) {
        guide_direction_[i].store(has_direction ? d[i] : 0.0);
      }
      guide_has_direction_.store(has_direction);
      guide_level_.store(level);
      guide_level_valid_.store(valid);
      guide_last_seq_.store(seq);
      guide_update_ns_.store(steadyNowNanoseconds());
    }
    status.phase = guide_phase_.load();
    status.reason = guide_reason_.load();
    status.level = guide_filtered_level_.load();
    return status;
  }

  GuideStatus GuideResume() {
    requireGuideRunning();
    guide_resume_requests_.fetch_add(1);
    GuideStatus status;
    status.phase = guide_phase_.load();
    status.reason = guide_reason_.load();
    status.level = guide_filtered_level_.load();
    return status;
  }

  void StopGuide() {
    if (!guide_running_.load() && !guide_thread_.joinable()) {
      return;
    }
    guide_stop_requested_.store(true);
    if (guide_thread_.joinable()) {
      guide_thread_.join();
    }
    if (guide_failed_.load()) {
      throw std::runtime_error(getGuideError());
    }
    std::cout << "[GUIDE] Stopped" << std::endl;
  }

  void SetLoad(double mass, const std::vector<double> &com,
               const std::vector<double> &inertia) {
    if (control_running_.load()) {
      throw std::runtime_error(
          "Stop the active control loop before setting the load");
    }
    if (!std::isfinite(mass) || mass < 0.0 || mass > 3.0) {
      throw std::runtime_error("load mass must be within 0 to 3 kg");
    }
    if (com.size() != 3) {
      throw std::runtime_error("load center of mass must contain three values");
    }
    std::array<double, 3> com_array{};
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(com[i]) || std::abs(com[i]) > 0.5) {
        throw std::runtime_error(
            "load center of mass must be within 0.5 m of the flange");
      }
      com_array[i] = com[i];
    }
    // A small point-mass inertia unless the caller knows better.
    std::array<double, 9> inertia_array{1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3};
    if (!inertia.empty()) {
      if (inertia.size() != 9) {
        throw std::runtime_error("load inertia must contain nine values");
      }
      for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(inertia[i])) {
          throw std::runtime_error("load inertia must be finite");
        }
        inertia_array[i] = inertia[i];
      }
    }
    robot_->setLoad(mass, com_array, inertia_array);
    cacheRobotState(robot_->readOnce());
    std::cout << "[LOAD] Set " << mass << " kg at (" << com_array[0] << ", "
              << com_array[1] << ", " << com_array[2] << ") m" << std::endl;
  }

  // Clears a reflex (e.g. a velocity or torque limit hit while guiding) so a
  // new loop can start without restarting the node.
  void RecoverErrors() {
    if (control_running_.load()) {
      throw std::runtime_error(
          "Stop the active control loop before recovering errors");
    }
    robot_->automaticErrorRecovery();
    cacheRobotState(robot_->readOnce());
    std::cout << "[SERVER] Error recovery done" << std::endl;
  }

  struct Payload {
    double m_ee, m_load, m_total;
    std::array<double, 3> com_ee, com_load, com_total;
  };

  Payload GetPayload() {
    if (!control_running_.load()) {
      cacheRobotState(robot_->readOnce());
    }
    Payload payload{};
    payload.m_ee = state_m_ee_.load();
    payload.m_load = state_m_load_.load();
    payload.m_total = state_m_total_.load();
    for (int i = 0; i < 3; ++i) {
      payload.com_ee[i] = state_com_ee_[i].load();
      payload.com_load[i] = state_com_load_[i].load();
      payload.com_total[i] = state_com_total_[i].load();
    }
    return payload;
  }

  bool ExecuteJointImpedanceTrajectory(
      const bamboo_msgs::TrajectoryRequest &request) {
    std::cout << "[SERVER] Received trajectory with "
              << request.waypoints.size() << " waypoints" << std::endl;

    try {
      // Check if already running
      if (control_running_.load()) {
        throw std::runtime_error("Control loop already running");
      }

      if (request.waypoints.empty()) {
        throw std::runtime_error("Empty trajectory");
      }

      // Parse and prepare all waypoints
      std::vector<Eigen::Matrix<double, 7, 1>> trajectory_goals;
      std::vector<Eigen::Matrix<double, 7, 1>> trajectory_velocities;
      std::vector<double> trajectory_durations;
      std::vector<std::optional<std::array<double, 7>>> trajectory_kp;
      std::vector<std::optional<std::array<double, 7>>> trajectory_kd;

      for (size_t i = 0; i < request.waypoints.size(); ++i) {
        const bamboo_msgs::TimedWaypoint &waypoint = request.waypoints[i];

        // Validate goal has 7 values
        if (waypoint.q_goal.size() != 7) {
          throw std::runtime_error("Joint configuration must have 7 values");
        }

        // Get waypoint duration
        double waypoint_duration;
        if (waypoint.duration > 0) {
          waypoint_duration = waypoint.duration;
        } else if (request.default_duration > 0) {
          waypoint_duration = request.default_duration;
        } else {
          waypoint_duration = 1.0; // Fallback default
        }

        // Get waypoint velocity
        Eigen::Matrix<double, 7, 1> waypoint_velocity =
            Eigen::Matrix<double, 7, 1>::Zero();
        if (waypoint.velocity.size() == 7) {
          // Use waypoint-specific velocity if provided
          for (int j = 0; j < 7; ++j) {
            waypoint_velocity(j) = waypoint.velocity[j];
          }
        } else if (request.default_velocity.size() == 7) {
          // Use default velocity if provided
          for (int j = 0; j < 7; ++j) {
            waypoint_velocity(j) = request.default_velocity[j];
          }
        }
        // If neither is provided, waypoint_velocity remains zero

        // Get waypoint kp (optional)
        std::optional<std::array<double, 7>> waypoint_kp = std::nullopt;
        if (waypoint.kp.size() == 7) {
          std::array<double, 7> kp_array;
          for (size_t j = 0; j < 7; ++j) {
            kp_array[j] = waypoint.kp[j];
          }
          waypoint_kp = kp_array;
        }

        // Get waypoint kd (optional)
        std::optional<std::array<double, 7>> waypoint_kd = std::nullopt;
        if (waypoint.kd.size() == 7) {
          std::array<double, 7> kd_array;
          for (size_t j = 0; j < 7; ++j) {
            kd_array[j] = waypoint.kd[j];
          }
          waypoint_kd = kd_array;
        }

        // Check for termination
        if (global_shutdown) {
          std::cout << "[SERVER] Termination requested" << std::endl;
          break;
        }

        // Store goal position
        Eigen::Matrix<double, 7, 1> goal;
        for (int j = 0; j < 7; ++j) {
          goal(j) = waypoint.q_goal[j];
        }

        trajectory_goals.push_back(goal);
        trajectory_velocities.push_back(waypoint_velocity);
        trajectory_durations.push_back(waypoint_duration);
        trajectory_kp.push_back(waypoint_kp);
        trajectory_kd.push_back(waypoint_kd);
      }

      // Execute entire trajectory in single control call
      bool success =
          executeTrajectory(trajectory_goals, trajectory_velocities,
                            trajectory_durations, trajectory_kp, trajectory_kd);
      if (!success) {
        throw std::runtime_error("Trajectory execution failed");
      }

      std::cout << "[SERVER] Trajectory completed successfully" << std::endl;
      return true;

    } catch (const franka::ControlException &e) {
      std::cerr << "[SERVER] Control exception: " << e.what() << std::endl;
      throw;
    } catch (const std::exception &e) {
      std::cerr << "[SERVER] Exception: " << e.what() << std::endl;
      throw;
    }
  }

  bool OpenGripper(double width, double speed) {
    std::cout << "[SERVER] Opening gripper to width: " << width
              << "m, speed: " << speed << "m/s" << std::endl;
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }
      return gripper_->move(width, speed);
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception during gripper open: " << e.what()
                << std::endl;
      throw;
    }
  }

  bool CloseGripper(double speed, double force) {
    std::cout << "[SERVER] Closing gripper: speed=" << speed
              << "m/s, force=" << force << "N" << std::endl;
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }
      // epsilon is set such that the grasp succeeds if the gripper_width is
      // b/w (0.0 - 0.08, 0.0 + 0.08). this way we don't throw when
      // grasping large objects
      return gripper_->grasp(0.0, speed, force, 0.08, 0.08);
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception during gripper close: "
                << e.what() << std::endl;
      throw;
    }
  }

  std::map<std::string, double> GetGripperState() {
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }

      franka::GripperState state = gripper_->readOnce();
      std::map<std::string, double> result;
      result["width"] = state.width;
      result["max_width"] = state.max_width;
      result["is_grasped"] = state.is_grasped ? 1.0 : 0.0;
      // open_gripper and close_gripper are blocking, so the gripper is never
      // mid-motion when queried.
      result["is_moving"] = 0.0;
      result["temperature"] = state.temperature;

      return result;
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception reading gripper state: "
                << e.what() << std::endl;
      throw;
    }
  }

private:
  void cacheRobotState(const franka::RobotState &state) {
    for (size_t i = 0; i < 7; ++i) {
      state_q_[i].store(state.q[i], std::memory_order_relaxed);
      state_dq_[i].store(state.dq[i], std::memory_order_relaxed);
      state_tau_[i].store(state.tau_J[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < 16; ++i) {
      state_pose_[i].store(state.O_T_EE[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < 6; ++i) {
      state_wrench_[i].store(state.O_F_ext_hat_K[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < 7; ++i) {
      state_tau_ext_[i].store(state.tau_ext_hat_filtered[i],
                              std::memory_order_relaxed);
    }
    state_m_ee_.store(state.m_ee, std::memory_order_relaxed);
    state_m_load_.store(state.m_load, std::memory_order_relaxed);
    state_m_total_.store(state.m_total, std::memory_order_relaxed);
    for (size_t i = 0; i < 3; ++i) {
      state_com_ee_[i].store(state.F_x_Cee[i], std::memory_order_relaxed);
      state_com_load_[i].store(state.F_x_Cload[i], std::memory_order_relaxed);
      state_com_total_[i].store(state.F_x_Ctotal[i], std::memory_order_relaxed);
    }
    state_time_.store(state.time.toSec(), std::memory_order_relaxed);
  }

  std::string describeYield(int reason, double force,
                            const ContactLimits &limits) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1);
    switch (reason) {
    case static_cast<int>(bamboo::safety::GuardTrip::kSustained):
      text << "Contact guard: Franka's external force estimate stayed above "
           << limits.guard.force << " N (" << force << " N)";
      break;
    case static_cast<int>(bamboo::safety::GuardTrip::kHard):
      text << "Contact guard: Franka's external force estimate reached "
           << force << " N, over the hard limit";
      break;
    case static_cast<int>(bamboo::safety::GuardTrip::kInvalid):
      text << "Contact guard: Franka's external force estimate is unavailable "
              "(near a singularity)";
      break;
    default: {
      std::lock_guard<std::mutex> lock(yield_reason_mutex_);
      text << "Yielded on request"
           << (yield_reason_.empty() ? "" : ": " + yield_reason_);
    }
    }
    text << "; the arm went soft and streaming stopped";
    return text.str();
  }

  void setStreamError(const std::string &message) {
    std::lock_guard<std::mutex> lock(stream_error_mutex_);
    stream_error_message_ = message;
  }

  std::string getStreamError() {
    std::lock_guard<std::mutex> lock(stream_error_mutex_);
    return stream_error_message_;
  }

  void requireGuideRunning() {
    if (!guide_running_.load() || !guide_ready_.load()) {
      const std::string error = getGuideError();
      throw std::runtime_error(error.empty() ? "Guide mode is not running"
                                             : "Guide mode stopped: " + error);
    }
  }

  // Franka's reflex thresholds: the Cartesian force ones at `force` (0 for
  // Bamboo's default), everything else at Bamboo's default.
  void applyCollisionForce(double force) {
    const double f = force > 0.0 ? force : kDefaultCollisionThreshold;
    std::array<double, 7> joints;
    joints.fill(kDefaultCollisionThreshold);
    const std::array<double, 6> cartesian = {f,
                                             f,
                                             f,
                                             kDefaultCollisionThreshold,
                                             kDefaultCollisionThreshold,
                                             kDefaultCollisionThreshold};
    robot_->setCollisionBehavior(joints, joints, cartesian, cartesian);
  }

  void setGuideError(const std::string &message) {
    std::lock_guard<std::mutex> lock(guide_error_mutex_);
    guide_error_message_ = message;
  }

  std::string getGuideError() {
    std::lock_guard<std::mutex> lock(guide_error_mutex_);
    return guide_error_message_;
  }

  void guideControlLoop() {
    try {
      bamboo::safety::GuideState state;
      uint64_t seen_resume = guide_resume_requests_.load();
      int64_t stop_started_ns = 0;
      bamboo::safety::Vec7 lower, upper;
      for (int i = 0; i < 7; ++i) {
        lower[i] = joint_lower_[i];
        upper[i] = joint_upper_[i];
      }
      bool started = false;

      auto control_callback = [&](const franka::RobotState &robot_state,
                                  franka::Duration period) -> franka::Torques {
        cacheRobotState(robot_state);
        if (!started) {
          // Franka has accepted the motion: StartGuide may report success.
          started = true;
          guide_ready_.store(true);
        }
        bamboo::safety::GuideInputs in;
        in.q = Eigen::Map<const Vector7d>(robot_state.q.data());
        in.dq = Eigen::Map<const Vector7d>(robot_state.dq.data());
        in.lower = lower;
        in.upper = upper;
        in.dt = std::clamp(period.toSec(), 0.0001, 0.01);
        bamboo::safety::GovernorBand(
            franka::computeUpperLimitsJointVelocity(robot_state.q),
            franka::computeLowerLimitsJointVelocity(robot_state.q),
            guide_max_velocity_, kGuideVelocityFraction, &in.velocity_low,
            &in.velocity_high);

        const std::array<double, 42> jacobian =
            model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
        const Mat67 J = Eigen::Map<const Mat67>(jacobian.data());
        const Eigen::Matrix4d pose =
            Eigen::Map<const Eigen::Matrix4d>(robot_state.O_T_EE.data());
        const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
        in.J_lin =
            bamboo::safety::PointJacobian(J, rotation * guide_contact_point_);
        in.J_ang = J.bottomRows<3>();
        in.contact_position =
            pose.topRightCorner<3, 1>() + rotation * guide_contact_point_;

        const int64_t now = steadyNowNanoseconds();
        in.update_fresh = now - guide_update_ns_.load() <=
                          static_cast<int64_t>(guide_watchdog_ms_) * 1000000;
        in.level = guide_level_.load();
        in.level_valid = guide_level_valid_.load();
        in.has_direction = guide_has_direction_.load();
        if (in.has_direction) {
          const Eigen::Vector3d direction_ee(guide_direction_[0].load(),
                                             guide_direction_[1].load(),
                                             guide_direction_[2].load());
          in.direction = (rotation * direction_ee).normalized();
        }
        const uint64_t resume = guide_resume_requests_.load();
        in.resume_requested = resume != seen_resume;
        seen_resume = resume;
        in.stop_requested =
            guide_stop_requested_.load() || global_shutdown.load();
        if (in.stop_requested && stop_started_ns == 0) {
          stop_started_ns = now;
        }

        const bamboo::safety::Vec7 tau =
            bamboo::safety::GuideStep(guide_params_, in, &state);
        guide_phase_.store(static_cast<int>(state.phase));
        guide_reason_.store(static_cast<int>(state.reason));
        guide_filtered_level_.store(state.level);
        guide_lead_.store(state.lead);
        guide_ratchet_.store(state.ratchet_engaged);

        std::array<double, 7> torques{};
        for (int i = 0; i < 7; ++i) {
          const double value = std::isfinite(tau[i]) ? tau[i] : 0.0;
          torques[i] =
              std::clamp(value, -kGuideTorqueLimits[i], kGuideTorqueLimits[i]);
        }
        torques = franka::limitRate(franka::kMaxTorqueRate, torques,
                                    robot_state.tau_J_d);

        if (in.stop_requested) {
          const bool still = in.dq.cwiseAbs().maxCoeff() < 0.01;
          const double stopping_for =
              static_cast<double>(now - stop_started_ns) / 1e9;
          if ((state.phase == bamboo::safety::GuidePhase::kHold && still) ||
              stopping_for > kGuideStopTimeout) {
            return franka::MotionFinished(franka::Torques(torques));
          }
        }
        return franka::Torques(torques);
      };

      robot_->control(control_callback);
    } catch (const franka::Exception &e) {
      guide_failed_.store(true);
      setGuideError(std::string("Franka guide control error: ") + e.what());
    } catch (const std::exception &e) {
      guide_failed_.store(true);
      setGuideError(std::string("Guide control error: ") + e.what());
    }
    guide_ready_.store(false);
    guide_running_.store(false);
    control_running_.store(false);
  }

  void streamControlLoop() {
    try {
      Vector7d desired_position;
      Vector7d desired_velocity = Vector7d::Zero();
      for (size_t i = 0; i < 7; ++i) {
        desired_position[static_cast<int>(i)] = state_q_[i].load();
      }
      controller_->RestoreDefaultGains();
      // Contact limits are fixed for the life of this stream.
      const ContactLimits limits = contact_limits_;
      const Vector7d kp = controller_->Kp();
      const Vector7d kd = controller_->Kd();
      const Vector7d yield_damping = bamboo::safety::GuideParams().hold_damping;
      double guard_impulse = 0.0;
      bool yielding = false;
      double yield_elapsed = 0.0;
      int trip_reason = 0;
      double trip_force = 0.0;
      stream_ready_.store(true);

      auto control_callback = [&](const franka::RobotState &robot_state,
                                  franka::Duration period) -> franka::Torques {
        cacheRobotState(robot_state);
        q_current_ = Eigen::Map<const Vector7d>(robot_state.q.data());
        const double dt = std::clamp(period.toSec(), 0.0001, 0.01);
        const Vector7d measured_velocity =
            Eigen::Map<const Vector7d>(robot_state.dq.data());

        // A tripped guard, or a client asking to yield: stop pushing (damping
        // only) so the contact can unload, then end the stream.
        if (!yielding) {
          double force = 0.0;
          const bamboo::safety::GuardTrip trip =
              limits.enabled
                  ? bamboo::safety::ForceGuardStep(limits.guard,
                                                   robot_state.O_F_ext_hat_K,
                                                   dt, &guard_impulse, &force)
                  : bamboo::safety::GuardTrip::kNone;
          if (trip != bamboo::safety::GuardTrip::kNone ||
              stream_yield_requested_.load()) {
            yielding = true;
            trip_reason = trip != bamboo::safety::GuardTrip::kNone
                              ? static_cast<int>(trip)
                              : kYieldRequested;
            trip_force = force;
            guard_trip_.store(trip_reason);
          }
        }
        if (yielding) {
          yield_elapsed += dt;
          std::array<double, 7> damping{};
          for (int i = 0; i < 7; ++i) {
            damping[i] = -yield_damping[i] * measured_velocity[i];
          }
          const std::array<double, 7> torques = franka::limitRate(
              franka::kMaxTorqueRate, damping, robot_state.tau_J_d);
          if (yield_elapsed >= limits.yield_time) {
            return franka::MotionFinished(franka::Torques(torques));
          }
          return franka::Torques(torques);
        }

        Vector7d requested_velocity;
        const int64_t command_age_ns =
            steadyNowNanoseconds() - stream_last_update_ns_.load();
        const bool watchdog_expired =
            command_age_ns >
            static_cast<int64_t>(stream_watchdog_ms_) * 1000000;
        const bool ending =
            stream_stop_requested_.load() || global_shutdown.load();
        if (ending && stream_stop_request_ns_.load() == 0) {
          int64_t expected = 0;
          stream_stop_request_ns_.compare_exchange_strong(
              expected, steadyNowNanoseconds());
        }

        for (size_t i = 0; i < 7; ++i) {
          requested_velocity[static_cast<int>(i)] =
              (watchdog_expired || ending) ? 0.0
                                           : stream_target_velocity_[i].load();
        }

        const double max_velocity_delta = stream_max_acceleration_ * dt;
        for (int i = 0; i < 7; ++i) {
          const double difference = requested_velocity[i] - desired_velocity[i];
          desired_velocity[i] +=
              std::clamp(difference, -max_velocity_delta, max_velocity_delta);
        }

        bamboo::safety::Mat37 J_point = bamboo::safety::Mat37::Zero();
        if (limits.enabled) {
          const std::array<double, 42> jacobian =
              model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
          const Mat67 J = Eigen::Map<const Mat67>(jacobian.data());
          const Eigen::Matrix4d pose =
              Eigen::Map<const Eigen::Matrix4d>(robot_state.O_T_EE.data());
          J_point = bamboo::safety::PointJacobian(
              J, pose.topLeftCorner<3, 3>() * limits.contact_point);
          desired_velocity *= bamboo::safety::PointSpeedScale(
              J_point, desired_velocity, limits.max_point_speed);
        }

        for (int i = 0; i < 7; ++i) {
          const double next = desired_position[i] + desired_velocity[i] * dt;
          const double lower = joint_lower_[i] + kJointMargin;
          const double upper = joint_upper_[i] - kJointMargin;
          if (next < lower || next > upper) {
            desired_velocity[i] = 0.0;
            desired_position[i] = std::clamp(next, lower, upper);
          } else {
            desired_position[i] = next;
          }

          // Bound reference wind-up if contact prevents target tracking.
          desired_position[i] = std::clamp(
              desired_position[i], robot_state.q[i] - kMaxTrackingError,
              robot_state.q[i] + kMaxTrackingError);
        }

        if (limits.enabled && limits.max_force > 0.0) {
          // Push cap: scale the reference toward the measured state until the
          // controller could push no harder than max_force at the contact
          // point if the contact stopped it there.
          const Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
          const Vector7d position_error = desired_position - q;
          const Vector7d velocity_error = desired_velocity - measured_velocity;
          const Eigen::Vector3d push = bamboo::safety::CommandedPush(
              J_point, kp,
              kp.cwiseProduct(position_error) +
                  kd.cwiseProduct(velocity_error));
          const double push_norm =
              push.allFinite() ? push.norm()
                               : std::numeric_limits<double>::infinity();
          const double scale =
              bamboo::safety::PushCapScale(push_norm, limits.max_force);
          contact_push_.store(push_norm);
          contact_scale_.store(scale);
          if (!limits.shadow && scale < 1.0) {
            desired_position = q + scale * position_error;
            desired_velocity = measured_velocity + scale * velocity_error;
          }
        }

        const bamboo::controllers::ControllerResult result = controller_->Step(
            robot_state, desired_position, desired_velocity, Vector7d::Zero());
        const std::array<double, 7> torques = franka::limitRate(
            franka::kMaxTorqueRate, result.torques, robot_state.tau_J_d);
        if (result.torque_limit_violated) {
          stream_failed_.store(true);
          setStreamError("Torque limit reached during streaming control");
          return franka::MotionFinished(franka::Torques(torques));
        }

        const double stop_age_seconds =
            static_cast<double>(steadyNowNanoseconds() -
                                stream_stop_request_ns_.load()) /
            1e9;
        if (ending && desired_velocity.norm() < 1e-4 &&
            (measured_velocity.norm() < 0.01 || stop_age_seconds > 2.0)) {
          return franka::MotionFinished(franka::Torques(torques));
        }
        return franka::Torques(torques);
      };

      robot_->control(control_callback);
      if (yielding) {
        const std::string message =
            describeYield(trip_reason, trip_force, limits);
        stream_failed_.store(true);
        setStreamError(message);
        std::cout << "[STREAM] " << message << std::endl;
      }
    } catch (const franka::Exception &e) {
      stream_failed_.store(true);
      setStreamError(std::string("Franka streaming control error: ") +
                     e.what());
    } catch (const std::exception &e) {
      stream_failed_.store(true);
      setStreamError(std::string("Streaming control error: ") + e.what());
    }
    stream_ready_.store(false);
    stream_running_.store(false);
    control_running_.store(false);
  }

  bool executeTrajectory(
      const std::vector<Eigen::Matrix<double, 7, 1>> &goals,
      const std::vector<Eigen::Matrix<double, 7, 1>> &velocities,
      const std::vector<double> &durations,
      const std::vector<std::optional<std::array<double, 7>>> &kp_per_waypoint,
      const std::vector<std::optional<std::array<double, 7>>>
          &kd_per_waypoint) {
    if (goals.empty())
      return false;

    control_running_ = true;
    joint_limit_hit_ = false;
    double control_time = 0.0;

    // Reset velocity and acceleration tracking for new trajectory
    velocity_cmd_prev_.setZero();
    a_cmd_latest_.setZero();

    // Current waypoint tracking
    std::size_t current_waypoint = 0;
    double waypoint_start_time = 0.0;

    // Max joint error tracking (L1 norm across waypoint final errors)
    double max_joint_error_rad = 0.0;

    // Max end-effector error tracking
    double max_ee_position_error_m = 0.0;
    double max_ee_orientation_error_rad = 0.0;

    // Final waypoint end-effector error tracking
    double final_ee_position_error_m = 0.0;
    double final_ee_orientation_error_rad = 0.0;

    // Initialize first waypoint
    q_goal_ = goals[0];
    Eigen::Matrix<double, 7, 1> velocity_start =
        Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> velocity_goal =
        velocities.empty() ? Eigen::Matrix<double, 7, 1>::Zero()
                           : velocities[0];
    interpolator_->Reset(control_time, q_current_, q_goal_, velocity_start,
                         velocity_goal, traj_rate_, durations[0]);

    // Set gains for first waypoint (or restore to defaults if not specified)
    if (kp_per_waypoint[0].has_value() && kd_per_waypoint[0].has_value()) {
      controller_->SetGains(kp_per_waypoint[0].value(),
                            kd_per_waypoint[0].value());
    } else {
      controller_->RestoreDefaultGains();
    }

    std::cout << "[CONTROL] Starting trajectory with " << goals.size()
              << " waypoints" << std::endl;

    // Single control callback for entire trajectory
    auto control_callback = [&](const franka::RobotState &robot_state,
                                franka::Duration period) -> franka::Torques {
      try {
        // Get time step
        const double dt = period.toSec();

        // Update time
        control_time += dt;

        // Update current position
        q_current_ = Eigen::VectorXd::Map(robot_state.q.data(), 7);

        // Check if current waypoint is complete
        double waypoint_elapsed = control_time - waypoint_start_time;
        if (waypoint_elapsed >= durations[current_waypoint]) {
          if (log_err_) {
            // Log joint error for waypoint that timed out
            Eigen::Matrix<double, 7, 1> waypoint_error = q_goal_ - q_current_;
            double waypoint_final_error_rad = waypoint_error.cwiseAbs().sum();
            // Update max error across all waypoints
            if (waypoint_final_error_rad > max_joint_error_rad) {
              max_joint_error_rad = waypoint_final_error_rad;
            }

            // Calculate end-effector position and orientation errors for
            // completed waypoint Get desired EE pose from goal joint angles
            std::array<double, 7> q_goal_array;
            Eigen::VectorXd::Map(&q_goal_array[0], 7) = q_goal_;

            // Create a temporary robot state with goal joint positions
            franka::RobotState temp_state = robot_state;
            temp_state.q = q_goal_array;

            std::array<double, 16> desired_ee_pose_array =
                model_->pose(franka::Frame::kEndEffector, temp_state);
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                desired_ee_pose(desired_ee_pose_array.data());

            // Get current EE pose from robot state
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                current_ee_pose(robot_state.O_T_EE.data());

            // Calculate position error (translation part)
            Eigen::Vector3d desired_position =
                desired_ee_pose.block<3, 1>(0, 3);
            Eigen::Vector3d current_position =
                current_ee_pose.block<3, 1>(0, 3);
            double current_ee_position_error =
                (desired_position - current_position).norm();
            if (current_ee_position_error > max_ee_position_error_m) {
              max_ee_position_error_m = current_ee_position_error;
            }

            // Calculate orientation error (rotation part)
            Eigen::Matrix3d desired_rotation =
                desired_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d current_rotation =
                current_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d rotation_error =
                desired_rotation * current_rotation.transpose();

            // Convert rotation matrix to angle-axis to get scalar error
            Eigen::AngleAxisd angle_axis(rotation_error);
            double current_ee_orientation_error = std::abs(angle_axis.angle());
            if (current_ee_orientation_error > max_ee_orientation_error_rad) {
              max_ee_orientation_error_rad = current_ee_orientation_error;
            }
          }

          // Move to next waypoint
          current_waypoint++;

          if (current_waypoint >= goals.size()) {
            // If still moving, continue with current control to let robot
            // settle
            current_waypoint = goals.size() - 1; // Stay on last waypoint
          } else {
            // Setup next waypoint
            waypoint_start_time = control_time;
            q_goal_ = goals[current_waypoint];
            Eigen::Matrix<double, 7, 1> velocity_prev =
                (current_waypoint > 0 &&
                 current_waypoint - 1 < velocities.size())
                    ? velocities[current_waypoint - 1]
                    : Eigen::Matrix<double, 7, 1>::Zero();
            Eigen::Matrix<double, 7, 1> velocity_curr =
                (current_waypoint < velocities.size())
                    ? velocities[current_waypoint]
                    : Eigen::Matrix<double, 7, 1>::Zero();
            interpolator_->Reset(control_time, q_current_, q_goal_,
                                 velocity_prev, velocity_curr, traj_rate_,
                                 durations[current_waypoint]);

            // Set gains for new waypoint (or restore to defaults if not
            // specified)
            if (kp_per_waypoint[current_waypoint].has_value() &&
                kd_per_waypoint[current_waypoint].has_value()) {
              controller_->SetGains(kp_per_waypoint[current_waypoint].value(),
                                    kd_per_waypoint[current_waypoint].value());
            } else {
              controller_->RestoreDefaultGains();
            }
          }
        }

        // Get interpolated desired position and velocity for current waypoint
        Eigen::Matrix<double, 7, 1> q_desired;
        Eigen::Matrix<double, 7, 1> dq_desired;
        interpolator_->GetNextStep(control_time, q_desired, dq_desired);

        // Compute desired acceleration via finite differencing
        Eigen::Matrix<double, 7, 1> ddq_desired =
            Eigen::Matrix<double, 7, 1>::Zero();
        if (dt > 0.0) {
          // Compute raw acceleration from velocity difference
          Eigen::Matrix<double, 7, 1> a_cmd_raw =
              (dq_desired - velocity_cmd_prev_) / dt;

          // Apply low-pass filter
          for (int i = 0; i < 7; ++i) {
            a_cmd_latest_[i] = franka::lowpassFilter(
                dt, a_cmd_raw[i], a_cmd_latest_[i], diff_low_pass_freq_);
          }
          ddq_desired = a_cmd_latest_;

          // Update previous velocity for next iteration
          velocity_cmd_prev_ = dq_desired;
        }

        // Compute control torques
        bamboo::controllers::ControllerResult result =
            controller_->Step(robot_state, q_desired, dq_desired, ddq_desired);

        // Check for torque limit violation
        if (result.torque_limit_violated) {
          joint_limit_hit_ = true;
          std::cout
              << "[CONTROL] Torque limit violated - ending trajectory early"
              << std::endl;
          std::array<double, 7> zero_torques = {0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0};
          return franka::MotionFinished(franka::Torques(zero_torques));
        }

        // Apply rate limiting
        std::array<double, 7> tau_d_rate_limited = franka::limitRate(
            franka::kMaxTorqueRate, result.torques, robot_state.tau_J_d);

        // Check if all waypoints completed and robot has stopped
        if (current_waypoint >= goals.size() - 1 &&
            waypoint_elapsed >= durations[current_waypoint]) {
          Eigen::VectorXd dq_current =
              Eigen::VectorXd::Map(robot_state.dq.data(), 7);
          double velocity_norm = dq_current.norm();

          if (velocity_norm <
              0.01) { // Robot has stopped (threshold: 0.01 rad/s)
            // Calculate final waypoint errors (robot vs last goal) - robot is
            // now still
            Eigen::Matrix<double, 7, 1> final_goal = goals.back();
            std::array<double, 7> final_goal_array;
            Eigen::VectorXd::Map(&final_goal_array[0], 7) = final_goal;

            // Get desired EE pose for final waypoint
            franka::RobotState final_temp_state = robot_state;
            final_temp_state.q = final_goal_array;
            std::array<double, 16> final_desired_ee_pose_array =
                model_->pose(franka::Frame::kEndEffector, final_temp_state);
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                final_desired_ee_pose(final_desired_ee_pose_array.data());

            // Get current EE pose from robot state
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                current_ee_pose(robot_state.O_T_EE.data());

            // Calculate final position error
            Eigen::Vector3d final_desired_position =
                final_desired_ee_pose.block<3, 1>(0, 3);
            Eigen::Vector3d current_position =
                current_ee_pose.block<3, 1>(0, 3);
            final_ee_position_error_m =
                (final_desired_position - current_position).norm();

            // Calculate final orientation error
            Eigen::Matrix3d final_desired_rotation =
                final_desired_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d current_rotation =
                current_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d final_rotation_error =
                final_desired_rotation * current_rotation.transpose();
            Eigen::AngleAxisd final_angle_axis(final_rotation_error);
            final_ee_orientation_error_rad = std::abs(final_angle_axis.angle());

            std::cout << "[CONTROL] All waypoints completed, robot stopped"
                      << std::endl;
            return franka::MotionFinished(franka::Torques(tau_d_rate_limited));
          }
        }

        // Check if we should stop
        if (!control_running_ || global_shutdown) {
          return franka::MotionFinished(franka::Torques(tau_d_rate_limited));
        }

        return franka::Torques(tau_d_rate_limited);
      } catch (const std::exception &e) {
        std::cerr << "[CONTROL_CALLBACK] Exception: " << e.what() << std::endl;
        std::array<double, 7> zero_torques = {0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0};
        return franka::MotionFinished(franka::Torques(zero_torques));
      }
    };

    try {
      // Execute control
      robot_->control(control_callback);
      control_running_ = false;

      // Check if trajectory failed due to joint limit violation
      if (joint_limit_hit_) {
        std::cerr
            << "[TRAJECTORY] Trajectory failed due to joint limit violation"
            << std::endl;
        return false;
      }

      if (log_err_) {
        // Print max joint error across all waypoint final errors
        double max_joint_error_deg = max_joint_error_rad * 180.0 / M_PI;
        std::cout << "[CONTROL] Max sum of joint errors during trajectory: "
                  << std::fixed << std::setprecision(2) << max_joint_error_deg
                  << " degrees" << std::endl;

        // Print end-effector error metrics
        double max_ee_orientation_error_deg =
            max_ee_orientation_error_rad * 180.0 / M_PI;
        std::cout << "[CONTROL] Max EE position error during trajectory: "
                  << std::fixed << std::setprecision(4)
                  << max_ee_position_error_m * 1000.0 << " mm" << std::endl;
        std::cout << "[CONTROL] Max EE orientation error during trajectory: "
                  << std::fixed << std::setprecision(2)
                  << max_ee_orientation_error_deg << " degrees" << std::endl;
      }

      // Print final waypoint errors
      double final_ee_orientation_error_deg =
          final_ee_orientation_error_rad * 180.0 / M_PI;
      std::cout << "[CONTROL] Final EE position error (vs last waypoint): "
                << std::fixed << std::setprecision(4)
                << final_ee_position_error_m * 1000.0 << " mm" << std::endl;
      std::cout << "[CONTROL] Final EE orientation error (vs last waypoint): "
                << std::fixed << std::setprecision(2)
                << final_ee_orientation_error_deg << " degrees" << std::endl;

      return true;
    } catch (const franka::ControlException &e) {
      std::cerr << "[TRAJECTORY] Control exception: " << e.what() << std::endl;
      std::cerr << "[TRAJECTORY] Resetting Error State" << std::endl;
      std::cerr << "[Trajectory] If errors persist, you may have to restart "
                   "the controller by ending this tmux session "
                   "and running RunBambooController"
                << std::endl;
      robot_->automaticErrorRecovery();
      control_running_ = false;
      return false;
    }
  }
};

// Message parsing helpers
using RequestMap = std::map<std::string, msgpack::object>;

template <typename T>
T valueOr(const RequestMap &values, const std::string &key, const T &fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  T value;
  it->second.convert(value);
  return value;
}

RequestMap getRequestData(const RequestMap &request_map) {
  const auto it = request_map.find("data");
  if (it == request_map.end()) {
    return {};
  }
  RequestMap data;
  it->second.convert(data);
  return data;
}

std::string parseCommand(const RequestMap &request_map) {
  std::string command;
  auto it = request_map.find("command");
  if (it != request_map.end()) {
    it->second.convert(command);
  }
  return command;
}

std::vector<double> toVector(const Eigen::Vector3d &value) {
  return {value.x(), value.y(), value.z()};
}

void packContactLimits(msgpack::packer<msgpack::sbuffer> &packer,
                       const ContactLimits &limits) {
  packer.pack_map(11);
  packer.pack("enabled");
  packer.pack(limits.enabled);
  packer.pack("shadow");
  packer.pack(limits.shadow);
  packer.pack("max_force_n");
  packer.pack(limits.max_force);
  packer.pack("contact_point");
  packer.pack(toVector(limits.contact_point));
  packer.pack("max_point_speed");
  packer.pack(limits.max_point_speed);
  packer.pack("guard_force_n");
  packer.pack(limits.guard.force);
  packer.pack("guard_hard_force_n");
  packer.pack(limits.guard.hard_force);
  packer.pack("guard_impulse_ns");
  packer.pack(limits.guard.impulse);
  packer.pack("guard_on_invalid");
  packer.pack(limits.guard.trip_on_invalid);
  packer.pack("yield_time");
  packer.pack(limits.yield_time);
  packer.pack("collision_force_n");
  packer.pack(limits.collision_force);
}

msgpack::sbuffer handleGetCapabilities(BambooControlServer &server) {
  const BambooControlServer::Payload payload = server.GetPayload();
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(5);
  packer.pack("success");
  packer.pack(true);
  packer.pack("protocol_version");
  packer.pack(1);
  packer.pack("features");
  packer.pack(std::vector<std::string>{
      "stream_joint_velocity", "cached_robot_state", "watchdog",
      "external_wrench", "guide_mode", "set_load", "recover_errors",
      "guide_speed_leash", "gel_ratchet", "contact_limits"});
  packer.pack("contact_limits");
  packContactLimits(packer, server.GetContactLimits());
  packer.pack("payload");
  packer.pack_map(6);
  packer.pack("m_ee");
  packer.pack(payload.m_ee);
  packer.pack("m_load");
  packer.pack(payload.m_load);
  packer.pack("m_total");
  packer.pack(payload.m_total);
  packer.pack("F_x_Cee");
  packer.pack(
      std::vector<double>(payload.com_ee.begin(), payload.com_ee.end()));
  packer.pack("F_x_Cload");
  packer.pack(
      std::vector<double>(payload.com_load.begin(), payload.com_load.end()));
  packer.pack("F_x_Ctotal");
  packer.pack(
      std::vector<double>(payload.com_total.begin(), payload.com_total.end()));
  return response_buf;
}

msgpack::sbuffer
packGuideStatus(const BambooControlServer::GuideStatus &status) {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(6);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  packer.pack("stale");
  packer.pack(status.stale);
  packer.pack("phase");
  packer.pack(status.phase);
  packer.pack("reason");
  packer.pack(status.reason);
  packer.pack("level");
  packer.pack(status.level);
  return response_buf;
}

msgpack::sbuffer packSuccess() {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleStartGuide(BambooControlServer &server,
                                  const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  BambooControlServer::GuideOptions options;
  options.watchdog_ms = valueOr<int>(data, "watchdog_ms", 250);
  options.max_joint_velocity =
      valueOr<double>(data, "max_joint_velocity", 0.35);
  options.joint_damping =
      valueOr<std::vector<double>>(data, "joint_damping", {});
  options.contact_point =
      valueOr<std::vector<double>>(data, "contact_point", {});
  options.wall_damping = valueOr<double>(data, "wall_damping", -1.0);
  options.idle_hold_s = valueOr<double>(data, "idle_hold_s", -1.0);
  options.max_start_force = valueOr<double>(data, "max_start_force", 5.0);
  server.StartGuide(options);
  return packSuccess();
}

msgpack::sbuffer handleGuideUpdate(BambooControlServer &server,
                                   const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  return packGuideStatus(server.GuideUpdate(
      valueOr<double>(data, "level", 0.0), valueOr<bool>(data, "valid", true),
      valueOr<std::vector<double>>(data, "direction", {}),
      valueOr<uint64_t>(data, "seq", 0)));
}

msgpack::sbuffer handleGuideResume(BambooControlServer &server) {
  return packGuideStatus(server.GuideResume());
}

msgpack::sbuffer handleStopGuide(BambooControlServer &server) {
  server.StopGuide();
  return packSuccess();
}

msgpack::sbuffer handleRecoverErrors(BambooControlServer &server) {
  server.RecoverErrors();
  return packSuccess();
}

msgpack::sbuffer handleSetLoad(BambooControlServer &server,
                               const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  const auto mass = data.find("mass");
  if (mass == data.end()) {
    throw std::runtime_error("set_load requires mass");
  }
  server.SetLoad(valueOr<double>(data, "mass", 0.0),
                 valueOr<std::vector<double>>(data, "center_of_mass", {}),
                 valueOr<std::vector<double>>(data, "inertia", {}));
  return packSuccess();
}

msgpack::sbuffer handleSetContactLimits(BambooControlServer &server,
                                        const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  ContactLimits limits;
  limits.shadow = valueOr<bool>(data, "shadow", false);
  limits.max_force = valueOr<double>(data, "max_force_n", 0.0);
  const std::vector<double> point =
      valueOr<std::vector<double>>(data, "contact_point", {0.0, 0.0, 0.0});
  if (point.size() != 3) {
    throw std::runtime_error("contact_point must contain three values");
  }
  limits.contact_point = Eigen::Vector3d(point[0], point[1], point[2]);
  limits.max_point_speed = valueOr<double>(data, "max_point_speed", 0.0);
  limits.guard.force = valueOr<double>(data, "guard_force_n", 0.0);
  limits.guard.hard_force = valueOr<double>(data, "guard_hard_force_n", 0.0);
  limits.guard.impulse = valueOr<double>(data, "guard_impulse_ns", 0.05);
  limits.guard.trip_on_invalid = valueOr<bool>(data, "guard_on_invalid", true);
  limits.yield_time = valueOr<double>(data, "yield_time", 0.2);
  limits.collision_force = valueOr<double>(data, "collision_force_n", 0.0);
  const ContactLimits applied = server.SetContactLimits(limits);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(3);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  packer.pack("limits");
  packContactLimits(packer, applied);
  return response_buf;
}

msgpack::sbuffer handleClearContactLimits(BambooControlServer &server) {
  server.ClearContactLimits();
  return packSuccess();
}

msgpack::sbuffer handleYieldStream(BambooControlServer &server,
                                   const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  server.YieldStream(valueOr<std::string>(data, "reason", ""));
  return packSuccess();
}

msgpack::sbuffer handleStartStream(BambooControlServer &server,
                                   const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  server.StartStreaming(valueOr<int>(data, "watchdog_ms", 250),
                        valueOr<double>(data, "max_joint_velocity", 0.35),
                        valueOr<double>(data, "max_joint_acceleration", 1.5));

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleStreamJointVelocity(BambooControlServer &server,
                                           const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  server.SetStreamVelocity(valueOr<std::vector<double>>(data, "velocity", {}));

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleStopStream(BambooControlServer &server) {
  server.StopStreaming();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleGetRobotState(BambooControlServer &server) {
  bamboo_msgs::RobotState state = server.GetRobotState();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("data");
  packer.pack(state);

  return response_buf;
}

msgpack::sbuffer handleExecuteTrajectory(
    BambooControlServer &server,
    const std::map<std::string, msgpack::object> &request_map) {
  bamboo_msgs::TrajectoryRequest traj_req;
  auto it = request_map.find("data");
  if (it != request_map.end()) {
    it->second.convert(traj_req);
  }

  bool success = server.ExecuteJointImpedanceTrajectory(traj_req);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(success);
  packer.pack("error");
  if (!success) {
    packer.pack(
        std::string("Joint limit violated during trajectory execution"));
  } else {
    packer.pack(std::string(""));
  }

  return response_buf;
}

msgpack::sbuffer
handleOpenGripper(BambooControlServer &server,
                  const std::map<std::string, msgpack::object> &request_map) {
  // Parse gripper parameters
  double width = 0.08; // Default max width for Franka Hand
  double speed = 0.05; // Default speed

  auto it_width = request_map.find("width");
  if (it_width != request_map.end()) {
    it_width->second.convert(width);
  }

  auto it_speed = request_map.find("speed");
  if (it_speed != request_map.end()) {
    it_speed->second.convert(speed);
  }

  bool success = server.OpenGripper(width, speed);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(1);
  packer.pack("success");
  packer.pack(success);

  return response_buf;
}

msgpack::sbuffer
handleCloseGripper(BambooControlServer &server,
                   const std::map<std::string, msgpack::object> &request_map) {
  // Parse gripper parameters
  double speed = 0.05; // Default speed
  double force = 20.0; // Default force in Newtons

  auto it_speed = request_map.find("speed");
  if (it_speed != request_map.end()) {
    it_speed->second.convert(speed);
  }

  auto it_force = request_map.find("force");
  if (it_force != request_map.end()) {
    double force_normalized;
    it_force->second.convert(force_normalized);
    // Convert from 0-1 range to Newtons (5-70N)
    force = 5.0 + force_normalized * 65.0;
  }

  bool success = server.CloseGripper(speed, force);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(1);
  packer.pack("success");
  packer.pack(success);

  return response_buf;
}

msgpack::sbuffer handleGetGripperState(BambooControlServer &server) {
  std::map<std::string, double> state = server.GetGripperState();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("state");
  packer.pack(state);

  return response_buf;
}

msgpack::sbuffer handleTerminate() {
  std::cout << "[SERVER] Terminate request received" << std::endl;
  global_shutdown = true;

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));

  return response_buf;
}

msgpack::sbuffer handleUnknownCommand(const std::string &command) {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(false);
  packer.pack("error");
  packer.pack(std::string("Unknown command: ") + command);

  return response_buf;
}

msgpack::sbuffer handleError(const std::string &error_msg) {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(false);
  packer.pack("error");
  packer.pack(error_msg);

  return response_buf;
}

void RunServer(const std::string &server_address, franka::Robot *robot,
               franka::Model *model, franka::Gripper *gripper,
               bamboo::controllers::JointImpedanceController *controller,
               bamboo::interpolators::JointInterpolator *interpolator,
               const std::array<double, 7> &joint_lower,
               const std::array<double, 7> &joint_upper) {

  BambooControlServer server(robot, model, gripper, controller, interpolator,
                             joint_lower, joint_upper);

  // Create context and socket
  zmq::context_t context(1);
  zmq::socket_t socket(context, zmq::socket_type::rep);
  socket.bind(server_address);

  std::cout << "Server listening on " << server_address << std::endl;

  // Message handling loop
  while (!global_shutdown) {
    try {
      // Receive message
      zmq::message_t request_msg;
#if CPPZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 3, 0)
      auto result = socket.recv(request_msg, zmq::recv_flags::none);
#else
      auto result = socket.recv(&request_msg, 0);
#endif

      if (!result) {
        continue;
      }

      // Unpack the request
      msgpack::object_handle oh = msgpack::unpack(
          static_cast<const char *>(request_msg.data()), request_msg.size());
      msgpack::object obj = oh.get();

      // Parse as a map to get the command
      std::map<std::string, msgpack::object> request_map;
      obj.convert(request_map);

      std::string command = parseCommand(request_map);
      // guide_update arrives with every gel frame; logging it would only add
      // console latency to the watchdog path.
      if (command != "guide_update") {
        std::cout << "[SERVER] Received command: " << command << std::endl;
      }

      // Handle the command
      msgpack::sbuffer response_buf;

      try {
        if (command == "get_capabilities") {
          response_buf = handleGetCapabilities(server);
        } else if (command == "get_robot_state") {
          response_buf = handleGetRobotState(server);
        } else if (command == "start_stream") {
          response_buf = handleStartStream(server, request_map);
        } else if (command == "stream_joint_velocity") {
          response_buf = handleStreamJointVelocity(server, request_map);
        } else if (command == "stop_stream") {
          response_buf = handleStopStream(server);
        } else if (command == "start_guide") {
          response_buf = handleStartGuide(server, request_map);
        } else if (command == "guide_update") {
          response_buf = handleGuideUpdate(server, request_map);
        } else if (command == "guide_resume") {
          response_buf = handleGuideResume(server);
        } else if (command == "stop_guide") {
          response_buf = handleStopGuide(server);
        } else if (command == "set_load") {
          response_buf = handleSetLoad(server, request_map);
        } else if (command == "recover_errors") {
          response_buf = handleRecoverErrors(server);
        } else if (command == "set_contact_limits") {
          response_buf = handleSetContactLimits(server, request_map);
        } else if (command == "clear_contact_limits") {
          response_buf = handleClearContactLimits(server);
        } else if (command == "yield_stream") {
          response_buf = handleYieldStream(server, request_map);
        } else if (command == "execute_trajectory") {
          response_buf = handleExecuteTrajectory(server, request_map);
        } else if (command == "open_gripper") {
          response_buf = handleOpenGripper(server, request_map);
        } else if (command == "close_gripper") {
          response_buf = handleCloseGripper(server, request_map);
        } else if (command == "get_gripper_state") {
          response_buf = handleGetGripperState(server);
        } else if (command == "terminate") {
          response_buf = handleTerminate();
        } else {
          response_buf = handleUnknownCommand(command);
        }
      } catch (const std::exception &e) {
        std::cerr << "[SERVER] Error handling command: " << e.what()
                  << std::endl;
        response_buf = handleError(e.what());
      }

      // Send response
      zmq::message_t response_msg(response_buf.data(), response_buf.size());
#if CPPZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 3, 0)
      socket.send(response_msg, zmq::send_flags::none);
#else
      socket.send(response_msg,
                  0); // Old API uses reference, not pointer
                      // (https://github.com/zeromq/cppzmq/issues/69)
#endif

    } catch (const zmq::error_t &e) {
      if (e.num() == EINTR || global_shutdown) {
        break;
      }
      std::cerr << "[SERVER] Error: " << e.what() << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[SERVER] Exception in message loop: " << e.what()
                << std::endl;
    }
  }

  std::cout << "Shutting down server..." << std::endl;
  socket.close();
}

int main(int argc, char **argv) {
  // Register signal handler for graceful shutdown
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::signal(SIGHUP, signalHandler);

  std::string robot_ip;
  std::string port;
  std::string listen_address = "*";  // default
  std::string gripper_type = "none"; // default: no gripper control in C++ node
  std::string robot_model = "fr3";
  bool use_min_jerk = false;

  int opt;
  while ((opt = getopt(argc, argv, "r:p:l:g:e:mh")) != -1) {
    switch (opt) {
    case 'r':
      robot_ip = optarg;
      break;
    case 'p':
      port = optarg;
      break;
    case 'l':
      listen_address = optarg;
      break;
    case 'g':
      gripper_type = optarg;
      break;
    case 'e':
      robot_model = optarg;
      break;
    case 'm':
      use_min_jerk = true;
      break;
    case 'h':
    case '?':
    default:
      std::cerr << "Usage: " << argv[0]
                << " -r <robot-ip> -p <port> [-l <listen-address>] [-g "
                   "<gripper-type>] [-e <fr3|panda>] [-m]"
                << std::endl;
      std::cerr << "  -r: Robot IP address (required)" << std::endl;
      std::cerr << "  -p: Port number (required)" << std::endl;
      std::cerr << "  -l: Listen address (default: * for all interfaces)"
                << std::endl;
      std::cerr << "  -g: Gripper type: 'franka' or 'none' (default: none)"
                << std::endl;
      std::cerr << "  -e: Robot model: 'fr3' or 'panda' (default: fr3)"
                << std::endl;
      std::cerr << "  -m: Use min-jerk interpolation (default: linear)"
                << std::endl;
      std::cerr << "  -h: Show this help" << std::endl;
      return -1;
    }
  }

  // Validate required arguments
  if (robot_ip.empty() || port.empty()) {
    std::cerr << "Error: Robot IP and port are required" << std::endl;
    std::cerr << "Usage: " << argv[0]
              << " -r <robot-ip> -p <port> [-l <listen-address>] [-g "
                 "<gripper-type>] [-e <fr3|panda>] [-m]"
              << std::endl;
    return -1;
  }

  // Validate gripper type
  if (gripper_type != "franka" && gripper_type != "none") {
    std::cerr << "Error: Invalid gripper type '" << gripper_type
              << "'. Must be 'franka' or 'none'" << std::endl;
    return -1;
  }
  if (robot_model != "fr3" && robot_model != "panda") {
    std::cerr << "Error: Invalid robot model '" << robot_model
              << "'. Must be 'fr3' or 'panda'" << std::endl;
    return -1;
  }

  const std::string server_address = "tcp://" + listen_address + ":" + port;

  std::cout << "Bamboo Control Node Starting..." << std::endl;
  std::cout << "Robot IP: " << robot_ip << std::endl;
  std::cout << "Port: " << port << std::endl;
  std::cout << "Listen address: " << listen_address << std::endl;
  std::cout << "Gripper type: " << gripper_type << std::endl;
  std::cout << "Robot model: " << robot_model << std::endl;

  try {
    // Connect to robot
    std::cout << "Connecting to robot..." << std::endl;
    franka::Robot robot(robot_ip);
    robot.automaticErrorRecovery();

    // Set collision behavior
    robot.setCollisionBehavior(
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // Load model
    franka::Model model = robot.loadModel();

    // Conditionally connect to gripper based on gripper_type
    franka::Gripper *gripper_ptr = nullptr;
    std::unique_ptr<franka::Gripper> gripper_holder;

    if (gripper_type == "franka") {
      std::cout << "Connecting to Franka Hand gripper..." << std::endl;
      gripper_holder = std::make_unique<franka::Gripper>(robot_ip);
      gripper_ptr = gripper_holder.get();

      // Home the gripper
      std::cout << "Homing gripper..." << std::endl;
      gripper_ptr->homing();
      std::cout << "Franka Hand gripper ready" << std::endl;
    } else {
      std::cout << "No gripper control in C++ node (gripper_type=none)"
                << std::endl;
    }

    // Create controller and interpolator
    bamboo::controllers::JointImpedanceController controller(&model);
    bamboo::interpolators::InterpolatorType interp_type =
        use_min_jerk ? bamboo::interpolators::InterpolatorType::kMinJerk
                     : bamboo::interpolators::InterpolatorType::kLinear;
    bamboo::interpolators::JointInterpolator interpolator(interp_type);

    // Start server
    const auto &joint_lower =
        robot_model == "fr3" ? kFr3JointLower : kPandaJointLower;
    const auto &joint_upper =
        robot_model == "fr3" ? kFr3JointUpper : kPandaJointUpper;
    RunServer(server_address, &robot, &model, gripper_ptr, &controller,
              &interpolator, joint_lower, joint_upper);

    std::cout << "Control node terminated successfully" << std::endl;

  } catch (const franka::Exception &e) {
    std::cerr << "Franka exception: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
