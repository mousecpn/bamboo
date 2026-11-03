#pragma once

#include <msgpack.hpp>
#include <string>
#include <vector>

namespace bamboo_msgs {
// Robot state message
struct RobotState {
  std::vector<double> q;      // Joint positions (7)
  std::vector<double> dq;     // Joint velocities (7)
  std::vector<double> tau_J;  // Joint torques (7)
  std::vector<double> O_T_EE; // End-effector pose (16 - 4x4 matrix)
  double time_sec;
  // Franka's estimates, robot on environment. The wrench is zeroed by Franka
  // near singularities.
  std::vector<double> O_F_ext_hat_K;        // External wrench (6), base frame
  std::vector<double> tau_ext_hat_filtered; // External joint torque (7)
  int loop_mode = 0;        // 0 idle, 1 trajectory, 2 stream, 3 guide
  int guide_phase = 0;      // safety::GuidePhase while guiding
  int guide_reason = 0;     // safety::HoldReason while guiding
  double guide_level = 0.0; // filtered wall level while guiding

  MSGPACK_DEFINE_MAP(q, dq, tau_J, O_T_EE, time_sec, O_F_ext_hat_K,
                     tau_ext_hat_filtered, loop_mode, guide_phase, guide_reason,
                     guide_level)
};

// Waypoint with timing information
struct TimedWaypoint {
  std::vector<double> q_goal;   // Goal joint positions (7)
  std::vector<double> velocity; // Desired velocity at waypoint (7)
  double duration;              // Duration for this waypoint
  std::vector<double> kp;       // Stiffness (7)
  std::vector<double> kd;       // Damping (7)

  MSGPACK_DEFINE_MAP(q_goal, velocity, duration, kp, kd)
};

// Trajectory request
struct TrajectoryRequest {
  std::vector<TimedWaypoint> waypoints;
  double default_duration;
  std::vector<double> default_velocity;

  MSGPACK_DEFINE_MAP(waypoints, default_duration, default_velocity)
};
} // namespace bamboo_msgs
