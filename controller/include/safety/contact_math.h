// Contact-safety and hand-guiding math for the real-time loop.
//
// Everything here is header-only Eigen with no libfranka types, so each
// formula can be unit-tested off the robot (controller/test). The control node
// computes the robot-specific inputs (Jacobians, Franka's position-dependent
// velocity limits) and passes them in.

#ifndef BAMBOO_SAFETY_CONTACT_MATH_H
#define BAMBOO_SAFETY_CONTACT_MATH_H

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>

namespace bamboo {
namespace safety {

using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat37 = Eigen::Matrix<double, 3, 7>;
using Mat67 = Eigen::Matrix<double, 6, 7>;

inline bool AllFinite(const Vec7 &v) { return v.allFinite(); }

// Linear Jacobian of a point rigidly attached to the end effector, offset by
// r (base frame) from the end-effector origin: v_p = v + w x r = v - [r]x w.
inline Mat37 PointJacobian(const Mat67 &J_ee, const Vec3 &r_base) {
  Mat3 skew;
  skew << 0.0, -r_base.z(), r_base.y(), r_base.z(), 0.0, -r_base.x(),
      -r_base.y(), r_base.x(), 0.0;
  return J_ee.topRows<3>() - skew * J_ee.bottomRows<3>();
}

// Franka zeroes its external-wrench estimate near singularities, so an
// all-zero (or non-finite) estimate carries no information.
inline bool WrenchEstimateValid(const std::array<double, 6> &wrench) {
  bool any_nonzero = false;
  for (double value : wrench) {
    if (!std::isfinite(value)) {
      return false;
    }
    any_nonzero = any_nonzero || value != 0.0;
  }
  return any_nonzero;
}

// ---------------------------------------------------------------------------
// Hand guiding
// ---------------------------------------------------------------------------

// Viscous joint damping: the free-mode "feel". Pure damping is passive.
inline Vec7 JointDampingTorque(const Vec7 &dq, const Vec7 &damping) {
  return -damping.cwiseProduct(dq);
}

// Smooth Coulomb friction: a nearly constant resistance once a joint moves
// faster than `velocity_scale`. It holds small residual biases (gravity-model
// error) to a slow creep, where damping alone would let the arm drift at
// bias/damping. tanh is odd, so this is passive.
inline Vec7 FrictionTorque(const Vec7 &dq, const Vec7 &torque,
                           double velocity_scale) {
  Vec7 tau;
  for (int i = 0; i < 7; ++i) {
    tau[i] = -torque[i] * std::tanh(dq[i] / velocity_scale);
  }
  return tau;
}

// Extra damping on whatever part of each joint's velocity exceeds its band
// [v_lo, v_hi]. Continuous at the band edge, zero inside it.
inline Vec7 GovernorTorque(const Vec7 &dq, const Vec7 &v_lo, const Vec7 &v_hi,
                           const Vec7 &damping) {
  Vec7 tau = Vec7::Zero();
  for (int i = 0; i < 7; ++i) {
    if (dq[i] > v_hi[i]) {
      tau[i] = -damping[i] * (dq[i] - v_hi[i]);
    } else if (dq[i] < v_lo[i]) {
      tau[i] = -damping[i] * (dq[i] - v_lo[i]);
    }
  }
  return tau;
}

// Soft joint-limit zones. Within `zone` of a limit the joint is pushed away by
// a spring that ramps from zero at the zone edge to `edge_torque` at the
// limit, plus damping on motion toward the limit that scales the same way.
inline Vec7 JointLimitTorque(const Vec7 &q, const Vec7 &dq, const Vec7 &lower,
                             const Vec7 &upper, const Vec7 &zone,
                             const Vec7 &edge_torque, const Vec7 &damping) {
  Vec7 tau = Vec7::Zero();
  for (int i = 0; i < 7; ++i) {
    if (zone[i] <= 0.0) {
      continue;
    }
    const double to_lower = q[i] - lower[i];
    if (to_lower < zone[i]) {
      const double depth = std::clamp((zone[i] - to_lower) / zone[i], 0.0, 1.0);
      tau[i] += edge_torque[i] * depth;
      if (dq[i] < 0.0) {
        tau[i] -= damping[i] * depth * dq[i];
      }
    }
    const double to_upper = upper[i] - q[i];
    if (to_upper < zone[i]) {
      const double depth = std::clamp((zone[i] - to_upper) / zone[i], 0.0, 1.0);
      tau[i] -= edge_torque[i] * depth;
      if (dq[i] > 0.0) {
        tau[i] -= damping[i] * depth * dq[i];
      }
    }
  }
  return tau;
}

inline bool WithinMarginOfLimit(const Vec7 &q, const Vec7 &lower,
                                const Vec7 &upper, double margin) {
  for (int i = 0; i < 7; ++i) {
    if (q[i] - lower[i] < margin || upper[i] - q[i] < margin) {
      return true;
    }
  }
  return false;
}

// One-way damper "wall". With a direction n (unit, base frame) it resists only
// motion of the contact point along +n, at `level * damping` N s/m, and leaves
// motion along -n free. Without a direction it damps translation and rotation
// alike. A damper whose coefficient changes over time is still passive, which
// is why the wall is not a spring: a level-modulated spring would inject
// energy when the slow (~19 Hz, delayed) level rises while it is compressed.
inline Vec7 DamperWallTorque(const Mat37 &J_lin, const Mat37 &J_ang,
                             const Vec7 &dq, double level, bool has_direction,
                             const Vec3 &direction, double damping,
                             double rotational_damping) {
  if (level <= 0.0) {
    return Vec7::Zero();
  }
  const Vec3 v = J_lin * dq;
  if (has_direction) {
    const double toward = direction.dot(v);
    if (toward <= 0.0) {
      return Vec7::Zero();
    }
    const Vec3 force = -level * damping * toward * direction;
    return J_lin.transpose() * force;
  }
  const Vec3 force = -level * damping * v;
  const Vec3 moment = -level * rotational_damping * (J_ang * dq);
  return J_lin.transpose() * force + J_ang.transpose() * moment;
}

enum class GuidePhase : int {
  kFree = 0,
  kHoldDamp = 1, // damp to a stop before latching, so the hold never springs
  kHoldRamp = 2, // latched; hold stiffness ramps in
  kHold = 3,
  kRelease = 4, // operator resumed; hold torque ramps out
};

enum class HoldReason : int {
  kNone = 0,
  kWatchdog = 1,   // guide updates stopped arriving
  kGelInvalid = 2, // the client says its gel signal is not trustworthy
  kGelLimit = 3,   // gel load reached the client's limit
  kJointLimit = 4, // a joint came within the hold margin of its limit
  kStopping = 5,   // stop_guide
  kIdle = 6,       // the arm was left still; breaks free when pushed
};

// Damping defaults are chosen so that any combination active at once stays
// within the joint damping of Franka's own joint impedance example
// (50, 50, 50, 50, 30, 25, 15 N m s/rad), which is known to be stable with the
// FCI torque filter.
struct GuideParams {
  Vec7 damping = (Vec7() << 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.5).finished();
  // Friction torque reached above ~friction_velocity; its slope near zero
  // speed (torque / friction_velocity) stays within the damping budget.
  Vec7 friction = (Vec7() << 2.0, 2.0, 2.0, 2.0, 0.8, 0.8, 0.4).finished();
  double friction_velocity = 0.04; // rad/s
  // Idle hold: still for idle_time -> hold; pushed past the breakaway
  // deflection -> free again, without a resume. idle_time <= 0 disables it.
  // "Still" is set above the creep that friction leaves for a ~1 N m
  // gravity-model bias, so a released arm is caught rather than creeping.
  double idle_velocity = 0.025; // rad/s, all joints
  double idle_time = 1.0;       // s
  double breakaway = 0.005;     // rad from the held pose, any joint
  Vec7 governor_damping =
      (Vec7() << 25.0, 25.0, 25.0, 25.0, 10.0, 10.0, 5.0).finished();
  Vec7 limit_zone = Vec7::Constant(0.25);
  Vec7 limit_torque =
      (Vec7() << 12.0, 12.0, 12.0, 12.0, 4.0, 4.0, 4.0).finished();
  Vec7 limit_damping =
      (Vec7() << 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 3.0).finished();
  double hold_margin = 0.05;   // rad from a hard limit
  double resume_margin = 0.10; // must be this far out to resume
  Vec7 hold_kp = (Vec7() << 437.5, 525.0, 525.0, 437.5, 87.5, 175.0, 43.75)
                     .finished(); // half of Bamboo's default stiffness
  Vec7 hold_kd = (Vec7() << 37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5).finished();
  Vec7 hold_damping =
      (Vec7() << 50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0).finished();
  double settle_velocity = 0.02;        // rad/s, all joints, to latch the hold
  double settle_timeout = 0.3;          // s of damping before latching anyway
  double hold_ramp = 0.25;              // s for the hold stiffness to ramp in
  double release_time = 0.2;            // s for the hold torque to ramp out
  double level_time_constant = 0.03;    // s, smooths the ~19 Hz gel level
  double wall_damping = 100.0;          // N s/m at level 1
  double wall_rotational_damping = 2.0; // N m s/rad at level 1
};

struct GuideInputs {
  Vec7 q = Vec7::Zero();
  Vec7 dq = Vec7::Zero();
  Vec7 lower = Vec7::Zero();
  Vec7 upper = Vec7::Zero();
  Vec7 velocity_low = Vec7::Zero();  // governor band, already combined with
  Vec7 velocity_high = Vec7::Zero(); // Franka's position-dependent limits
  Mat37 J_lin = Mat37::Zero();       // contact point, base frame
  Mat37 J_ang = Mat37::Zero();
  double dt = 0.001;
  bool update_fresh = true;
  bool level_valid = true;
  double level = 0.0;
  bool has_direction = false;
  Vec3 direction = Vec3::Zero(); // unit, base frame
  bool resume_requested = false;
  bool stop_requested = false;
};

struct GuideState {
  GuidePhase phase = GuidePhase::kFree;
  HoldReason reason = HoldReason::kNone;
  double phase_time = 0.0;
  double idle_time = 0.0;
  double level = 0.0; // filtered
  Vec7 q_hold = Vec7::Zero();
};

inline Vec7 HoldTorque(const GuideParams &p, const Vec7 &q, const Vec7 &dq,
                       const Vec7 &q_hold, double stiffness_scale) {
  return stiffness_scale * p.hold_kp.cwiseProduct(q_hold - q) -
         p.hold_kd.cwiseProduct(dq);
}

inline Vec7 FreeTorque(const GuideParams &p, const GuideInputs &in,
                       double level) {
  return JointDampingTorque(in.dq, p.damping) +
         FrictionTorque(in.dq, p.friction, p.friction_velocity) +
         GovernorTorque(in.dq, in.velocity_low, in.velocity_high,
                        p.governor_damping) +
         JointLimitTorque(in.q, in.dq, in.lower, in.upper, p.limit_zone,
                          p.limit_torque, p.limit_damping) +
         DamperWallTorque(in.J_lin, in.J_ang, in.dq, level, in.has_direction,
                          in.direction, p.wall_damping,
                          p.wall_rotational_damping);
}

inline HoldReason HoldTrigger(const GuideParams &p, const GuideInputs &in) {
  if (in.stop_requested) {
    return HoldReason::kStopping;
  }
  if (!in.update_fresh) {
    return HoldReason::kWatchdog;
  }
  if (!in.level_valid || !std::isfinite(in.level)) {
    return HoldReason::kGelInvalid;
  }
  if (in.level >= 1.0) {
    return HoldReason::kGelLimit;
  }
  if (WithinMarginOfLimit(in.q, in.lower, in.upper, p.hold_margin)) {
    return HoldReason::kJointLimit;
  }
  return HoldReason::kNone;
}

inline bool ResumeAllowed(const GuideParams &p, const GuideInputs &in) {
  return !in.stop_requested && in.update_fresh && in.level_valid &&
         std::isfinite(in.level) && in.level < 0.5 &&
         !WithinMarginOfLimit(in.q, in.lower, in.upper, p.resume_margin);
}

// One 1 kHz step of the hand-guiding controller. Returns joint torques
// (gravity is compensated by Franka; no Coriolis term, so free mode stays
// passive). Holds are entered automatically but left only on an explicit,
// allowed resume, so the arm never lurches while someone is pushing on it.
inline Vec7 GuideStep(const GuideParams &p, const GuideInputs &in,
                      GuideState *s) {
  const double dt = std::clamp(in.dt, 1e-4, 1e-2);
  const double target_level =
      std::isfinite(in.level) ? std::clamp(in.level, 0.0, 1.0) : 1.0;
  const double alpha = dt / (p.level_time_constant + dt);
  s->level += alpha * (target_level - s->level);
  s->phase_time += dt;

  const HoldReason trigger = HoldTrigger(p, in);
  const bool holding = s->phase == GuidePhase::kHoldDamp ||
                       s->phase == GuidePhase::kHoldRamp ||
                       s->phase == GuidePhase::kHold;
  const bool idle_hold = holding && s->reason == HoldReason::kIdle;
  const bool still = in.dq.cwiseAbs().maxCoeff() < p.idle_velocity;
  s->idle_time =
      (s->phase == GuidePhase::kFree && still) ? s->idle_time + dt : 0.0;
  if (!holding && trigger != HoldReason::kNone) {
    s->phase = GuidePhase::kHoldDamp;
    s->reason = trigger;
    s->phase_time = 0.0;
  } else if (idle_hold && trigger != HoldReason::kNone) {
    // A safety reason replaces the idle one, so pushing no longer frees it.
    s->reason = trigger;
  } else if (holding && in.stop_requested) {
    s->reason = HoldReason::kStopping;
  } else if (s->phase == GuidePhase::kFree && p.idle_time > 0.0 &&
             s->idle_time >= p.idle_time) {
    // Already still, so latch straight away and ramp the stiffness in.
    s->phase = GuidePhase::kHoldRamp;
    s->reason = HoldReason::kIdle;
    s->phase_time = 0.0;
    s->q_hold = in.q;
  } else if (idle_hold && s->phase != GuidePhase::kHoldDamp &&
             (in.q - s->q_hold).cwiseAbs().maxCoeff() > p.breakaway) {
    s->phase = GuidePhase::kRelease;
    s->reason = HoldReason::kNone;
    s->phase_time = 0.0;
  } else if (s->phase == GuidePhase::kHold && in.resume_requested &&
             ResumeAllowed(p, in)) {
    s->phase = GuidePhase::kRelease;
    s->reason = HoldReason::kNone;
    s->phase_time = 0.0;
  }

  const Vec7 limits =
      JointLimitTorque(in.q, in.dq, in.lower, in.upper, p.limit_zone,
                       p.limit_torque, p.limit_damping);
  switch (s->phase) {
  case GuidePhase::kFree:
    return FreeTorque(p, in, s->level);
  case GuidePhase::kHoldDamp: {
    const bool settled = in.dq.cwiseAbs().maxCoeff() < p.settle_velocity;
    if (settled || s->phase_time >= p.settle_timeout) {
      s->phase = GuidePhase::kHoldRamp;
      s->phase_time = 0.0;
      s->q_hold = in.q;
      return -p.hold_kd.cwiseProduct(in.dq) + limits;
    }
    return -p.hold_damping.cwiseProduct(in.dq) + limits;
  }
  case GuidePhase::kHoldRamp: {
    const double ramp = std::clamp(s->phase_time / p.hold_ramp, 0.0, 1.0);
    if (ramp >= 1.0) {
      s->phase = GuidePhase::kHold;
      s->phase_time = 0.0;
    }
    return HoldTorque(p, in.q, in.dq, s->q_hold, ramp);
  }
  case GuidePhase::kHold:
    return HoldTorque(p, in.q, in.dq, s->q_hold, 1.0);
  case GuidePhase::kRelease: {
    const double remaining =
        std::clamp(1.0 - s->phase_time / p.release_time, 0.0, 1.0);
    if (remaining <= 0.0) {
      s->phase = GuidePhase::kFree;
      s->phase_time = 0.0;
      return FreeTorque(p, in, s->level);
    }
    return remaining * HoldTorque(p, in.q, in.dq, s->q_hold, 1.0) +
           (1.0 - remaining) * FreeTorque(p, in, s->level);
  }
  }
  return Vec7::Zero();
}

// The governor band for one tick: the configured speed, tightened to a
// fraction of Franka's position-dependent joint velocity limits (they shrink
// to zero at the position limits, and crossing them is a reflex error).
inline void GovernorBand(const std::array<double, 7> &franka_upper,
                         const std::array<double, 7> &franka_lower,
                         double max_velocity, double franka_fraction, Vec7 *low,
                         Vec7 *high) {
  for (int i = 0; i < 7; ++i) {
    (*high)[i] = std::min(max_velocity, franka_fraction * franka_upper[i]);
    (*low)[i] = std::max(-max_velocity, franka_fraction * franka_lower[i]);
    if ((*low)[i] > (*high)[i]) {
      (*low)[i] = (*high)[i] = 0.0;
    }
  }
}

} // namespace safety
} // namespace bamboo

#endif // BAMBOO_SAFETY_CONTACT_MATH_H
