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
#include <limits>

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

// Joint limits, in two layers. Within `zone` of a limit the joint is pushed
// away by a spring that ramps from zero at the zone edge to `edge_torque` at
// the limit, with damping on motion toward the limit that scales the same way.
// Within `wall_margin` a stiff one-way wall (`wall_stiffness`, damped both
// ways) adds to it and keeps the joint off Franka's hard limit, which would be
// a reflex error. Moving away from the limit stays free, so reaching it never
// needs a resume.
inline Vec7 JointLimitTorque(const Vec7 &q, const Vec7 &dq, const Vec7 &lower,
                             const Vec7 &upper, const Vec7 &zone,
                             const Vec7 &edge_torque, const Vec7 &damping,
                             double wall_margin, const Vec7 &wall_stiffness,
                             const Vec7 &wall_damping) {
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
    if (to_lower < wall_margin) {
      tau[i] += wall_stiffness[i] * (wall_margin - to_lower) -
                wall_damping[i] * dq[i];
    }
    const double to_upper = upper[i] - q[i];
    if (to_upper < zone[i]) {
      const double depth = std::clamp((zone[i] - to_upper) / zone[i], 0.0, 1.0);
      tau[i] -= edge_torque[i] * depth;
      if (dq[i] > 0.0) {
        tau[i] -= damping[i] * depth * dq[i];
      }
    }
    if (to_upper < wall_margin) {
      tau[i] -= wall_stiffness[i] * (wall_margin - to_upper) +
                wall_damping[i] * dq[i];
    }
  }
  return tau;
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

// Speed leash. Each joint has a shadow that follows it but may move only
// within the speed band [v_lo, v_hi]. While the arm stays inside the band the
// shadow sits on it and the leash is slack. Faster, the arm gets ahead of its
// shadow and a spring (kp) pulls it back, with damping (kd) that fades in over
// the first `blend` rad of lead, so the leash takes up and lets go without a
// torque step. The spring stops growing at `max_torque`: pushing harder than
// that drags the shadow along. The shadow only ever moves toward the arm,
// which removes spring energy and never adds any, so the leash is passive.
// Returns the torque; updates *q_shadow and the largest lead in *max_lead_out.
inline Vec7 LeashTorque(const Vec7 &q, const Vec7 &dq, const Vec7 &v_lo,
                        const Vec7 &v_hi, double dt, const Vec7 &kp,
                        const Vec7 &kd, const Vec7 &max_torque, double blend,
                        Vec7 *q_shadow, double *max_lead_out) {
  Vec7 tau = Vec7::Zero();
  double max_lead = 0.0;
  for (int i = 0; i < 7; ++i) {
    double &shadow = (*q_shadow)[i];
    shadow += std::clamp(q[i] - shadow, std::min(v_lo[i], 0.0) * dt,
                         std::max(v_hi[i], 0.0) * dt);
    double lead = q[i] - shadow;
    const double lead_limit = kp[i] > 0.0 ? max_torque[i] / kp[i] : 0.0;
    if (std::abs(lead) > lead_limit) {
      lead = std::copysign(lead_limit, lead);
      shadow = q[i] - lead;
    }
    const double weight =
        blend > 0.0 ? std::min(1.0, std::abs(lead) / blend) : 1.0;
    tau[i] = -kp[i] * lead - weight * kd[i] * dq[i];
    max_lead = std::max(max_lead, std::abs(lead));
  }
  if (max_lead_out != nullptr) {
    *max_lead_out = max_lead;
  }
  return tau;
}

// One-way wall in Cartesian space ("ratchet"): the contact point p may not
// advance along n (unit, base frame) past `anchor`, but backing off is free and
// drags the anchor along, so the wall always sits where the point last was.
// Inside the wall a spring and damper push back (never pull); at the wall only
// advancing motion is damped. The anchor moves only while the spring is
// unloaded, so it never releases stored energy into the arm.
inline Vec7 RatchetTorque(const Mat37 &J_lin, const Vec7 &dq, const Vec3 &p,
                          const Vec3 &n, double stiffness, double damping,
                          Vec3 *anchor, double *depth_out) {
  double depth = n.dot(p - *anchor);
  if (depth < 0.0) {
    *anchor += depth * n;
    depth = 0.0;
  }
  if (depth_out != nullptr) {
    *depth_out = depth;
  }
  const double speed = n.dot(J_lin * dq);
  const double force = depth > 0.0
                           ? std::max(0.0, stiffness * depth + damping * speed)
                           : damping * std::max(0.0, speed);
  return -(J_lin.transpose() * (force * n));
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
  kJointLimit = 4, // reserved: joint limits use a one-way wall, not a hold
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
  double limit_wall_margin = 0.05; // rad from a hard limit; the wall uses
                                   // hold_kp and hold_kd
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
  // Speed leash: Bamboo's streaming stiffness, so a hand cannot get the arm
  // ahead of what streaming could replay by more than max_torque / kp.
  Vec7 leash_kp =
      (Vec7() << 875.0, 1050.0, 1050.0, 875.0, 175.0, 350.0, 87.5).finished();
  Vec7 leash_kd = (Vec7() << 37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5).finished();
  Vec7 leash_max_torque =
      (Vec7() << 40.0, 40.0, 40.0, 40.0, 20.0, 10.0, 10.0).finished();
  double leash_blend = 0.005; // rad of lead over which leash damping fades in
  // Gel ratchet: from ratchet_engage_level (with a load direction) the contact
  // point cannot advance further along it; it lets go below the release level.
  double ratchet_stiffness = 3000.0; // N/m
  double ratchet_damping = 120.0;    // N s/m
  double ratchet_engage_level = 1.0;
  double ratchet_release_level = 0.6;
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
  Vec3 contact_position = Vec3::Zero(); // contact point, base frame
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
  // Speed leash; re-seeded on the arm whenever it is not floating.
  Vec7 q_shadow = Vec7::Zero();
  bool shadow_valid = false;
  double lead = 0.0; // largest |q - q_shadow|, rad
  // Gel ratchet; kept through holds so a hold cannot reset it deeper.
  bool ratchet_engaged = false;
  Vec3 ratchet_anchor = Vec3::Zero();
  Vec3 ratchet_direction = Vec3::Zero();
  double ratchet_depth = 0.0; // m past the anchor
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
                          p.limit_torque, p.limit_damping, p.limit_wall_margin,
                          p.hold_kp, p.hold_kd) +
         DamperWallTorque(in.J_lin, in.J_ang, in.dq, level, in.has_direction,
                          in.direction, p.wall_damping,
                          p.wall_rotational_damping);
}

inline HoldReason HoldTrigger(const GuideInputs &in) {
  if (in.stop_requested) {
    return HoldReason::kStopping;
  }
  if (!in.update_fresh) {
    return HoldReason::kWatchdog;
  }
  if (!in.level_valid || !std::isfinite(in.level)) {
    return HoldReason::kGelInvalid;
  }
  // With a load direction the ratchet stops further loading and still lets
  // the arm back off; a hold would freeze the load in place instead.
  if (in.level >= 1.0 && !in.has_direction) {
    return HoldReason::kGelLimit;
  }
  return HoldReason::kNone;
}

inline bool ResumeAllowed(const GuideInputs &in) {
  return !in.stop_requested && in.update_fresh && in.level_valid &&
         std::isfinite(in.level) && (in.has_direction || in.level < 0.5);
}

// Engages, keeps or releases the gel ratchet and returns its torque.
inline Vec7 GelRatchetStep(const GuideParams &p, const GuideInputs &in,
                           GuideState *s) {
  const bool directed =
      in.has_direction && in.direction.allFinite() && in.direction.norm() > 0.5;
  const double level = std::isfinite(in.level) ? in.level : 0.0;
  if (!directed || !in.level_valid || level < p.ratchet_release_level) {
    s->ratchet_engaged = false;
  } else if (!s->ratchet_engaged && level >= p.ratchet_engage_level) {
    s->ratchet_engaged = true;
    s->ratchet_direction = in.direction.normalized();
    s->ratchet_anchor = in.contact_position;
  }
  if (!s->ratchet_engaged) {
    s->ratchet_depth = 0.0;
    return Vec7::Zero();
  }
  return RatchetTorque(in.J_lin, in.dq, in.contact_position,
                       s->ratchet_direction, p.ratchet_stiffness,
                       p.ratchet_damping, &s->ratchet_anchor,
                       &s->ratchet_depth);
}

// Free-mode torque with the stateful parts: the speed leash on top of
// FreeTorque. Seeds the shadow on the arm the first time.
inline Vec7 FreeStep(const GuideParams &p, const GuideInputs &in, GuideState *s,
                     double dt) {
  if (!s->shadow_valid) {
    s->q_shadow = in.q;
    s->shadow_valid = true;
  }
  return FreeTorque(p, in, s->level) +
         LeashTorque(in.q, in.dq, in.velocity_low, in.velocity_high, dt,
                     p.leash_kp, p.leash_kd, p.leash_max_torque, p.leash_blend,
                     &s->q_shadow, &s->lead);
}

inline Vec7 PhaseTorque(const GuideParams &p, const GuideInputs &in,
                        GuideState *s, double dt);

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

  const HoldReason trigger = HoldTrigger(in);
  const bool holding = s->phase == GuidePhase::kHoldDamp ||
                       s->phase == GuidePhase::kHoldRamp ||
                       s->phase == GuidePhase::kHold;
  const bool idle_hold = holding && s->reason == HoldReason::kIdle;
  const bool still = in.dq.cwiseAbs().maxCoeff() < p.idle_velocity;
  // Held still by the ratchet is not "left still": no idle hold, which would
  // otherwise cycle with the breakaway while someone pushes on the wall.
  s->idle_time = (s->phase == GuidePhase::kFree && still && !s->ratchet_engaged)
                     ? s->idle_time + dt
                     : 0.0;
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
             ResumeAllowed(in)) {
    s->phase = GuidePhase::kRelease;
    s->reason = HoldReason::kNone;
    s->phase_time = 0.0;
  }

  const Vec7 ratchet = GelRatchetStep(p, in, s);
  const bool floating =
      s->phase == GuidePhase::kFree || s->phase == GuidePhase::kRelease;
  if (!floating) {
    s->shadow_valid = false;
    s->lead = 0.0;
  }
  return ratchet + PhaseTorque(p, in, s, dt);
}

// The phase's own torque for GuideStep (everything except the ratchet).
inline Vec7 PhaseTorque(const GuideParams &p, const GuideInputs &in,
                        GuideState *s, double dt) {
  const Vec7 limits = JointLimitTorque(
      in.q, in.dq, in.lower, in.upper, p.limit_zone, p.limit_torque,
      p.limit_damping, p.limit_wall_margin, p.hold_kp, p.hold_kd);
  switch (s->phase) {
  case GuidePhase::kFree:
    return FreeStep(p, in, s, dt);
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
      return FreeStep(p, in, s, dt);
    }
    return remaining * HoldTorque(p, in.q, in.dq, s->q_hold, 1.0) +
           (1.0 - remaining) * FreeStep(p, in, s, dt);
  }
  }
  return Vec7::Zero();
}

// ---------------------------------------------------------------------------
// Contact limits for streaming
// ---------------------------------------------------------------------------

// The force the streaming impedance would push with at the contact point if
// contact stopped the point where it is. The controller's reference torque
// tau = Kp e + Kd e_dot is a joint deflection Kp^-1 tau, which moves the point
// by J Kp^-1 tau against the arm's stiffness there, (J Kp^-1 J^T)^-1. Motion
// the contact does not block (self-motion, sliding the wrist) pushes nothing.
// Returns infinity when it cannot be computed, so callers fail closed.
inline Vec3 CommandedPush(const Mat37 &J, const Vec7 &kp, const Vec7 &tau) {
  const Vec7 compliance = kp.cwiseInverse();
  const Mat3 C = J * compliance.asDiagonal() * J.transpose();
  const Vec3 dx = J * compliance.cwiseProduct(tau);
  const Eigen::LDLT<Mat3> ldlt(C);
  const Vec3 force = ldlt.solve(dx);
  if (ldlt.info() != Eigen::Success || !force.allFinite() ||
      !compliance.allFinite()) {
    return Vec3::Constant(std::numeric_limits<double>::infinity());
  }
  return force;
}

// Factor in [0, 1] for the reference error that keeps a push within
// max_force (no limit when max_force <= 0); 0 when the push is unknown.
inline double PushCapScale(double push, double max_force) {
  if (!(max_force > 0.0)) {
    return 1.0;
  }
  if (!std::isfinite(push)) {
    return 0.0;
  }
  return push <= max_force ? 1.0 : max_force / push;
}

// Factor in [0, 1] for a joint velocity reference that keeps the contact
// point within max_speed (no limit when max_speed <= 0).
inline double PointSpeedScale(const Mat37 &J, const Vec7 &dq_ref,
                              double max_speed) {
  if (!(max_speed > 0.0)) {
    return 1.0;
  }
  const double speed = (J * dq_ref).norm();
  if (!std::isfinite(speed)) {
    return 0.0;
  }
  return speed <= max_speed ? 1.0 : max_speed / speed;
}

// Guard on Franka's external force estimate (robot on environment). With
// nobody touching the arm this is the force at the tool. A load above `force`
// trips once its excess integrates to `impulse` (so millisecond spikes from
// the estimate pass), anything above `hard_force` trips at once, and an
// estimate Franka has zeroed (near a singularity) trips when trip_on_invalid.
struct ForceGuardParams {
  double force = 0.0;      // N; 0 disables the guard
  double impulse = 0.05;   // N s above `force`
  double hard_force = 0.0; // N; 0 means 1.5 x force
  bool trip_on_invalid = true;
};

enum class GuardTrip : int {
  kNone = 0,
  kSustained = 1,
  kHard = 2,
  kInvalid = 3
};

inline GuardTrip ForceGuardStep(const ForceGuardParams &p,
                                const std::array<double, 6> &wrench, double dt,
                                double *impulse, double *force_out) {
  const Vec3 force(wrench[0], wrench[1], wrench[2]);
  const double magnitude = force.allFinite() ? force.norm() : 0.0;
  if (force_out != nullptr) {
    *force_out = magnitude;
  }
  if (!(p.force > 0.0)) {
    return GuardTrip::kNone;
  }
  if (!WrenchEstimateValid(wrench)) {
    return p.trip_on_invalid ? GuardTrip::kInvalid : GuardTrip::kNone;
  }
  const double hard = p.hard_force > 0.0 ? p.hard_force : 1.5 * p.force;
  if (magnitude >= hard) {
    return GuardTrip::kHard;
  }
  *impulse = std::max(0.0, *impulse + (magnitude - p.force) * dt);
  return *impulse > p.impulse ? GuardTrip::kSustained : GuardTrip::kNone;
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
