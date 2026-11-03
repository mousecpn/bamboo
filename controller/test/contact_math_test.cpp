// Off-robot tests for safety/contact_math.h.

#include "safety/contact_math.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>

using namespace bamboo::safety;

namespace {

// FR3 forward kinematics, ported from fr3_demo/kinematics.py
// (franka_description).
struct Chain {
  Eigen::Matrix4d flange;
  std::array<Vec3, 7> origins;
  std::array<Vec3, 7> axes;
};

Chain Fr3Chain(const Vec7 &q) {
  const double xyz[7][3] = {{0, 0, 0.333},  {0, 0, 0},           {0, -0.316, 0},
                            {0.0825, 0, 0}, {-0.0825, 0.384, 0}, {0, 0, 0},
                            {0.088, 0, 0}};
  const double roll[7] = {0,         -M_PI / 2, M_PI / 2, M_PI / 2,
                          -M_PI / 2, M_PI / 2,  M_PI / 2};
  Chain chain;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 7; ++i) {
    Eigen::Matrix4d fixed = Eigen::Matrix4d::Identity();
    fixed.block<3, 3>(0, 0) =
        Eigen::AngleAxisd(roll[i], Vec3::UnitX()).toRotationMatrix();
    fixed.block<3, 1>(0, 3) = Vec3(xyz[i][0], xyz[i][1], xyz[i][2]);
    pose = pose * fixed;
    chain.origins[i] = pose.block<3, 1>(0, 3);
    chain.axes[i] = pose.block<3, 1>(0, 2);
    Eigen::Matrix4d joint = Eigen::Matrix4d::Identity();
    joint.block<3, 3>(0, 0) =
        Eigen::AngleAxisd(q[i], Vec3::UnitZ()).toRotationMatrix();
    pose = pose * joint;
  }
  Eigen::Matrix4d link8 = Eigen::Matrix4d::Identity();
  link8(2, 3) = 0.107;
  chain.flange = pose * link8;
  return chain;
}

Mat67 Fr3Jacobian(const Vec7 &q) {
  const Chain chain = Fr3Chain(q);
  const Vec3 end = chain.flange.block<3, 1>(0, 3);
  Mat67 J;
  for (int i = 0; i < 7; ++i) {
    J.block<3, 1>(0, i) = chain.axes[i].cross(end - chain.origins[i]);
    J.block<3, 1>(3, i) = chain.axes[i];
  }
  return J;
}

Vec7 Home() {
  return (Vec7() << 0.0, -M_PI / 4, 0.0, -3 * M_PI / 4, 0.0, M_PI / 2, M_PI / 4)
      .finished();
}

const Vec7 kLower =
    (Vec7() << -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508)
        .finished();
const Vec7 kUpper =
    (Vec7() << 2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508)
        .finished();

GuideInputs FreeInputs(const Vec7 &q, const Vec7 &dq) {
  GuideInputs in;
  in.q = q;
  in.dq = dq;
  in.lower = kLower;
  in.upper = kUpper;
  in.velocity_low = Vec7::Constant(-0.35);
  in.velocity_high = Vec7::Constant(0.35);
  const Mat67 J = Fr3Jacobian(q);
  in.J_lin = J.topRows<3>();
  in.J_ang = J.bottomRows<3>();
  return in;
}

// Joint-space double integrator with roughly FR3-like reflected inertia.
const Vec7 kInertia = (Vec7() << 1.5, 1.5, 0.8, 0.8, 0.3, 0.2, 0.1).finished();

} // namespace

TEST(Jacobian, PointJacobianMatchesFiniteDifference) {
  const Vec7 q = Home() + Vec7::Constant(0.1);
  const Chain chain = Fr3Chain(q);
  const Mat3 R = chain.flange.block<3, 3>(0, 0);
  const Vec3 offset_ee(0.01, -0.02, 0.2);
  const Mat37 J_p = PointJacobian(Fr3Jacobian(q), R * offset_ee);
  const auto point = [&](const Vec7 &joints) {
    const Chain c = Fr3Chain(joints);
    return Vec3(c.flange.block<3, 1>(0, 3) +
                c.flange.block<3, 3>(0, 0) * offset_ee);
  };
  const double eps = 1e-7;
  for (int i = 0; i < 7; ++i) {
    Vec7 dq = q;
    dq[i] += eps;
    const Vec3 numeric = (point(dq) - point(q)) / eps;
    EXPECT_LT((numeric - J_p.col(i)).norm(), 1e-5) << "joint " << i;
  }
}

TEST(Wrench, ZeroOrNonFiniteEstimateIsInvalid) {
  EXPECT_FALSE(WrenchEstimateValid({0, 0, 0, 0, 0, 0}));
  EXPECT_FALSE(WrenchEstimateValid({1, 0, 0, 0, 0, NAN}));
  EXPECT_TRUE(WrenchEstimateValid({0, 0, -9.8, 0, 0, 0}));
}

TEST(Guide, DampingTermsArePassive) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const GuideParams p;
  for (int trial = 0; trial < 500; ++trial) {
    Vec7 q, dq;
    for (int i = 0; i < 7; ++i) {
      q[i] = Home()[i] + 0.6 * u(rng);
      dq[i] = 0.8 * u(rng);
    }
    const Mat67 J = Fr3Jacobian(q);
    Vec3 n(u(rng), u(rng), u(rng));
    n.normalize();
    const double level = 0.5 * (u(rng) + 1.0);
    EXPECT_LE(dq.dot(JointDampingTorque(dq, p.damping)), 1e-12);
    EXPECT_LE(dq.dot(GovernorTorque(dq, Vec7::Constant(-0.35),
                                    Vec7::Constant(0.35), p.governor_damping)),
              1e-12);
    EXPECT_LE(dq.dot(DamperWallTorque(J.topRows<3>(), J.bottomRows<3>(), dq,
                                      level, true, n, p.wall_damping,
                                      p.wall_rotational_damping)),
              1e-9);
    EXPECT_LE(dq.dot(DamperWallTorque(J.topRows<3>(), J.bottomRows<3>(), dq,
                                      level, false, n, p.wall_damping,
                                      p.wall_rotational_damping)),
              1e-9);
  }
}

TEST(Guide, WallOnlyResistsMotionAlongItsDirection) {
  const Vec7 q = Home();
  const Mat67 J = Fr3Jacobian(q);
  const Mat37 J_lin = J.topRows<3>();
  // Joint velocity that moves the flange along +x.
  const Vec3 v(0.05, 0.0, 0.0);
  const Vec7 dq =
      J_lin.transpose() * (J_lin * J_lin.transpose()).ldlt().solve(v);
  const Vec3 x = Vec3::UnitX();
  const Vec7 against =
      DamperWallTorque(J_lin, J.bottomRows<3>(), dq, 1.0, true, x, 100.0, 2.0);
  const Vec7 along =
      DamperWallTorque(J_lin, J.bottomRows<3>(), dq, 1.0, true, -x, 100.0, 2.0);
  EXPECT_LT(dq.dot(against), -1e-4);   // moving into +x is resisted
  EXPECT_DOUBLE_EQ(along.norm(), 0.0); // backing off along -x is free
  EXPECT_DOUBLE_EQ(
      DamperWallTorque(J_lin, J.bottomRows<3>(), dq, 0.0, true, x, 100.0, 2.0)
          .norm(),
      0.0);
}

TEST(Guide, GovernorIsZeroInsideTheBandAndContinuousAtItsEdge) {
  const Vec7 d = Vec7::Constant(10.0);
  EXPECT_DOUBLE_EQ(GovernorTorque(Vec7::Constant(0.3), Vec7::Constant(-0.35),
                                  Vec7::Constant(0.35), d)
                       .norm(),
                   0.0);
  const Vec7 edge =
      GovernorTorque(Vec7::Constant(0.35 + 1e-9), Vec7::Constant(-0.35),
                     Vec7::Constant(0.35), d);
  EXPECT_LT(edge.cwiseAbs().maxCoeff(), 1e-6);
  const Vec7 over = GovernorTorque(Vec7::Constant(0.55), Vec7::Constant(-0.35),
                                   Vec7::Constant(0.35), d);
  EXPECT_NEAR(over[0], -2.0, 1e-9);
}

TEST(Guide, JointLimitZonePushesAwayAndIsContinuous) {
  const GuideParams p;
  Vec7 q = Home();
  EXPECT_DOUBLE_EQ(JointLimitTorque(q, Vec7::Zero(), kLower, kUpper,
                                    p.limit_zone, p.limit_torque,
                                    p.limit_damping, p.limit_wall_margin,
                                    p.hold_kp, p.hold_kd)
                       .norm(),
                   0.0);
  // At the zone edge of joint 4's upper limit the spring is still zero.
  q[3] = kUpper[3] - p.limit_zone[3];
  EXPECT_NEAR(JointLimitTorque(q, Vec7::Zero(), kLower, kUpper, p.limit_zone,
                               p.limit_torque, p.limit_damping,
                               p.limit_wall_margin, p.hold_kp, p.hold_kd)[3],
              0.0, 1e-9);
  // Halfway in, it pushes back toward the lower side with half the torque.
  q[3] = kUpper[3] - 0.5 * p.limit_zone[3];
  const Vec7 tau = JointLimitTorque(
      q, Vec7::Zero(), kLower, kUpper, p.limit_zone, p.limit_torque,
      p.limit_damping, p.limit_wall_margin, p.hold_kp, p.hold_kd);
  EXPECT_NEAR(tau[3], -0.5 * p.limit_torque[3], 1e-9);
  // Moving away from the limit is not damped; moving toward it is.
  Vec7 away = Vec7::Zero(), toward = Vec7::Zero();
  away[3] = -0.2;
  toward[3] = 0.2;
  EXPECT_NEAR(JointLimitTorque(q, away, kLower, kUpper, p.limit_zone,
                               p.limit_torque, p.limit_damping,
                               p.limit_wall_margin, p.hold_kp, p.hold_kd)[3],
              tau[3], 1e-12);
  EXPECT_LT(JointLimitTorque(q, toward, kLower, kUpper, p.limit_zone,
                             p.limit_torque, p.limit_damping,
                             p.limit_wall_margin, p.hold_kp, p.hold_kd)[3],
            tau[3]);
}

TEST(Guide, FreeModeStaysFreeWhileUpdatesArrive) {
  const GuideParams p;
  GuideState s;
  const GuideInputs in = FreeInputs(Home(), Vec7::Constant(0.1));
  for (int tick = 0; tick < 100; ++tick) {
    const Vec7 tau = GuideStep(p, in, &s);
    EXPECT_EQ(s.phase, GuidePhase::kFree);
    EXPECT_LT((tau - FreeTorque(p, in, s.level)).norm(), 1e-12);
  }
}

namespace {

// Simulates the arm moving at 0.3 rad/s when updates stop, and returns the
// largest per-tick torque change and the drift from the latched pose.
struct HoldRun {
  double max_torque_step = 0.0;
  double drift_after_latch = 0.0;
  GuideState state;
};

HoldRun RunWatchdogHold(const GuideParams &p, int ticks) {
  HoldRun run;
  Vec7 q = Home();
  Vec7 dq = Vec7::Constant(0.3);
  Vec7 previous = Vec7::Zero();
  bool first = true;
  for (int tick = 0; tick < ticks; ++tick) {
    GuideInputs in = FreeInputs(q, dq);
    in.update_fresh = false;
    const Vec7 tau = GuideStep(p, in, &run.state);
    if (!first) {
      run.max_torque_step =
          std::max(run.max_torque_step, (tau - previous).cwiseAbs().maxCoeff());
    }
    first = false;
    previous = tau;
    dq += tau.cwiseQuotient(kInertia) * in.dt;
    q += dq * in.dt;
    if (run.state.phase == GuidePhase::kHold) {
      run.drift_after_latch = std::max(
          run.drift_after_latch, (q - run.state.q_hold).cwiseAbs().maxCoeff());
    }
  }
  return run;
}

} // namespace

TEST(Guide, WatchdogDampsToAStopThenHolds) {
  const GuideParams p;
  const HoldRun run = RunWatchdogHold(p, 1500);
  EXPECT_EQ(run.state.phase, GuidePhase::kHold);
  EXPECT_EQ(run.state.reason, HoldReason::kWatchdog);
  // Damping first means the latch happens near rest: no spring-back.
  EXPECT_LT(run.drift_after_latch, 0.005);
}

TEST(Guide, HoldEntryHasNoTorqueStepBeyondTheFirstDampingTick) {
  const GuideParams p;
  const HoldRun run = RunWatchdogHold(p, 1500);
  // The only jump is the damping switching on at 0.3 rad/s
  // (50 N m s/rad * 0.3 rad/s = 15 N m on the large joints); the latch and the
  // stiffness ramp add nothing comparable.
  EXPECT_LT(run.max_torque_step, 2.0);
}

TEST(Guide, ResumeOnlyWhenAllowedAndRampsOut) {
  const GuideParams p;
  GuideState s;
  GuideInputs in = FreeInputs(Home(), Vec7::Zero());
  in.update_fresh = false;
  for (int tick = 0; tick < 800; ++tick) {
    GuideStep(p, in, &s);
  }
  ASSERT_EQ(s.phase, GuidePhase::kHold);

  // A resume with updates still missing is ignored.
  in.resume_requested = true;
  GuideStep(p, in, &s);
  EXPECT_EQ(s.phase, GuidePhase::kHold);

  // Fresh updates and a low level allow it; the hold torque ramps out.
  in.update_fresh = true;
  in.level = 0.0;
  GuideStep(p, in, &s);
  EXPECT_EQ(s.phase, GuidePhase::kRelease);
  in.resume_requested = false;
  for (int tick = 0; tick < 250; ++tick) {
    GuideStep(p, in, &s);
  }
  EXPECT_EQ(s.phase, GuidePhase::kFree);
}

TEST(Guide, GelLevelAndValidityTriggerHolds) {
  const GuideParams p;
  {
    GuideState s;
    GuideInputs in = FreeInputs(Home(), Vec7::Zero());
    in.level = 1.0;
    GuideStep(p, in, &s);
    EXPECT_EQ(s.reason, HoldReason::kGelLimit);
  }
  {
    GuideState s;
    GuideInputs in = FreeInputs(Home(), Vec7::Zero());
    in.level_valid = false;
    GuideStep(p, in, &s);
    EXPECT_EQ(s.reason, HoldReason::kGelInvalid);
  }
  {
    GuideState s;
    GuideInputs in = FreeInputs(Home(), Vec7::Zero());
    in.level = NAN;
    GuideStep(p, in, &s);
    EXPECT_EQ(s.reason, HoldReason::kGelInvalid);
  }
}

TEST(Guide, JointLimitWallStopsAPushShortOfTheLimitWithoutHolding) {
  const GuideParams p;
  Vec7 q = Home();
  q[3] = kUpper[3] - 0.3;
  Vec7 dq = Vec7::Zero();
  Vec7 push = Vec7::Zero();
  push[3] = 10.0; // an operator keeps pushing joint 4 toward its upper limit
  GuideState s;
  double closest = 1.0;
  for (int tick = 0; tick < 3000; ++tick) {
    const GuideInputs in = FreeInputs(q, dq);
    const Vec7 tau = GuideStep(p, in, &s);
    EXPECT_TRUE(s.reason == HoldReason::kNone || s.reason == HoldReason::kIdle);
    dq += (tau + push).cwiseQuotient(kInertia) * in.dt;
    q += dq * in.dt;
    closest = std::min(closest, kUpper[3] - q[3]);
  }
  EXPECT_GT(closest, 0.02); // never reaches Franka's hard limit

  // Pulling the other way frees it: nothing needs a resume.
  push[3] = -10.0;
  for (int tick = 0; tick < 1500; ++tick) {
    const GuideInputs in = FreeInputs(q, dq);
    const Vec7 tau = GuideStep(p, in, &s);
    dq += (tau + push).cwiseQuotient(kInertia) * in.dt;
    q += dq * in.dt;
  }
  EXPECT_GT(kUpper[3] - q[3], p.limit_wall_margin);
}

TEST(Guide, JointLimitWallIsContinuousAtItsMargin) {
  const GuideParams p;
  Vec7 outside = Home(), inside = Home();
  outside[3] = kUpper[3] - p.limit_wall_margin - 1e-9;
  inside[3] = kUpper[3] - p.limit_wall_margin + 1e-9;
  const auto torque = [&](const Vec7 &q) {
    return JointLimitTorque(q, Vec7::Zero(), kLower, kUpper, p.limit_zone,
                            p.limit_torque, p.limit_damping,
                            p.limit_wall_margin, p.hold_kp, p.hold_kd)[3];
  };
  EXPECT_NEAR(torque(outside), torque(inside), 1e-6);
}

TEST(Guide, StopOverridesResume) {
  const GuideParams p;
  GuideState s;
  GuideInputs in = FreeInputs(Home(), Vec7::Zero());
  in.stop_requested = true;
  for (int tick = 0; tick < 800; ++tick) {
    GuideStep(p, in, &s);
  }
  EXPECT_EQ(s.phase, GuidePhase::kHold);
  EXPECT_EQ(s.reason, HoldReason::kStopping);
  in.resume_requested = true;
  GuideStep(p, in, &s);
  EXPECT_EQ(s.phase, GuidePhase::kHold);
}

TEST(Guide, FrictionIsPassiveAndSaturates) {
  const GuideParams p;
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int trial = 0; trial < 200; ++trial) {
    Vec7 dq;
    for (int i = 0; i < 7; ++i) {
      dq[i] = u(rng);
    }
    EXPECT_LE(dq.dot(FrictionTorque(dq, p.friction, p.friction_velocity)), 0.0);
  }
  const Vec7 fast =
      FrictionTorque(Vec7::Constant(0.5), p.friction, p.friction_velocity);
  EXPECT_NEAR(fast[0], -p.friction[0], 1e-6);
}

namespace {

// Simulates a released arm (zero velocity) under a constant external torque,
// e.g. gravity-model bias or an operator's push.
GuideState RunWithExternalTorque(const GuideParams &p, const Vec7 &external,
                                 int ticks, Vec7 *q_out, double *peak_speed,
                                 GuideState state = GuideState(),
                                 Vec7 q = Home()) {
  Vec7 dq = Vec7::Zero();
  *peak_speed = 0.0;
  for (int tick = 0; tick < ticks; ++tick) {
    const GuideInputs in = FreeInputs(q, dq);
    const Vec7 tau = GuideStep(p, in, &state);
    dq += (tau + external).cwiseQuotient(kInertia) * in.dt;
    q += dq * in.dt;
    *peak_speed = std::max(*peak_speed, dq.cwiseAbs().maxCoeff());
  }
  *q_out = q;
  return state;
}

} // namespace

TEST(Guide, ReleasedArmWithSmallBiasCreepsThenIdleHolds) {
  const GuideParams p;
  Vec7 bias = Vec7::Zero();
  bias[1] = 1.0; // 1 N m on the shoulder: a realistic gravity-model error
  Vec7 q_end;
  double peak = 0.0;
  const GuideState s = RunWithExternalTorque(p, bias, 3000, &q_end, &peak);
  EXPECT_EQ(s.phase, GuidePhase::kHold);
  EXPECT_EQ(s.reason, HoldReason::kIdle);
  EXPECT_LT(peak, p.idle_velocity); // a slow creep, not a drift
  // Held: the bias only deflects the hold spring, far below the breakaway.
  EXPECT_LT((q_end - s.q_hold).cwiseAbs().maxCoeff(), p.breakaway);
}

TEST(Guide, IdleHoldBreaksFreeWhenPushed) {
  const GuideParams p;
  Vec7 q_end;
  double peak = 0.0;
  GuideState s = RunWithExternalTorque(p, Vec7::Zero(), 2000, &q_end, &peak);
  ASSERT_EQ(s.reason, HoldReason::kIdle);
  ASSERT_EQ(s.phase, GuidePhase::kHold);
  Vec7 push = Vec7::Zero();
  push[0] = 6.0; // an operator pushing the base joint
  s = RunWithExternalTorque(p, push, 400, &q_end, &peak, s, q_end);
  EXPECT_EQ(s.phase, GuidePhase::kFree);
  EXPECT_EQ(s.reason, HoldReason::kNone);
}

TEST(Guide, SafetyReasonDuringIdleHoldNeedsAResume) {
  const GuideParams p;
  Vec7 q_end;
  double peak = 0.0;
  GuideState s = RunWithExternalTorque(p, Vec7::Zero(), 2000, &q_end, &peak);
  ASSERT_EQ(s.reason, HoldReason::kIdle);
  GuideInputs in = FreeInputs(q_end, Vec7::Zero());
  in.update_fresh = false; // the gel signal stops
  GuideStep(p, in, &s);
  EXPECT_EQ(s.reason, HoldReason::kWatchdog);
  // Pushing past the breakaway no longer frees it.
  in.q = s.q_hold + Vec7::Constant(0.02);
  GuideStep(p, in, &s);
  EXPECT_NE(s.phase, GuidePhase::kRelease);
}

TEST(Guide, GovernorBandShrinksWithFrankasLimits) {
  Vec7 low, high;
  GovernorBand({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2},
               {-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -0.1}, 0.35, 0.5, &low,
               &high);
  EXPECT_DOUBLE_EQ(high[0], 0.35);
  EXPECT_DOUBLE_EQ(low[0], -0.35);
  EXPECT_DOUBLE_EQ(high[6], 0.1);
  EXPECT_DOUBLE_EQ(low[6], -0.05);
}

// ---------------------------------------------------------------------------
// Speed leash and gel ratchet
// ---------------------------------------------------------------------------

namespace {

GuideInputs ContactInputs(const Vec7 &q, const Vec7 &dq) {
  GuideInputs in = FreeInputs(q, dq);
  in.contact_position = Fr3Chain(q).flange.block<3, 1>(0, 3);
  return in;
}

// The arm as a joint-space double integrator under GuideStep plus an external
// (operator) torque, with bookkeeping for torque steps and controller work.
struct Sim {
  Vec7 q = Home();
  Vec7 dq = Vec7::Zero();
  GuideState state;
  double level = 0.0;
  bool has_direction = false;
  Vec3 direction = Vec3::Zero();
  bool fresh = true;
  bool resume = false;
  Vec7 previous_tau = Vec7::Zero();
  bool started = false;
  double max_torque_step = 0.0;

  GuideInputs Inputs() const {
    GuideInputs in = ContactInputs(q, dq);
    in.level = level;
    in.has_direction = has_direction;
    in.direction = direction;
    in.update_fresh = fresh;
    in.resume_requested = resume;
    return in;
  }

  Vec3 Contact() const { return Fr3Chain(q).flange.block<3, 1>(0, 3); }

  // External torque of a force applied at the contact point.
  Vec7 ForceAtContact(const Vec3 &force) const {
    return Inputs().J_lin.transpose() * force;
  }

  Vec7 Step(const GuideParams &p, const Vec7 &external) {
    const GuideInputs in = Inputs();
    const Vec7 tau = GuideStep(p, in, &state);
    if (started) {
      max_torque_step =
          std::max(max_torque_step, (tau - previous_tau).cwiseAbs().maxCoeff());
    }
    started = true;
    previous_tau = tau;
    dq += (tau + external).cwiseQuotient(kInertia) * in.dt;
    q += dq * in.dt;
    return tau;
  }
};

GuideParams NoIdleHold() {
  GuideParams p;
  p.idle_time = 0.0;
  return p;
}

} // namespace

TEST(Leash, SlackBelowTheReplaySpeed) {
  const GuideParams p = NoIdleHold();
  Sim sim;
  Vec7 push = Vec7::Zero();
  push[0] = 2.6; // settles near 0.2 rad/s against damping and friction
  for (int tick = 0; tick < 3000; ++tick) {
    sim.Step(p, push);
  }
  EXPECT_NEAR(sim.dq[0], 0.2, 0.02);
  EXPECT_LT(sim.state.lead, 1e-9);
}

TEST(Leash, AHardPushStillAveragesTheReplaySpeed) {
  const GuideParams p = NoIdleHold();
  Vec7 push = Vec7::Zero();
  push[0] = 25.0; // a hard shove on the base joint
  push[4] = 8.0;  // and a wrist twist
  Sim sim;
  Vec7 q_after_1s;
  for (int tick = 0; tick < 3000; ++tick) {
    sim.Step(p, push);
    if (tick == 999) {
      q_after_1s = sim.q;
    }
  }
  const Vec7 average = (sim.q - q_after_1s) / 2.0;
  EXPECT_LE(average[0], 0.35 * 1.01);
  EXPECT_LE(average[4], 0.35 * 1.01);
  EXPECT_GT(average[0], 0.30); // held at the limit, not stopped
  EXPECT_LE(sim.state.lead,
            p.leash_max_torque.cwiseQuotient(p.leash_kp).maxCoeff() + 1e-9);

  // The same push without the leash runs well past the replay speed (checked
  // after 1 s, before the joint reaches its limit zone).
  GuideParams loose = p;
  loose.leash_max_torque.setZero();
  Sim free_run;
  for (int tick = 0; tick < 1000; ++tick) {
    free_run.Step(loose, push);
  }
  EXPECT_GT(free_run.dq[0], 0.8);
}

TEST(Leash, NeverStretchesPastItsMaxTorque) {
  const GuideParams p = NoIdleHold();
  Vec7 push = Vec7::Constant(80.0); // far beyond what it resists
  Sim sim;
  for (int tick = 0; tick < 2000; ++tick) {
    sim.Step(p, push);
    ASSERT_LE(sim.state.lead,
              p.leash_max_torque.cwiseQuotient(p.leash_kp).maxCoeff() + 1e-9);
  }
}

TEST(Leash, TakesUpAndLetsGoWithoutATorqueStep) {
  const GuideParams p = NoIdleHold();
  Sim sim;
  Vec7 push = Vec7::Zero();
  push[0] = 25.0;
  push[5] = 6.0;
  for (int tick = 0; tick < 1000; ++tick) {
    sim.Step(p, push);
  }
  ASSERT_GT(sim.state.lead, 0.005);
  for (int tick = 0; tick < 2000; ++tick) {
    sim.Step(p, Vec7::Zero()); // let go
  }
  EXPECT_LT(sim.max_torque_step, 2.0);
  EXPECT_LT(sim.state.lead, 1e-9);               // slack again
  EXPECT_LT(sim.dq.cwiseAbs().maxCoeff(), 0.05); // and coasting out
}

TEST(Leash, AddsNoEnergy) {
  // The leash alone on a randomly pushed arm: the work it does on the arm,
  // plus what its spring holds at the end, never exceeds zero.
  const GuideParams p;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const Vec7 scale = (Vec7() << 30, 30, 20, 20, 8, 6, 4).finished();
  Vec7 q = Home(), dq = Vec7::Zero(), shadow = Home(), push = Vec7::Zero();
  const double dt = 0.001;
  double work = 0.0, pushed = 0.0;
  for (int tick = 0; tick < 20000; ++tick) {
    if (tick % 50 == 0) {
      for (int i = 0; i < 7; ++i) {
        push[i] = scale[i] * u(rng);
      }
    }
    const Vec7 tau = LeashTorque(
        q, dq, Vec7::Constant(-0.35), Vec7::Constant(0.35), dt, p.leash_kp,
        p.leash_kd, p.leash_max_torque, p.leash_blend, &shadow, nullptr);
    const Vec7 dq_next =
        dq + (tau + push - 3.0 * dq).cwiseQuotient(kInertia) * dt;
    const Vec7 mid = 0.5 * (dq + dq_next);
    work += tau.dot(mid) * dt;
    pushed += std::abs(push.dot(mid)) * dt;
    dq = dq_next;
    q += dq * dt;
  }
  const Vec7 lead = q - shadow;
  const double stored = 0.5 * lead.cwiseProduct(lead).dot(p.leash_kp);
  EXPECT_LE(work + stored, 1e-3 * pushed);
}

TEST(GelRatchet, StopsLoadingButLetsTheArmBackOff) {
  const GuideParams p = NoIdleHold();
  Sim sim;
  sim.level = 1.0;
  sim.has_direction = true;
  sim.direction = Vec3(0.0, 0.0, -1.0); // pressing down loads the gel
  const double start_z = sim.Contact().z();
  for (int tick = 0; tick < 2000; ++tick) {
    sim.Step(p, sim.ForceAtContact(Vec3(0.0, 0.0, -20.0)));
  }
  EXPECT_TRUE(sim.state.ratchet_engaged);
  EXPECT_EQ(sim.state.phase, GuidePhase::kFree); // a wall, not a hold
  const double pressed = start_z - sim.Contact().z();
  EXPECT_LT(pressed, 1.5 * 20.0 / p.ratchet_stiffness); // ~7 mm of 20 N

  // Backing off is free, and the wall follows.
  for (int tick = 0; tick < 1000; ++tick) {
    sim.Step(p, sim.ForceAtContact(Vec3(0.0, 0.0, 15.0)));
  }
  const double backed_z = sim.Contact().z();
  EXPECT_GT(backed_z - start_z, 0.02);
  EXPECT_NEAR(sim.state.ratchet_anchor.z(), backed_z, 0.002);

  // Pressing again stops near the new anchor, not the old one.
  for (int tick = 0; tick < 2000; ++tick) {
    sim.Step(p, sim.ForceAtContact(Vec3(0.0, 0.0, -20.0)));
  }
  EXPECT_LT(backed_z - sim.Contact().z(), 1.5 * 20.0 / p.ratchet_stiffness);
}

TEST(GelRatchet, ReleasesOnlyBelowTheReleaseLevel) {
  const GuideParams p = NoIdleHold();
  Sim sim;
  sim.has_direction = true;
  sim.direction = Vec3(1.0, 0.0, 0.0);
  sim.level = 1.0;
  sim.Step(p, Vec7::Zero());
  ASSERT_TRUE(sim.state.ratchet_engaged);
  sim.level = 0.7;
  sim.Step(p, Vec7::Zero());
  EXPECT_TRUE(sim.state.ratchet_engaged);
  sim.level = 0.5;
  sim.Step(p, Vec7::Zero());
  EXPECT_FALSE(sim.state.ratchet_engaged);
}

TEST(GelRatchet, PressingOnItIsNotAnIdleHold) {
  const GuideParams p; // idle hold on
  Sim sim;
  sim.level = 1.0;
  sim.has_direction = true;
  sim.direction = Vec3(0.0, 0.0, -1.0);
  for (int tick = 0; tick < 4000; ++tick) {
    sim.Step(p, sim.ForceAtContact(Vec3(0.0, 0.0, -15.0)));
  }
  EXPECT_TRUE(sim.state.ratchet_engaged);
  EXPECT_EQ(sim.state.phase, GuidePhase::kFree);
}

TEST(GelRatchet, AddsNoEnergy) {
  const GuideParams p;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Vec7 q = Home(), dq = Vec7::Zero();
  Vec3 anchor = Fr3Chain(q).flange.block<3, 1>(0, 3);
  const Vec3 n(0.0, 0.6, -0.8);
  Vec3 force = Vec3::Zero();
  const double dt = 0.001;
  double work = 0.0, pushed = 0.0, depth = 0.0;
  for (int tick = 0; tick < 20000; ++tick) {
    if (tick % 40 == 0) {
      force = 30.0 * Vec3(u(rng), u(rng), u(rng));
    }
    const Mat37 J = Fr3Jacobian(q).topRows<3>();
    const Vec3 point = Fr3Chain(q).flange.block<3, 1>(0, 3);
    const Vec7 tau = RatchetTorque(J, dq, point, n, p.ratchet_stiffness,
                                   p.ratchet_damping, &anchor, &depth);
    const Vec7 push = J.transpose() * force;
    const Vec7 dq_next =
        dq + (tau + push - 3.0 * dq).cwiseQuotient(kInertia) * dt;
    const Vec7 mid = 0.5 * (dq + dq_next);
    work += tau.dot(mid) * dt;
    pushed += std::abs(push.dot(mid)) * dt;
    dq = dq_next;
    q += dq * dt;
  }
  const double stored = 0.5 * p.ratchet_stiffness * depth * depth;
  EXPECT_LE(work + stored, 1e-2 * pushed);
}

TEST(GelRatchet, ADirectedLimitDoesNotHold) {
  const GuideParams p;
  Sim sim;
  sim.level = 1.2;
  sim.has_direction = true;
  sim.direction = Vec3(0.0, 1.0, 0.0);
  sim.Step(p, Vec7::Zero());
  EXPECT_EQ(sim.state.phase, GuidePhase::kFree);
  EXPECT_EQ(sim.state.reason, HoldReason::kNone);
  EXPECT_TRUE(sim.state.ratchet_engaged);
}

TEST(GelRatchet, ResumingWithALoadedGelNeedsADirection) {
  const GuideParams p;
  for (const bool directed : {false, true}) {
    Sim sim;
    sim.fresh = false;
    for (int tick = 0; tick < 800; ++tick) {
      sim.Step(p, Vec7::Zero());
    }
    ASSERT_EQ(sim.state.phase, GuidePhase::kHold);
    sim.fresh = true;
    sim.level = 0.8;
    sim.has_direction = directed;
    sim.direction = Vec3(0.0, 0.0, -1.0);
    sim.resume = true;
    sim.Step(p, Vec7::Zero());
    EXPECT_EQ(sim.state.phase,
              directed ? GuidePhase::kRelease : GuidePhase::kHold);
  }
}

// ---------------------------------------------------------------------------
// Streaming contact limits
// ---------------------------------------------------------------------------

namespace {

const Vec7 kStreamKp =
    (Vec7() << 875.0, 1050.0, 1050.0, 875.0, 175.0, 350.0, 87.5).finished();

} // namespace

TEST(Contact, CommandedPushIsTheBlockedArmsContactForce) {
  // Pin the contact point and let the arm settle under joint stiffness kp
  // toward a reference offset e: it relaxes in the motions the pin allows
  // (J dq = 0) and the pin carries J^T F = kp (e - dq). That F must be what
  // CommandedPush predicts from the reference torque kp e alone.
  const Mat37 J = Fr3Jacobian(Home()).topRows<3>();
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(-0.01, 0.01);
  for (int trial = 0; trial < 20; ++trial) {
    Vec7 e;
    for (int i = 0; i < 7; ++i) {
      e[i] = u(rng);
    }
    Eigen::JacobiSVD<Mat37> svd(J, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 7, 4> N = svd.matrixV().rightCols<4>();
    const Eigen::Matrix<double, 4, 4> reduced =
        N.transpose() * kStreamKp.asDiagonal() * N;
    const Vec7 relaxed =
        N * reduced.ldlt().solve(N.transpose() * kStreamKp.cwiseProduct(e));
    const Vec7 residual = kStreamKp.cwiseProduct(e - relaxed);
    const Vec3 pin_force = (J * J.transpose()).ldlt().solve(J * residual);
    const Vec3 predicted =
        CommandedPush(J, kStreamKp, kStreamKp.cwiseProduct(e));
    EXPECT_LT((predicted - pin_force).norm(), 1e-6 * (1.0 + pin_force.norm()));
  }
}

TEST(Contact, MotionTheContactDoesNotBlockPushesNothing) {
  const Mat37 J = Fr3Jacobian(Home()).topRows<3>();
  Eigen::JacobiSVD<Mat37> svd(J, Eigen::ComputeFullV);
  const Vec7 self_motion = svd.matrixV().col(6);
  const Vec3 push =
      CommandedPush(J, kStreamKp, kStreamKp.cwiseProduct(0.05 * self_motion));
  EXPECT_LT(push.norm(), 1e-6);
}

TEST(Contact, PushCapScaleHoldsTheLimitAndFailsClosed) {
  EXPECT_DOUBLE_EQ(PushCapScale(5.0, 10.0), 1.0);
  EXPECT_DOUBLE_EQ(PushCapScale(20.0, 10.0), 0.5);
  EXPECT_DOUBLE_EQ(PushCapScale(INFINITY, 10.0), 0.0);
  EXPECT_DOUBLE_EQ(PushCapScale(NAN, 10.0), 0.0);
  EXPECT_DOUBLE_EQ(PushCapScale(1e9, 0.0), 1.0); // no cap configured
  // Scaling the reference torque by the factor puts the push on the limit.
  const Mat37 J = Fr3Jacobian(Home()).topRows<3>();
  Vec7 e = Vec7::Zero();
  e[1] = 0.02; // a shoulder reference 0.02 rad into the contact
  const Vec7 tau = kStreamKp.cwiseProduct(e);
  const double push = CommandedPush(J, kStreamKp, tau).norm();
  ASSERT_GT(push, 10.0);
  const double scale = PushCapScale(push, 10.0);
  EXPECT_NEAR(CommandedPush(J, kStreamKp, scale * tau).norm(), 10.0, 1e-9);
}

TEST(Contact, PointSpeedScaleKeepsTheContactPointUnderItsLimit) {
  const Mat37 J = Fr3Jacobian(Home()).topRows<3>();
  const Vec7 dq = Vec7::Constant(0.35);
  const double speed = (J * dq).norm();
  ASSERT_GT(speed, 0.1);
  const double scale = PointSpeedScale(J, dq, 0.1);
  EXPECT_NEAR((J * (scale * dq)).norm(), 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(PointSpeedScale(J, dq, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(PointSpeedScale(J, 0.01 * dq, 0.1), 1.0);
}

TEST(Contact, ForceGuardPassesSpikesButNotSustainedOrHardLoads) {
  ForceGuardParams p;
  p.force = 15.0;
  p.impulse = 0.05;
  p.hard_force = 25.0;
  const auto wrench = [](double fz) {
    return std::array<double, 6>{0.1, 0.0, fz, 0.0, 0.0, 0.0};
  };
  double impulse = 0.0, force = 0.0;
  // 20 N for 5 ms is 0.025 N s over the limit: a spike, not a trip.
  for (int tick = 0; tick < 5; ++tick) {
    EXPECT_EQ(ForceGuardStep(p, wrench(20.0), 0.001, &impulse, &force),
              GuardTrip::kNone);
  }
  for (int tick = 0; tick < 100; ++tick) {
    ForceGuardStep(p, wrench(5.0), 0.001, &impulse, &force);
  }
  EXPECT_DOUBLE_EQ(impulse, 0.0); // drained
  // 17 N sustained trips once 2 N of excess has lasted 25 ms.
  int ticks = 0;
  while (ForceGuardStep(p, wrench(17.0), 0.001, &impulse, &force) ==
         GuardTrip::kNone) {
    ++ticks;
    ASSERT_LT(ticks, 1000);
  }
  EXPECT_NEAR(ticks, 25, 2);
  impulse = 0.0;
  EXPECT_EQ(ForceGuardStep(p, wrench(26.0), 0.001, &impulse, &force),
            GuardTrip::kHard);
  EXPECT_NEAR(force, 26.0, 1e-3);
  EXPECT_EQ(ForceGuardStep(p, {0, 0, 0, 0, 0, 0}, 0.001, &impulse, &force),
            GuardTrip::kInvalid);
  p.trip_on_invalid = false;
  EXPECT_EQ(ForceGuardStep(p, {0, 0, 0, 0, 0, 0}, 0.001, &impulse, &force),
            GuardTrip::kNone);
  ForceGuardParams off;
  EXPECT_EQ(ForceGuardStep(off, wrench(100.0), 0.001, &impulse, &force),
            GuardTrip::kNone);
}
