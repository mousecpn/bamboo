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
