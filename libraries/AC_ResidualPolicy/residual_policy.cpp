#include "AC_ResidualPolicy_config.h"

#if HAL_RESIDUAL_POLICY_ENABLED

#include "residual_policy.h"
#include "residual_weights.h"

#include <cmath>
#include <cstring>

namespace residual {

namespace {

// Observation encoder constants (training configuration).
constexpr double kIntegralLimit = 6.0;
constexpr double kAccelTau = 0.08;
constexpr double kPosScale = 3.0;
constexpr double kVelScale = 3.0;
constexpr double kRateScale = 3.0;
constexpr double kAccScale = 5.0;
constexpr double kLags[2] = {0.1, 0.2};
constexpr double kPidVelScale = 3.0;
constexpr double kPidAccScale = 5.0;

// heading matrix H (world -> heading frame) from yaw, row-major
void heading_matrix(double yaw, double h[9])
{
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double m[9] = {c, s, 0, -s, c, 0, 0, 0, 1};
    std::memcpy(h, m, sizeof(m));
}

void mul3(const double m[9], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++) {
        out[i] = m[3*i] * v[0] + m[3*i+1] * v[1] + m[3*i+2] * v[2];
    }
}

void mul33(const double a[9], const double b[9], double out[9])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[3*i+j] = a[3*i] * b[j] + a[3*i+1] * b[3+j] + a[3*i+2] * b[6+j];
        }
    }
}

// a * b^T
void mul33t(const double a[9], const double b[9], double out[9])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[3*i+j] = a[3*i] * b[3*j] + a[3*i+1] * b[3*j+1] + a[3*i+2] * b[3*j+2];
        }
    }
}

double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

}  // namespace

void Policy::reset(float yaw_reference_rad)
{
    yaw_ref_ = yaw_reference_rad;
    for (int i = 0; i < 3; i++) { integral_[i] = 0; accel_[i] = 0; prev_vel_[i] = 0; }
    have_prev_vel_ = false;
    time_ = 0;
    for (int i = 0; i < kAct; i++) { prev_action_[i] = 0; }
    hist_len_ = 0;
}

void Policy::observe(const Inputs &in)
{
    double pos[3], vel[3], rot[9], rates[3], goal[3];
    for (int i = 0; i < 3; i++) {
        pos[i] = in.pos[i]; vel[i] = in.vel[i]; rates[i] = in.rates[i]; goal[i] = in.goal[i];
    }
    for (int i = 0; i < 9; i++) { rot[i] = in.rot[i]; }
    const double dt = in.dt;

    double error[3];
    for (int i = 0; i < 3; i++) { error[i] = goal[i] - pos[i]; }

    // integral and filtered acceleration (skipped on the first call, as in training)
    double integ[3] = {integral_[0], integral_[1], integral_[2]};
    double acc[3] = {accel_[0], accel_[1], accel_[2]};
    double t = time_;
    if (have_prev_vel_) {
        for (int i = 0; i < 3; i++) { integ[i] += error[i] * dt; }
        const double len = std::sqrt(integ[0]*integ[0] + integ[1]*integ[1]);
        if (len > kIntegralLimit) { integ[0] *= kIntegralLimit / len; integ[1] *= kIntegralLimit / len; }
        integ[2] = clampd(integ[2], -kIntegralLimit, kIntegralLimit);
        const double alpha = 1.0 - std::exp(-dt / kAccelTau);
        for (int i = 0; i < 3; i++) {
            const double observed = (vel[i] - prev_vel_[i]) / dt;
            acc[i] += alpha * (observed - acc[i]);
        }
        t += dt;
    }
    for (int i = 0; i < 3; i++) { integral_[i] = integ[i]; accel_[i] = acc[i]; prev_vel_[i] = vel[i]; }
    have_prev_vel_ = true;
    time_ = t;

    const double yaw = std::atan2(rot[3], rot[0]);   // rot[1][0], rot[0][0]
    double h[9];
    heading_matrix(yaw, h);

    // base frame (22)
    double base[22];
    double tmp[3];
    mul3(h, error, tmp);
    for (int i = 0; i < 3; i++) { base[i] = tmp[i] / kPosScale; }
    mul3(h, vel, tmp);
    for (int i = 0; i < 3; i++) { base[3+i] = tmp[i] / kVelScale; }
    double hr[9];
    mul33(h, rot, hr);
    for (int i = 0; i < 9; i++) { base[6+i] = hr[i]; }
    for (int i = 0; i < 3; i++) { base[15+i] = rates[i] / kRateScale; }
    for (int i = 0; i < 4; i++) { base[18+i] = clampd(prev_action_[i], -1.0, 1.0); }

    int k = 0;
    for (int i = 0; i < 22; i++) { obs_[k++] = (float)base[i]; }
    mul3(h, integ, tmp);
    for (int i = 0; i < 3; i++) { obs_[k++] = (float)(tmp[i] / kIntegralLimit); }
    mul3(h, acc, tmp);
    for (int i = 0; i < 3; i++) { obs_[k++] = (float)(tmp[i] / kAccScale); }
    const double yaw_error = (double)yaw_ref_ - yaw;
    obs_[k++] = (float)std::sin(yaw_error);
    obs_[k++] = (float)std::cos(yaw_error);

    // history: append, then drop frames older than the longest lag (keep one before it)
    if (hist_len_ == kHistory) {
        std::memmove(&hist_[0], &hist_[1], sizeof(Frame) * (kHistory - 1));
        hist_len_--;
    }
    Frame &f = hist_[hist_len_++];
    f.t = t;
    for (int i = 0; i < 22; i++) { f.base[i] = base[i]; }
    for (int i = 0; i < 9; i++) { f.heading[i] = h[i]; }
    const double oldest = t - kLags[1];
    while (hist_len_ > 1 && hist_[1].t <= oldest) {
        std::memmove(&hist_[0], &hist_[1], sizeof(Frame) * (hist_len_ - 1));
        hist_len_--;
    }

    for (double lag : kLags) {
        int sel = 0;
        for (int j = 0; j < hist_len_; j++) {
            if (hist_[j].t <= t - lag + 1e-9) { sel = j; } else { break; }
        }
        const Frame &s = hist_[sel];
        double old_h[9], change[9];
        for (int i = 0; i < 9; i++) { old_h[i] = s.heading[i]; }
        mul33t(h, old_h, change);                 // H_now * H_old^T
        double v3[3], o3[3];
        for (int blk = 0; blk < 2; blk++) {       // error and velocity vectors
            for (int i = 0; i < 3; i++) { v3[i] = s.base[3*blk+i]; }
            mul3(change, v3, o3);
            for (int i = 0; i < 3; i++) { obs_[k++] = (float)o3[i]; }
        }
        double m[9], cm[9];
        for (int i = 0; i < 9; i++) { m[i] = s.base[6+i]; }
        mul33(change, m, cm);                     // attitude in the current heading frame
        for (int i = 0; i < 9; i++) { obs_[k++] = (float)cm[i]; }
        for (int i = 15; i < 22; i++) { obs_[k++] = (float)s.base[i]; }   // body rates and actions unchanged
    }

    // position controller targets (6)
    double pv[3], pa[3];
    for (int i = 0; i < 3; i++) { pv[i] = in.vel_target[i]; pa[i] = in.acc_target[i]; }
    mul3(h, pv, tmp);
    for (int i = 0; i < 3; i++) { obs_[k++] = (float)(tmp[i] / kPidVelScale); }
    mul3(h, pa, tmp);
    for (int i = 0; i < 3; i++) { obs_[k++] = (float)(tmp[i] / kPidAccScale); }
}

void Policy::step(const Inputs &in, Outputs &out)
{
    observe(in);

    float h1[weights::kHidden1], h2[weights::kHidden2], y[kAct];
    for (int i = 0; i < weights::kHidden1; i++) {
        float s = weights::b0[i];
        for (int j = 0; j < kObs; j++) { s += weights::W0[i][j] * obs_[j]; }
        h1[i] = std::tanh(s);
    }
    for (int i = 0; i < weights::kHidden2; i++) {
        float s = weights::b1[i];
        for (int j = 0; j < weights::kHidden1; j++) { s += weights::W1[i][j] * h1[j]; }
        h2[i] = std::tanh(s);
    }
    for (int i = 0; i < kAct; i++) {
        float s = weights::b2[i];
        for (int j = 0; j < weights::kHidden2; j++) { s += weights::W2[i][j] * h2[j]; }
        y[i] = s < -1.f ? -1.f : (s > 1.f ? 1.f : s);
        out.raw[i] = y[i];
        prev_action_[i] = y[i];
    }
    // FLU (training) -> FRD (ArduPilot): roll unchanged, pitch and yaw change sign
    out.angle_offset_frd[0] = y[0] * kAngleMaxRad;
    out.angle_offset_frd[1] = -y[1] * kAngleMaxRad;
    out.angle_offset_frd[2] = -y[2] * kAngleMaxRad;
    out.thrust_fraction = y[3] * kThrustMax;
}

void ned_to_enu(const float ned[3], float enu[3])
{
    enu[0] = ned[1]; enu[1] = ned[0]; enu[2] = -ned[2];
}

void frd_rates_to_flu(const float frd[3], float flu[3])
{
    flu[0] = frd[0]; flu[1] = -frd[1]; flu[2] = -frd[2];
}

void dcm_ned_frd_to_enu_flu(const float d[9], float r[9])
{
    // R_enu_flu = T * R_ned_frd * F, T = [[0,1,0],[1,0,0],[0,0,-1]], F = diag(1,-1,-1)
    const float tr[9] = {d[3], d[4], d[5], d[0], d[1], d[2], -d[6], -d[7], -d[8]};
    for (int i = 0; i < 3; i++) {
        r[3*i]   =  tr[3*i];
        r[3*i+1] = -tr[3*i+1];
        r[3*i+2] = -tr[3*i+2];
    }
}

}  // namespace residual

#endif  // HAL_RESIDUAL_POLICY_ENABLED
