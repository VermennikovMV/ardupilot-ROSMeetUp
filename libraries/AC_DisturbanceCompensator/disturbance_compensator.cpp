#include "AC_DisturbanceCompensator_config.h"

#if HAL_DISTURBANCE_COMPENSATOR_ENABLED

#include "disturbance_compensator.h"
#include "estimator_weights.h"

#include <cmath>

namespace disturbance {

namespace {
float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
constexpr float kOutputScaleMax = 2.f;   // 30 deg, 30 % of thrust
}

constexpr double Compensator::kThrustTau[2];

void Compensator::reset(float yaw_reference_rad)
{
    encoder_.reset(yaw_reference_rad);
    thrust_started_ = false;
    thrust_ema_[0] = thrust_ema_[1] = 1.0;
}

void Compensator::step(const CompInputs &in, CompOutputs &out)
{
    encoder_.observe(in.state);
    const float *obs = encoder_.last_observation();

    // filtered thrust command
    const double t = in.thrust_ratio;
    if (!thrust_started_) {
        thrust_ema_[0] = thrust_ema_[1] = t;
        thrust_started_ = true;
    } else {
        for (int i = 0; i < 2; i++) {
            const double a = 1.0 - std::exp(-in.state.dt / kThrustTau[i]);
            thrust_ema_[i] += a * (t - thrust_ema_[i]);
        }
    }

    for (int i = 0; i < kObs; i++) { x_[i] = obs[i]; }
    x_[80] = (t - 1.0) * 5.0;
    x_[81] = (thrust_ema_[0] - 1.0) * 5.0;
    x_[82] = (thrust_ema_[1] - 1.0) * 5.0;

    // estimator MLP (double precision, as in the training code)
    double h1[est::kHidden1], h2[est::kHidden2], e[est::kOut];
    for (int i = 0; i < est::kHidden1; i++) {
        double s = est::b0[i];
        for (int j = 0; j < 83; j++) { s += (double)est::W0[i][j] * x_[j]; }
        h1[i] = std::tanh(s);
    }
    for (int i = 0; i < est::kHidden2; i++) {
        double s = est::b1[i];
        for (int j = 0; j < est::kHidden1; j++) { s += (double)est::W1[i][j] * h1[j]; }
        h2[i] = std::tanh(s);
    }
    for (int i = 0; i < est::kOut; i++) {
        double s = est::b2[i];
        for (int j = 0; j < est::kHidden2; j++) { s += (double)est::W2[i][j] * h2[j]; }
        e[i] = s * est::y_std[i] + est::y_mean[i];
        out.estimate[i] = (float)e[i];
    }

    // physics compensation, normalised by the training limits (15 deg, 15 %); up to 2x that is passed
    // on, the effective limit is set by RESID_ANG_MAX / RESID_THR_MAX in AC_Residual
    float c[kAct];
    c[0] = clampf((float)(kKxy * e[1] / kG / ObservationEncoder::kAngleMaxRad), -kOutputScaleMax, kOutputScaleMax);
    c[1] = clampf((float)(-kKxy * e[0] / kG / ObservationEncoder::kAngleMaxRad), -kOutputScaleMax, kOutputScaleMax);
    c[2] = 0.f;
    c[3] = clampf((float)(-kKz * e[2] / kG / ObservationEncoder::kThrustMax), -kOutputScaleMax, kOutputScaleMax);
    for (int i = 0; i < kAct; i++) { out.raw[i] = c[i]; }
    encoder_.set_previous_action(c);   // the applied action is part of the next observation

    // FLU (training) -> FRD (ArduPilot)
    out.angle_offset_frd[0] = c[0] * ObservationEncoder::kAngleMaxRad;
    out.angle_offset_frd[1] = -c[1] * ObservationEncoder::kAngleMaxRad;
    out.angle_offset_frd[2] = -c[2] * ObservationEncoder::kAngleMaxRad;
    out.thrust_fraction = c[3] * ObservationEncoder::kThrustMax;
}

}  // namespace disturbance

#endif  // HAL_DISTURBANCE_COMPENSATOR_ENABLED
