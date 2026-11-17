// Residual attitude/thrust policy for the ArduPilot cascade (inference only).
//
// Reproduces the training observation exactly (80 inputs):
//   30  current frame: goal error, velocity, attitude (heading frame), body rates,
//       previous action, error integral, filtered acceleration, sin/cos heading error
//   44  the first 22 of those 0.1 s and 0.2 s ago (re-expressed in the current heading frame)
//    6  position controller targets: velocity and acceleration (heading frame)
// followed by a dense tanh network 80-128-128-4.
//
// All vectors here are in the training convention: world ENU, body FLU.
// Use the helpers below to convert from ArduPilot NED / FRD.
#pragma once

#include <cstddef>

namespace residual {

constexpr int kObs = 80;
constexpr int kAct = 4;
constexpr int kHistory = 16;            // 50 Hz frames kept (> 0.2 s + margin)

struct Inputs {
    float pos[3];          // m, ENU
    float vel[3];          // m/s, ENU
    float rot[9];          // body FLU -> world ENU, row-major
    float rates[3];        // rad/s, body FLU
    float goal[3];         // current position target, m, ENU
    float vel_target[3];   // position controller velocity target, m/s, ENU
    float acc_target[3];   // position controller acceleration target, m/s^2, ENU
    double dt;             // s since the previous call (0.02 nominal); double: drives history timing
};

struct Outputs {
    float angle_offset_frd[3];   // rad, body FRD, for AC_Residual::set_residual
    float thrust_fraction;       // multiplier offset, (1 + thrust_fraction)
    float raw[kAct];             // network output in [-1, 1] (training convention)
};

class Policy {
public:
    // yaw_reference_rad: heading the position controller holds (ENU yaw).
    void reset(float yaw_reference_rad);
    void step(const Inputs &in, Outputs &out);
    // For verification against the training code.
    const float *last_observation() const { return obs_; }

    // Limits used in training; the firmware clips again with RESID_ANG_MAX / RESID_THR_MAX.
    static constexpr float kAngleMaxRad = 0.26f;
    static constexpr float kThrustMax = 0.15f;

private:
    struct Frame { double t; double base[22]; double heading[9]; };
    void observe(const Inputs &in);

    // double: the training code accumulates these in float64 (time drives history selection)
    double yaw_ref_ = 0.;
    double integral_[3] = {0, 0, 0};
    double accel_[3] = {0, 0, 0};
    double prev_vel_[3] = {0, 0, 0};
    bool have_prev_vel_ = false;
    double time_ = 0.;
    float prev_action_[kAct] = {0, 0, 0, 0};
    Frame hist_[kHistory];
    int hist_len_ = 0;
    float obs_[kObs];
};

// ArduPilot NED/FRD -> training ENU/FLU.
void ned_to_enu(const float ned[3], float enu[3]);
void frd_rates_to_flu(const float frd[3], float flu[3]);
// dcm_ned_frd: body FRD -> world NED, row-major (e.g. from Quaternion::rotation_matrix()).
void dcm_ned_frd_to_enu_flu(const float dcm_ned_frd[9], float rot_enu_flu[9]);

}  // namespace residual
