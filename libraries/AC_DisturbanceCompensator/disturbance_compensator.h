// Learned disturbance estimator + physics compensation for the ArduPilot cascade.
//
// Every control step (50 Hz):
//   1. build the 80 observation inputs (disturbance::ObservationEncoder);
//   2. filter the collective thrust command of the position controller
//      (relative to hover thrust) with 0.3 s and 1.5 s time constants;
//   3. estimator network 83 -> 128 -> 128 -> 3 gives the disturbance:
//      horizontal disturbance acceleration (heading frame x, y) and the vertical imbalance;
//   4. compensation: tilt = a_horizontal / g (towards the disturbance),
//      thrust fraction = -KZ * a_vertical / g;
//   5. output is bounded and returned in ArduPilot FRD for AC_Residual::set_residual.
#pragma once

#include "observation_encoder.h"

namespace disturbance {

struct CompInputs {
    Inputs state;              // vehicle state and targets (ENU / FLU)
    double thrust_ratio;       // collective thrust command / hover thrust (1.0 = hover)
};

struct CompOutputs {
    float angle_offset_frd[3]; // rad, body FRD
    float thrust_fraction;     // (1 + thrust_fraction) multiplier
    float estimate[3];         // a_x, a_y (m/s^2, heading frame), a_vertical (m/s^2)
    float raw[kAct];           // compensation in normalised units [-1, 1] (training convention)
};

class Compensator {
public:
    static constexpr double kG = 9.81;
    static constexpr double kKxy = 1.0;
    static constexpr double kKz = 0.5;
    static constexpr double kThrustTau[2] = {0.3, 1.5};

    void reset(float yaw_reference_rad);
    // heading the position controller holds (ENU yaw); may change between steps
    void set_yaw_reference(float yaw_reference_rad) { encoder_.set_yaw_reference(yaw_reference_rad); }
    void step(const CompInputs &in, CompOutputs &out);
    // For verification against the training code.
    const double *last_estimator_input() const { return x_; }

private:
    ObservationEncoder encoder_;
    bool thrust_started_ = false;
    double thrust_ema_[2] = {1.0, 1.0};
    double x_[83];
};

}  // namespace disturbance
