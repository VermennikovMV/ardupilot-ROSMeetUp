#pragma once

/*
  Bounded residual correction for the multicopter attitude controller.

  The correction is an angle offset in the body frame (roll, pitch, yaw),
  added to the attitude error vector before the angle P controller, and a
  multiplicative collective thrust factor (1 + thr). Both are always clipped
  here, whatever the producer sends, and drop to zero when stale or disabled.
*/

#include <AP_Param/AP_Param.h>
#include <AP_Math/AP_Math.h>

class AC_Residual {
public:
    AC_Residual();

    CLASS_NO_COPY(AC_Residual);

    static AC_Residual *get_singleton() { return _singleton; }

    // set a new correction: angle offset in radians, body frame FRD (roll right, pitch nose up,
    // yaw clockwise), as used by attitude_error; thrust fraction. A FLU producer passes (r, -p, -y).
    void set_residual(const Vector3f &ang_rad_body, float thr);

    // current clipped correction; zero when disabled or not updated for RESID_TIMEOUT_MS
    void get_residual(Vector3f &ang_rad_body, float &thr) const;

    bool enabled() const { return _enable.get() != 0; }

    // report the vehicle flight mode number (RESID_MODES selects the modes that accept the correction)
    void set_flight_mode(uint8_t mode_number);

    static const struct AP_Param::GroupInfo var_info[];

private:
    static constexpr uint32_t RESID_TIMEOUT_MS = 100;

    AP_Int8 _enable;
    AP_Float _ang_max_deg;
    AP_Float _thr_max;
    AP_Int32 _modes;

    Vector3f _ang_rad;
    float _thr;
    uint32_t _last_update_ms;
    uint8_t _mode_number;

    static AC_Residual *_singleton;
};
