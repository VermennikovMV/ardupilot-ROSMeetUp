#include "AC_Residual.h"
#include <AP_HAL/AP_HAL.h>

AC_Residual *AC_Residual::_singleton;

const AP_Param::GroupInfo AC_Residual::var_info[] = {
    // @Param: ENABLE
    // @DisplayName: Residual correction enable
    // @Description: 0 keeps the stock controller; 1 accepts bounded angle and thrust corrections
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO_FLAGS("ENABLE", 1, AC_Residual, _enable, 0, AP_PARAM_FLAG_ENABLE),

    // @Param: ANG_MAX
    // @DisplayName: Residual angle limit
    // @Description: Limit applied to each of the roll, pitch and yaw angle offsets
    // @Units: deg
    // @Range: 0 30
    // @User: Advanced
    AP_GROUPINFO("ANG_MAX", 2, AC_Residual, _ang_max_deg, 15.0f),

    // @Param: THR_MAX
    // @DisplayName: Residual thrust fraction limit
    // @Range: 0 0.5
    // @User: Advanced
    AP_GROUPINFO("THR_MAX", 3, AC_Residual, _thr_max, 0.15f),

    // @Param: TST_R
    // @DisplayName: Constant test roll offset
    // @Description: Non-zero overrides the input with a constant roll offset (still clipped)
    // @Units: deg
    // @User: Advanced
    AP_GROUPINFO("TST_R", 4, AC_Residual, _test_roll_deg, 0.0f),

    // @Param: TST_P
    // @DisplayName: Constant test pitch offset
    // @Description: Non-zero overrides the input with a constant pitch offset (still clipped)
    // @Units: deg
    // @User: Advanced
    AP_GROUPINFO("TST_P", 5, AC_Residual, _test_pitch_deg, 0.0f),

    // @Param: MODES
    // @DisplayName: Flight modes accepting the residual correction
    // @Description: Bitmask of vehicle flight mode numbers (bit N = mode number N). In any other mode the correction is zero. Default: GUIDED (mode 4) only
    // @Bitmask: 0:Stabilize,2:AltHold,3:Auto,4:Guided,5:Loiter
    // @User: Advanced
    AP_GROUPINFO("MODES", 6, AC_Residual, _modes, (1U << 4)),

    // @Param: TST_T
    // @DisplayName: Constant test thrust offset
    // @Description: Non-zero overrides the input with a constant collective thrust fraction (still clipped by THR_MAX)
    // @Range: -0.5 0.5
    // @User: Advanced
    AP_GROUPINFO("TST_T", 7, AC_Residual, _test_thr, 0.0f),

    AP_GROUPEND
};

AC_Residual::AC_Residual()
{
    AP_Param::setup_object_defaults(this, var_info);
    _thr = 0.0f;
    _last_update_ms = 0;
    _mode_number = 0;
    _singleton = this;
}

void AC_Residual::set_residual(const Vector3f &ang_rad_body, float thr)
{
    if (ang_rad_body.is_nan() || isnan(thr) || isinf(thr)) {
        return;
    }
    _ang_rad = ang_rad_body;
    _thr = thr;
    _last_update_ms = AP_HAL::millis();
}

void AC_Residual::set_flight_mode(uint8_t mode_number)
{
    if (mode_number != _mode_number) {
        // never carry a correction across a mode change
        _ang_rad.zero();
        _thr = 0.0f;
        _last_update_ms = 0;
    }
    _mode_number = mode_number;
}

void AC_Residual::get_residual(Vector3f &ang_rad_body, float &thr) const
{
    ang_rad_body.zero();
    thr = 0.0f;
    if (!enabled() || _mode_number >= 32 || (uint32_t(_modes.get()) & (1UL << _mode_number)) == 0) {
        return;
    }
    const bool test = !is_zero(_test_roll_deg.get()) || !is_zero(_test_pitch_deg.get()) || !is_zero(_test_thr.get());
    if (test) {
        ang_rad_body = Vector3f(radians(_test_roll_deg.get()), radians(_test_pitch_deg.get()), 0.0f);
        thr = _test_thr.get();
    } else if (_last_update_ms != 0 && AP_HAL::millis() - _last_update_ms <= RESID_TIMEOUT_MS) {
        ang_rad_body = _ang_rad;
        thr = _thr;
    }
    const float ang_max = radians(constrain_float(_ang_max_deg.get(), 0.0f, 30.0f));
    const float thr_max = constrain_float(_thr_max.get(), 0.0f, 0.5f);
    ang_rad_body.x = constrain_float(ang_rad_body.x, -ang_max, ang_max);
    ang_rad_body.y = constrain_float(ang_rad_body.y, -ang_max, ang_max);
    ang_rad_body.z = constrain_float(ang_rad_body.z, -ang_max, ang_max);
    thr = constrain_float(thr, -thr_max, thr_max);
}
