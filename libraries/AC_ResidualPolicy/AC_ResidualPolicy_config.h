#pragma once

#include <AP_HAL/AP_HAL_Boards.h>

// Embedded residual attitude/thrust policy (about 110 KB of weights): SITL by default.
#ifndef HAL_RESIDUAL_POLICY_ENABLED
#define HAL_RESIDUAL_POLICY_ENABLED (CONFIG_HAL_BOARD == HAL_BOARD_SITL)
#endif
