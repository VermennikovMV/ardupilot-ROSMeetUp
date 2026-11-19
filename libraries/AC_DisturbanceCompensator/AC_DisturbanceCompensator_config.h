#pragma once

#include <AP_HAL/AP_HAL_Boards.h>

// Embedded disturbance compensator (about 110 KB of estimator weights): SITL by default.
#ifndef HAL_DISTURBANCE_COMPENSATOR_ENABLED
#define HAL_DISTURBANCE_COMPENSATOR_ENABLED (CONFIG_HAL_BOARD == HAL_BOARD_SITL)
#endif
