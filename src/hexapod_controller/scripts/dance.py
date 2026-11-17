#!/usr/bin/env python3

import math

import numpy as np

from hexapod_kinematics import joints_from_foot, nominal_foot

# One leg lifts after another while the body sways and twists.
WAVE_PERIOD = 1.4
SWAY_PERIOD = 2.8
TWIST_PERIOD = 3.6
BOB_PERIOD = 0.8

WAVE_LIFT_MAX = 0.030
WAVE_DUTY = 0.35
SWAY_AMPLITUDE = 0.024
SURGE_AMPLITUDE = 0.016
YAW_AMPLITUDE = 0.060
ROLL_AMPLITUDE = 0.055
BOB_AMPLITUDE = 0.007


def stand_joint_positions(body_height):
    positions = []
    for leg in range(6):
        q1, q2, q3 = joints_from_foot(leg, nominal_foot(leg, body_height))
        positions.extend((q1, q2, q3))
    return positions


def dance_joint_positions(t, body_height, step_height):
    wave_phase = t / WAVE_PERIOD
    sway = math.sin(2.0 * math.pi * t / SWAY_PERIOD)
    surge = math.cos(2.0 * math.pi * t / SWAY_PERIOD)
    yaw = YAW_AMPLITUDE * math.sin(2.0 * math.pi * t / TWIST_PERIOD)
    roll = ROLL_AMPLITUDE * math.sin(2.0 * math.pi * t / SWAY_PERIOD)
    bob = BOB_AMPLITUDE * math.sin(2.0 * math.pi * t / BOB_PERIOD)
    lift_max = max(0.015, min(WAVE_LIFT_MAX, float(step_height)))

    c_yaw = math.cos(yaw)
    s_yaw = math.sin(yaw)

    positions = []
    for leg in range(6):
        nominal = nominal_foot(leg, body_height)

        local = (wave_phase - leg / 6.0) % 1.0
        if local < WAVE_DUTY:
            pulse = math.sin(math.pi * local / WAVE_DUTY)
            lift = lift_max * pulse * pulse
        else:
            lift = 0.0

        x = nominal[0] + SURGE_AMPLITUDE * surge
        y = nominal[1] + SWAY_AMPLITUDE * sway
        x, y = c_yaw * x - s_yaw * y, s_yaw * x + c_yaw * y
        z = nominal[2] + lift + bob + roll * y

        q1, q2, q3 = joints_from_foot(leg, np.array([x, y, z]))
        positions.extend((q1, q2, q3))
    return positions
