import math

import numpy as np

LEG_MOUNT_ANGLES = tuple(math.radians(30.0 + 60.0 * i) for i in range(6))

COXA_RADIUS = math.hypot(0.115181, 0.0665)
COXA_HEIGHT = 0.03555

FEMUR_JOINT_COXA = np.array([0.048342, 0.006033, -0.018004])
FEMUR_VECTOR_COXA = np.array([0.077851, 0.044948, 0.0])
TIBIA_FOOT_TIBIA = np.array([0.021830, 0.034581, -0.085801])

_COXA_MOUNT_ANGLE = LEG_MOUNT_ANGLES[0]
_COS_MOUNT = math.cos(_COXA_MOUNT_ANGLE)
_SIN_MOUNT = math.sin(_COXA_MOUNT_ANGLE)


def _rotate_xy(x, y, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def _to_mount(x, y, leg):
    angle = -LEG_MOUNT_ANGLES[leg]
    return _rotate_xy(x, y, angle)


def _from_mount(x, y, leg):
    angle = LEG_MOUNT_ANGLES[leg]
    return _rotate_xy(x, y, angle)


def _mount_geometry():
    fx, fy = _rotate_xy(FEMUR_JOINT_COXA[0], FEMUR_JOINT_COXA[1], -_COXA_MOUNT_ANGLE)
    vx, vy = _rotate_xy(FEMUR_VECTOR_COXA[0], FEMUR_VECTOR_COXA[1], -_COXA_MOUNT_ANGLE)
    wx, wy = _rotate_xy(TIBIA_FOOT_TIBIA[0], TIBIA_FOOT_TIBIA[1], -_COXA_MOUNT_ANGLE)
    return fx, fy, FEMUR_JOINT_COXA[2], vx, vy, FEMUR_VECTOR_COXA[2], wx, wy, TIBIA_FOOT_TIBIA[2]


FEMUR_PLANAR_X, FEMUR_TANGENTIAL_Y, FEMUR_PLANAR_Z, \
    FEMUR_LENGTH, FEMUR_VECTOR_TANGENTIAL_Y, FEMUR_VECTOR_Z, \
    TIBIA_PLANAR_X, TIBIA_TANGENTIAL_Y, TIBIA_PLANAR_Z = _mount_geometry()

TIBIA_LENGTH = math.hypot(TIBIA_PLANAR_X, TIBIA_PLANAR_Z)
TIBIA_ANGLE = math.atan2(TIBIA_PLANAR_Z, TIBIA_PLANAR_X)
NOMINAL_RADIUS = COXA_RADIUS + FEMUR_PLANAR_X + FEMUR_LENGTH + TIBIA_LENGTH * math.cos(TIBIA_ANGLE)

COXA_LIMIT = 1.1
FEMUR_LIMIT_LOWER = -1.57
FEMUR_LIMIT_UPPER = 1.57
TIBIA_LIMIT_LOWER = -1.57
TIBIA_LIMIT_UPPER = 1.57

JOINT_NAMES = tuple(
    f'leg_{leg}_{name}_joint'
    for leg in range(6)
    for name in ('coxa', 'femur', 'tibia')
)


def coxa_position(leg):
    angle = LEG_MOUNT_ANGLES[leg]
    return np.array([
        COXA_RADIUS * math.cos(angle),
        COXA_RADIUS * math.sin(angle),
        COXA_HEIGHT,
    ])


def foot_from_joints(leg, q1, q2, q3):
    phi_t = TIBIA_ANGLE + q2 - q3
    px = FEMUR_PLANAR_X + FEMUR_LENGTH * math.cos(q2) + TIBIA_LENGTH * math.cos(phi_t)
    pz = FEMUR_PLANAR_Z + FEMUR_LENGTH * math.sin(q2) + TIBIA_LENGTH * math.sin(phi_t)

    chain_x, chain_y = _rotate_xy(px, 0.0, -q1)
    mount_x = COXA_RADIUS + chain_x
    mount_y = chain_y
    base_x, base_y = _from_mount(mount_x, mount_y, leg)
    return np.array([base_x, base_y, COXA_HEIGHT + pz])


def joints_from_foot(leg, foot):
    coxa = coxa_position(leg)
    mount_x, mount_y = _to_mount(foot[0] - coxa[0], foot[1] - coxa[1], leg)

    radius = math.hypot(mount_x, mount_y)
    q1 = -math.atan2(mount_y, mount_x)
    q1 = max(-COXA_LIMIT, min(COXA_LIMIT, q1))

    target_r = radius - FEMUR_PLANAR_X
    target_z = foot[2] - COXA_HEIGHT - FEMUR_PLANAR_Z

    distance = math.hypot(target_r, target_z)
    reach_min = abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-6
    reach_max = FEMUR_LENGTH + TIBIA_LENGTH - 1e-6
    clamped = max(reach_min, min(reach_max, distance))
    if distance > 1e-9 and clamped != distance:
        scale = clamped / distance
        target_r *= scale
        target_z *= scale
        distance = clamped

    gamma = math.atan2(target_z, target_r)
    cosine = (FEMUR_LENGTH * FEMUR_LENGTH + distance * distance -
              TIBIA_LENGTH * TIBIA_LENGTH) / (2.0 * FEMUR_LENGTH * distance)
    delta = math.acos(max(-1.0, min(1.0, cosine)))
    q2 = gamma + delta

    phi_t = math.atan2(
        target_z - FEMUR_LENGTH * math.sin(q2),
        target_r - FEMUR_LENGTH * math.cos(q2),
    )
    q3 = TIBIA_ANGLE + q2 - phi_t

    q2 = max(FEMUR_LIMIT_LOWER, min(FEMUR_LIMIT_UPPER, q2))
    q3 = max(TIBIA_LIMIT_LOWER, min(TIBIA_LIMIT_UPPER, q3))
    return q1, q2, q3


def nominal_foot(leg, height):
    angle = LEG_MOUNT_ANGLES[leg]
    return np.array([
        NOMINAL_RADIUS * math.cos(angle),
        NOMINAL_RADIUS * math.sin(angle),
        -height,
    ])
