#!/usr/bin/env python3

import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from dance import dance_joint_positions, stand_joint_positions
from hexapod_kinematics import joints_from_foot, nominal_foot

CMD_TOPIC = '/hexapod/cmd_vel'
PARAMS_TOPIC = '/hexapod/gait_params'
ESTOP_TOPIC = '/hexapod/estop'
MODE_TOPIC = '/hexapod/mode'
COMMAND_TOPIC = '/joint_group_position_controller/commands'
TRIPOD_A = (0, 2, 4)

MODE_WALK = 'walk'
MODE_RUN = 'run'
MODE_DANCE = 'dance'
VALID_MODES = (MODE_WALK, MODE_RUN, MODE_DANCE)


def _smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _blend(start, end, alpha):
    return [a + (b - a) * alpha for a, b in zip(start, end)]


class HexapodGait(Node):
    def __init__(self):
        super().__init__('hexapod_gait')

        self.declare_parameter('frequency', 1.0)
        self.declare_parameter('duty', 0.5)
        self.declare_parameter('body_height', 0.068)
        self.declare_parameter('step_height', 0.03)
        self.declare_parameter('max_linear_speed', 0.08)
        self.declare_parameter('max_yaw_rate', 0.6)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('command_timeout', 0.5)
        self.declare_parameter('command_filter', 0.20)
        self.declare_parameter('activity_ramp', 0.20)
        self.declare_parameter('deadband_linear', 0.001)
        self.declare_parameter('deadband_yaw', 0.01)
        self.declare_parameter('run_linear_speed', 0.20)
        self.declare_parameter('run_frequency', 1.4)
        self.declare_parameter('run_step_height_limit', 0.025)
        self.declare_parameter('dance_ramp', 0.50)

        self._frequency = float(self.get_parameter('frequency').value)
        self._duty = float(self.get_parameter('duty').value)
        self._body_height = float(self.get_parameter('body_height').value)
        self._step_height = float(self.get_parameter('step_height').value)
        self._max_linear = float(self.get_parameter('max_linear_speed').value)
        self._max_yaw = float(self.get_parameter('max_yaw_rate').value)
        self._control_rate = float(self.get_parameter('control_rate').value)
        self._command_timeout = float(self.get_parameter('command_timeout').value)
        self._command_filter = float(self.get_parameter('command_filter').value)
        self._activity_ramp = float(self.get_parameter('activity_ramp').value)
        self._deadband_linear = float(self.get_parameter('deadband_linear').value)
        self._deadband_yaw = float(self.get_parameter('deadband_yaw').value)
        self._run_linear = float(self.get_parameter('run_linear_speed').value)
        self._run_frequency = float(self.get_parameter('run_frequency').value)
        self._run_step_limit = float(self.get_parameter('run_step_height_limit').value)
        self._dance_ramp = max(0.05, float(self.get_parameter('dance_ramp').value))

        self._lock = threading.Lock()
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._cmd_time = None
        self._estop = False
        self._mode = MODE_WALK

        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._activity = 0.0
        self._phase = 0.0
        self._last_time = None
        self._last_command = None
        self._last_positions = None
        self._dance_time = 0.0
        self._dance_weight = 0.0
        self._dance_start_pose = None

        self._publisher = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        self.create_subscription(Twist, CMD_TOPIC, self._on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, PARAMS_TOPIC, self._on_params, 10)
        self.create_subscription(Bool, ESTOP_TOPIC, self._on_estop, 10)
        self.create_subscription(String, MODE_TOPIC, self._on_mode, 10)
        self.create_timer(1.0 / self._control_rate, self._on_timer)

        self.get_logger().info(
            f'Gait ready on {COMMAND_TOPIC}: {self._frequency:.2f} Hz, '
            f'body {self._body_height:.3f} m, step {self._step_height:.3f} m; '
            f'run {self._run_linear:.2f} m/s, mode topic {MODE_TOPIC}'
        )

    def _on_cmd_vel(self, msg):
        with self._lock:
            self._cmd_vx = msg.linear.x
            self._cmd_vy = msg.linear.y
            self._cmd_wz = msg.angular.z
            self._cmd_time = self.get_clock().now()

    def _on_params(self, msg):
        if len(msg.data) < 3:
            return
        self._body_height = max(0.04, min(0.09, float(msg.data[0])))
        self._step_height = max(0.0, min(0.05, float(msg.data[1])))
        self._frequency = max(0.5, min(2.0, float(msg.data[2])))

    def _on_estop(self, msg):
        self._estop = bool(msg.data)

    def _on_mode(self, msg):
        mode = msg.data.strip().lower()
        if mode not in VALID_MODES or mode == self._mode:
            return

        self._mode = mode
        if mode == MODE_DANCE:
            with self._lock:
                self._cmd_vx = 0.0
                self._cmd_vy = 0.0
                self._cmd_wz = 0.0
            self._vx = 0.0
            self._vy = 0.0
            self._wz = 0.0
            self._activity = 0.0
            if self._last_positions is not None:
                self._dance_start_pose = list(self._last_positions)
            else:
                self._dance_start_pose = stand_joint_positions(self._body_height)
            self._dance_time = 0.0
            self._dance_weight = 0.0

        self.get_logger().info(f'Mode: {mode}')

    def _on_timer(self):
        now = self.get_clock().now()
        if self._last_time is None:
            dt = 0.0
        else:
            dt = (now - self._last_time).nanoseconds * 1e-9
            dt = max(0.0, min(0.1, dt))
        self._last_time = now

        if self._estop:
            if self._last_command is not None:
                self._publisher.publish(self._last_command)
            return

        with self._lock:
            cmd_vx, cmd_vy, cmd_wz = self._cmd_vx, self._cmd_vy, self._cmd_wz
            cmd_time = self._cmd_time
            mode = self._mode

        if cmd_time is None or (now - cmd_time).nanoseconds * 1e-9 > self._command_timeout:
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_wz = 0.0

        if mode == MODE_RUN:
            max_linear = self._run_linear
            frequency = min(2.0, max(self._frequency, self._run_frequency))
            step_height = min(self._step_height, self._run_step_limit)
        else:
            max_linear = self._max_linear
            frequency = self._frequency
            step_height = self._step_height

        cmd_vx = max(-max_linear, min(max_linear, cmd_vx))
        cmd_vy = max(-max_linear, min(max_linear, cmd_vy))
        cmd_wz = max(-self._max_yaw, min(self._max_yaw, cmd_wz))

        if dt > 0.0:
            alpha = dt / (self._command_filter + dt)
            self._vx += alpha * (cmd_vx - self._vx)
            self._vy += alpha * (cmd_vy - self._vy)
            self._wz += alpha * (cmd_wz - self._wz)
        else:
            self._vx, self._vy, self._wz = cmd_vx, cmd_vy, cmd_wz

        moving = (
            math.hypot(self._vx, self._vy) > self._deadband_linear
            or abs(self._wz) > self._deadband_yaw
        )
        target_activity = 1.0 if moving else 0.0
        if dt > 0.0:
            blend = min(1.0, dt / self._activity_ramp)
            self._activity += blend * (target_activity - self._activity)
        else:
            self._activity = target_activity

        if mode == MODE_DANCE:
            self._dance_weight = min(1.0, self._dance_weight + dt / self._dance_ramp)
            self._dance_time += dt
        elif self._dance_weight > 0.0:
            self._dance_weight = max(0.0, self._dance_weight - dt / self._dance_ramp)
            self._dance_time += dt

        if self._dance_weight > 0.0:
            dance_positions = dance_joint_positions(
                self._dance_time, self._body_height, self._step_height
            )
            alpha = _smoothstep(self._dance_weight)
            if mode == MODE_DANCE:
                start = self._dance_start_pose
                if start is None:
                    start = stand_joint_positions(self._body_height)
                positions = _blend(start, dance_positions, alpha)
            else:
                positions = _blend(
                    stand_joint_positions(self._body_height), dance_positions, alpha
                )
            self._publish_positions(positions)
            if self._dance_weight == 0.0:
                self._dance_time = 0.0
                self._dance_start_pose = None
            return

        period = 1.0 / frequency
        stance_time = self._duty * period
        self._phase = (self._phase + dt / period) % 1.0

        positions = []
        for leg in range(6):
            offset = 0.0 if leg in TRIPOD_A else 0.5
            phase = (self._phase + offset) % 1.0
            nominal = nominal_foot(leg, self._body_height)

            if phase < self._duty:
                progress = phase / self._duty
                tau = (progress - 0.5) * stance_time
                x, y = self._body_frame_foot(nominal, tau)
                z = -self._body_height
            else:
                progress = (phase - self._duty) / (1.0 - self._duty)
                end_x, end_y = self._body_frame_foot(nominal, 0.5 * stance_time)
                start_x, start_y = self._body_frame_foot(nominal, -0.5 * stance_time)
                blend = progress * progress * (3.0 - 2.0 * progress)
                x = end_x + blend * (start_x - end_x)
                y = end_y + blend * (start_y - end_y)
                lift = math.sin(math.pi * progress)
                z = -self._body_height + step_height * self._activity * lift

            q1, q2, q3 = joints_from_foot(leg, np.array([x, y, z]))
            positions.extend((q1, q2, q3))

        self._publish_positions(positions)

    def _publish_positions(self, positions):
        command = Float64MultiArray()
        command.data = [float(value) for value in positions]
        self._publisher.publish(command)
        self._last_command = command
        self._last_positions = [float(value) for value in positions]

    def _body_frame_foot(self, nominal, tau):
        dx = nominal[0] - self._vx * tau
        dy = nominal[1] - self._vy * tau
        angle = -self._wz * tau
        c = math.cos(angle)
        s = math.sin(angle)
        return c * dx - s * dy, s * dx + c * dy


def main(args=None):
    rclpy.init(args=args)
    node = HexapodGait()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
