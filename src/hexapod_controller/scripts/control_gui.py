#!/usr/bin/env python3

import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

CMD_TOPIC = '/hexapod/cmd_vel'
PARAMS_TOPIC = '/hexapod/gait_params'
ESTOP_TOPIC = '/hexapod/estop'

MAX_LINEAR_SPEED = 0.08
MAX_YAW_RATE = 0.6
DEFAULT_BODY_HEIGHT_MM = 68
DEFAULT_STEP_HEIGHT_MM = 30
DEFAULT_FREQUENCY_X10 = 10


class Joystick(QWidget):
    valueChanged = pyqtSignal(float, float)

    def __init__(self, horizontal_only=False, parent=None):
        super().__init__(parent)
        self._horizontal_only = horizontal_only
        self._value_x = 0.0
        self._value_y = 0.0
        self._dragging = False
        self.setMinimumSize(170, 170)
        self.setMouseTracking(True)

    def value(self):
        return self._value_x, self._value_y

    def reset(self):
        self._value_x = 0.0
        self._value_y = 0.0
        self._dragging = False
        self.update()
        self.valueChanged.emit(0.0, 0.0)

    def _set_from_position(self, position):
        radius = min(self.width(), self.height()) / 2.0 - 18.0
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        dx = position.x() - center.x()
        dy = position.y() - center.y()
        if self._horizontal_only:
            dy = 0.0
        length = math.hypot(dx, dy)
        if length > radius:
            scale = radius / length
            dx *= scale
            dy *= scale
            length = radius
        self._value_x = dx / radius
        self._value_y = -dy / radius
        self.update()
        self.valueChanged.emit(self._value_x, self._value_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._set_from_position(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._set_from_position(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._value_x = 0.0
            self._value_y = 0.0
            self.update()
            self.valueChanged.emit(0.0, 0.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        radius = min(self.width(), self.height()) / 2.0 - 18.0
        center = QPointF(self.width() / 2.0, self.height() / 2.0)

        painter.setPen(QPen(QColor('#444b55'), 2))
        painter.setBrush(QColor('#20252c'))
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QPen(QColor('#333a44'), 1))
        painter.drawLine(QPointF(center.x() - radius, center.y()),
                         QPointF(center.x() + radius, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - radius),
                         QPointF(center.x(), center.y() + radius))

        handle_radius = 20.0
        hx = center.x() + self._value_x * radius
        hy = center.y() - self._value_y * radius
        if self._horizontal_only:
            hy = center.y()
        painter.setPen(QPen(QColor('#5db9ff'), 2))
        painter.setBrush(QColor('#2f8fd8'))
        painter.drawEllipse(QPointF(hx, hy), handle_radius, handle_radius)


class HexapodControlWindow(QWidget):
    def __init__(self, node):
        super().__init__()
        self._node = node
        self._run_x = 0.0
        self._run_y = 0.0
        self._turn_x = 0.0
        self._frozen = False

        self._cmd_pub = node.create_publisher(Twist, CMD_TOPIC, 10)
        self._params_pub = node.create_publisher(Float64MultiArray, PARAMS_TOPIC, 10)
        self._estop_pub = node.create_publisher(Bool, ESTOP_TOPIC, 10)

        self.setWindowTitle('Hexapod Controller')
        self.setMinimumWidth(620)
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(50)

        self._params_timer = QTimer(self)
        self._params_timer.timeout.connect(self._publish_params)
        self._params_timer.start(1000)
        self._publish_params()

    def _build_ui(self):
        title = QLabel('HEXAPOD')
        title.setObjectName('title')
        subtitle = QLabel('Gait control / desktop interface')
        subtitle.setObjectName('subtitle')

        self._run_joystick = Joystick()
        self._run_joystick.valueChanged.connect(self._on_run)
        self._turn_joystick = Joystick(horizontal_only=True)
        self._turn_joystick.valueChanged.connect(self._on_turn)

        run_card = self._make_card('RUN', self._run_joystick)
        turn_card = self._make_card('TURN', self._turn_joystick)

        self._height_slider = self._make_slider(40, 90, DEFAULT_BODY_HEIGHT_MM)
        self._step_slider = self._make_slider(10, 50, DEFAULT_STEP_HEIGHT_MM)
        self._frequency_slider = self._make_slider(5, 20, DEFAULT_FREQUENCY_X10)

        self._height_label = QLabel()
        self._step_label = QLabel()
        self._frequency_label = QLabel()
        self._height_slider.valueChanged.connect(self._update_labels)
        self._step_slider.valueChanged.connect(self._update_labels)
        self._frequency_slider.valueChanged.connect(self._update_labels)
        self._update_labels()

        sliders = QGridLayout()
        sliders.addWidget(QLabel('Body height'), 0, 0)
        sliders.addWidget(self._height_slider, 0, 1)
        sliders.addWidget(self._height_label, 0, 2)
        sliders.addWidget(QLabel('Step height'), 1, 0)
        sliders.addWidget(self._step_slider, 1, 1)
        sliders.addWidget(self._step_label, 1, 2)
        sliders.addWidget(QLabel('Gait frequency'), 2, 0)
        sliders.addWidget(self._frequency_slider, 2, 1)
        sliders.addWidget(self._frequency_label, 2, 2)

        stand_button = QPushButton('STAND')
        stand_button.setObjectName('stand')
        stand_button.clicked.connect(self._stand)
        stop_button = QPushButton('FREEZE')
        stop_button.setObjectName('stop')
        stop_button.clicked.connect(self._freeze)
        reset_button = QPushButton('RESET')
        reset_button.clicked.connect(self._reset)

        buttons = QHBoxLayout()
        buttons.addWidget(stand_button)
        buttons.addWidget(reset_button)
        buttons.addWidget(stop_button)

        self._status = QLabel('Publishing on /hexapod/cmd_vel')
        self._status.setObjectName('status')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        sticks = QHBoxLayout()
        sticks.addWidget(run_card)
        sticks.addWidget(turn_card)
        layout.addLayout(sticks)
        layout.addLayout(sliders)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

        self.setStyleSheet(
            '''
            QWidget { background-color: #15191e; color: #e8edf3; font-size: 14px; }
            QLabel#title { font-size: 28px; font-weight: 700; letter-spacing: 4px; color: #6ec1ff; }
            QLabel#subtitle { color: #9aa6b2; margin-bottom: 6px; }
            QLabel#status { color: #7dcf9b; }
            QPushButton { background-color: #2b333d; border: 1px solid #46505c; border-radius: 8px; padding: 10px 18px; font-weight: 600; }
            QPushButton:hover { background-color: #38424e; }
            QPushButton#stand { background-color: #1f6f45; border-color: #2e9a61; }
            QPushButton#stop { background-color: #8a2d2d; border-color: #c14a4a; }
            QSlider::groove:horizontal { height: 6px; background: #2b333d; border-radius: 3px; }
            QSlider::handle:horizontal { width: 18px; margin: -7px 0; border-radius: 9px; background: #5db9ff; }
            '''
        )

    def _make_card(self, label, widget):
        card = QWidget()
        card.setStyleSheet('background-color: #1b2027; border: 1px solid #2c343d; border-radius: 12px;')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        caption = QLabel(label)
        caption.setStyleSheet('color: #9aa6b2; font-weight: 700; letter-spacing: 2px; border: none;')
        caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(caption)
        layout.addWidget(widget)
        return card

    def _make_slider(self, minimum, maximum, value):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        return slider

    def _update_labels(self):
        self._height_label.setText(f'{self._height_slider.value()} mm')
        self._step_label.setText(f'{self._step_slider.value()} mm')
        self._frequency_label.setText(f'{self._frequency_slider.value() / 10.0:.1f} Hz')
        self._publish_params()

    def _on_run(self, x, y):
        self._run_x = x
        self._run_y = y

    def _on_turn(self, x, y):
        self._turn_x = x

    def _on_timer(self):
        if not rclpy.ok():
            return
        deadband = 0.08
        vx = self._run_y * MAX_LINEAR_SPEED if abs(self._run_y) > deadband else 0.0
        vy = -self._run_x * MAX_LINEAR_SPEED if abs(self._run_x) > deadband else 0.0
        wz = -self._turn_x * MAX_YAW_RATE if abs(self._turn_x) > deadband else 0.0
        if self._frozen:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._cmd_pub.publish(msg)
        self._status.setText(f'cmd  vx={vx:+.3f} m/s  vy={vy:+.3f} m/s  wz={wz:+.2f} rad/s')

    def _publish_params(self):
        if not rclpy.ok():
            return
        msg = Float64MultiArray()
        msg.data = [
            self._height_slider.value() / 1000.0,
            self._step_slider.value() / 1000.0,
            self._frequency_slider.value() / 10.0,
        ]
        self._params_pub.publish(msg)

    def _send_stop(self):
        self._run_joystick.reset()
        self._turn_joystick.reset()
        self._cmd_pub.publish(Twist())

    def _freeze(self):
        self._frozen = True
        self._send_stop()
        self._estop_pub.publish(Bool(data=True))
        self._status.setText('FREEZE active')

    def _stand(self):
        self._frozen = False
        self._send_stop()
        self._estop_pub.publish(Bool(data=False))
        self._status.setText('Standing')

    def _reset(self):
        self._height_slider.setValue(DEFAULT_BODY_HEIGHT_MM)
        self._step_slider.setValue(DEFAULT_STEP_HEIGHT_MM)
        self._frequency_slider.setValue(DEFAULT_FREQUENCY_X10)
        self._stand()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._freeze()
        elif event.key() == Qt.Key_R:
            self._stand()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stand()
        self._timer.stop()
        self._params_timer.stop()
        event.accept()


def main(args=None):
    rclpy.init(args=args)
    node = Node('hexapod_control_gui')
    app = QApplication([sys.argv[0]])
    window = HexapodControlWindow(node)
    window.show()

    try:
        app.exec_()
    finally:
        window._timer.stop()
        window._params_timer.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
