#!/usr/bin/env python3
"""
motor_interface.py

Motor interface for maze_bot.

Important locked-PWM behavior:
- Subscribes to /motor_pwm_lock as Float32MultiArray:
      [left_pwm, right_pwm, duration_sec]
- When locked PWM is active:
      /cmd_vel is ignored completely.
      The locked left/right command is converted once into final motor outputs.
      The same final left/right outputs are applied repeatedly until duration ends.
- Locked PWM uses ratio-preserving scaling:
      motor scale factors are applied,
      then any min/max correction is applied to both sides together
      so the relative left/right steering ratio is not distorted mid-step.

This is intended to work with reactive_nav.py, where each autonomous step:
    calculate target once -> publish fixed PWM pair -> move fixed duration
"""

import math
import os
import time
from typing import Optional, Tuple

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from gpiozero import PWMOutputDevice, DigitalOutputDevice


class MotorInterface(Node):
    def __init__(self) -> None:
        super().__init__("motor_interface")

        # ---------------- Parameters ----------------
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("motor_pwm_lock_topic", "/motor_pwm_lock")

        self.declare_parameter("min_pwm", 0.50)
        self.declare_parameter("max_pwm", 1.0)

        self.declare_parameter("linear_scale", 1.0)
        self.declare_parameter("angular_scale", 1.0)

        # Forward/backward were correct, but left/right turns needed this sign.
        self.declare_parameter("angular_sign", -1.0)

        # Hardware calibration scale.
        # These are applied before final output.
        self.declare_parameter("left_motor_scale", .895)
        self.declare_parameter("right_motor_scale", 0.92)

        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("control_rate_hz", 50.0)

        # GPIO pins
        self.declare_parameter("stby_pin", 17)

        self.declare_parameter("pwma_pin", 13)
        self.declare_parameter("ain1_pin", 15)
        self.declare_parameter("ain2_pin", 14)

        self.declare_parameter("pwmb_pin", 12)
        self.declare_parameter("bin1_pin", 21)
        self.declare_parameter("bin2_pin", 19)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.motor_pwm_lock_topic = str(self.get_parameter("motor_pwm_lock_topic").value)

        self.min_pwm = float(self.get_parameter("min_pwm").value)
        self.max_pwm = float(self.get_parameter("max_pwm").value)

        self.linear_scale = float(self.get_parameter("linear_scale").value)
        self.angular_scale = float(self.get_parameter("angular_scale").value)
        self.angular_sign = float(self.get_parameter("angular_sign").value)

        self.left_motor_scale = float(self.get_parameter("left_motor_scale").value)
        self.right_motor_scale = float(self.get_parameter("right_motor_scale").value)

        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        stby_pin = int(self.get_parameter("stby_pin").value)

        pwma_pin = int(self.get_parameter("pwma_pin").value)
        ain1_pin = int(self.get_parameter("ain1_pin").value)
        ain2_pin = int(self.get_parameter("ain2_pin").value)

        pwmb_pin = int(self.get_parameter("pwmb_pin").value)
        bin1_pin = int(self.get_parameter("bin1_pin").value)
        bin2_pin = int(self.get_parameter("bin2_pin").value)

        # ---------------- GPIO setup ----------------
        self.stby = DigitalOutputDevice(stby_pin)

        self.left_pwm = PWMOutputDevice(pwma_pin, frequency=1000)
        self.left_in1 = DigitalOutputDevice(ain1_pin)
        self.left_in2 = DigitalOutputDevice(ain2_pin)

        self.right_pwm = PWMOutputDevice(pwmb_pin, frequency=1000)
        self.right_in1 = DigitalOutputDevice(bin1_pin)
        self.right_in2 = DigitalOutputDevice(bin2_pin)

        self.stby.on()

        # ---------------- ROS I/O ----------------
        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )

        self.lock_sub = self.create_subscription(
            Float32MultiArray,
            self.motor_pwm_lock_topic,
            self.locked_pwm_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(control_rate_hz, 1.0),
            self.control_loop,
        )

        # ---------------- State ----------------
        self.last_cmd_time = 0.0
        self.last_cmd: Optional[Twist] = None

        self.locked_pwm_active = False
        self.locked_pwm_end_time = 0.0

        # Raw command received from /motor_pwm_lock.
        self.locked_raw_left_pwm = 0.0
        self.locked_raw_right_pwm = 0.0

        # Final motor outputs calculated once at lock start.
        # These are what get repeatedly applied during the whole locked step.
        self.locked_output_left_pwm = 0.0
        self.locked_output_right_pwm = 0.0

        self.last_locked_log_time = 0.0

        self.get_logger().info("Motor interface started: ratio-preserving locked PWM version.")
        self.get_logger().info(f"Listening on {self.cmd_vel_topic} and {self.motor_pwm_lock_topic}")
        self.get_logger().info(
            f"Pins: STBY={stby_pin}, "
            f"L(PWM={pwma_pin}, IN1={ain1_pin}, IN2={ain2_pin}), "
            f"R(PWM={pwmb_pin}, IN1={bin1_pin}, IN2={bin2_pin})"
        )
        self.get_logger().info(
            f"min_pwm={self.min_pwm:.2f}, max_pwm={self.max_pwm:.2f}, "
            f"left_motor_scale={self.left_motor_scale:.3f}, "
            f"right_motor_scale={self.right_motor_scale:.3f}"
        )
        self.get_logger().info(
            "While locked_pwm_active=True, /cmd_vel is ignored and the same "
            "final left/right PWM outputs are held until duration expires."
        )

    # ---------------- Callbacks ----------------

    def cmd_vel_callback(self, msg: Twist) -> None:
        if self.locked_pwm_active:
            # Autonomous fixed movement step owns the motors.
            # Do not let /cmd_vel modify motor outputs mid-step.
            return

        self.last_cmd = msg
        self.last_cmd_time = time.monotonic()

    def locked_pwm_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 3:
            self.get_logger().warn(
                "Ignoring /motor_pwm_lock message with fewer than 3 values."
            )
            return

        raw_left = float(msg.data[0])
        raw_right = float(msg.data[1])
        duration = float(msg.data[2])

        if duration <= 0.0 or (abs(raw_left) < 1e-6 and abs(raw_right) < 1e-6):
            self.locked_pwm_active = False
            self.locked_raw_left_pwm = 0.0
            self.locked_raw_right_pwm = 0.0
            self.locked_output_left_pwm = 0.0
            self.locked_output_right_pwm = 0.0
            self.locked_pwm_end_time = 0.0
            self.stop_motors()
            self.get_logger().info("Locked PWM cancelled/stopped.")
            return

        # Clamp raw command to allowed range, but do not min-clamp here.
        # The locked final outputs are computed using ratio-preserving logic.
        raw_left = self.clamp_max_only(raw_left)
        raw_right = self.clamp_max_only(raw_right)

        final_left, final_right = self.prepare_locked_outputs_preserve_ratio(
            raw_left,
            raw_right,
        )

        self.locked_raw_left_pwm = raw_left
        self.locked_raw_right_pwm = raw_right

        self.locked_output_left_pwm = final_left
        self.locked_output_right_pwm = final_right

        self.locked_pwm_end_time = time.monotonic() + duration
        self.locked_pwm_active = True
        self.last_locked_log_time = 0.0

        self.get_logger().info(
            "Locked PWM active: "
            f"RAW L={raw_left:.3f}, R={raw_right:.3f}, "
            f"raw_ratio={self.safe_ratio(raw_left, raw_right):.3f}; "
            f"FINAL L={final_left:.3f}, R={final_right:.3f}, "
            f"final_ratio={self.safe_ratio(final_left, final_right):.3f}; "
            f"duration={duration:.2f}s"
        )

    # ---------------- Control ----------------

    def control_loop(self) -> None:
        now = time.monotonic()

        if self.locked_pwm_active:
            if now >= self.locked_pwm_end_time:
                self.locked_pwm_active = False

                finished_left = self.locked_output_left_pwm
                finished_right = self.locked_output_right_pwm

                self.locked_raw_left_pwm = 0.0
                self.locked_raw_right_pwm = 0.0
                self.locked_output_left_pwm = 0.0
                self.locked_output_right_pwm = 0.0
                self.locked_pwm_end_time = 0.0

                self.stop_motors()

                self.get_logger().info(
                    f"Locked PWM finished. Motors stopped. "
                    f"Last held outputs: L={finished_left:.3f}, R={finished_right:.3f}"
                )
            else:
                # This is the actual lock:
                # repeatedly apply the same final motor outputs.
                self.set_motor_outputs_direct(
                    self.locked_output_left_pwm,
                    self.locked_output_right_pwm,
                )

                if now - self.last_locked_log_time > 0.5:
                    remaining = max(0.0, self.locked_pwm_end_time - now)
                    self.get_logger().debug(
                        f"Holding locked PWM: "
                        f"L={self.locked_output_left_pwm:.3f}, "
                        f"R={self.locked_output_right_pwm:.3f}, "
                        f"remaining={remaining:.2f}s"
                    )
                    self.last_locked_log_time = now

            return

        # Manual /cmd_vel path.
        if self.last_cmd is None or (now - self.last_cmd_time) > self.cmd_timeout_sec:
            self.stop_motors()
            return

        left, right = self.twist_to_pwms(self.last_cmd)
        self.set_motor_outputs_direct(left, right)

    def twist_to_pwms(self, twist: Twist) -> Tuple[float, float]:
        linear = float(twist.linear.x) * self.linear_scale
        angular = float(twist.angular.z) * self.angular_scale * self.angular_sign

        left = linear - angular
        right = linear + angular

        # For manual driving, independent min clamp is okay.
        left = self.clamp_with_min(left)
        right = self.clamp_with_min(right)

        # Apply hardware motor scale for manual commands too.
        left *= self.left_motor_scale
        right *= self.right_motor_scale

        left = self.clamp_with_min(left)
        right = self.clamp_with_min(right)

        return left, right

    # ---------------- Locked PWM ratio-preserving helpers ----------------

    def prepare_locked_outputs_preserve_ratio(
        self,
        raw_left: float,
        raw_right: float,
    ) -> Tuple[float, float]:
        """
        Convert raw locked command into final motor outputs.

        This is intentionally different from manual cmd_vel handling.

        For locked PWM:
        - Apply left/right motor calibration scales.
        - Do NOT independently clamp each side to min_pwm.
        - If one side is under min_pwm, scale BOTH sides by the same factor.
        - If one side exceeds max_pwm, scale BOTH sides down by the same factor.
        - The final left/right values are stored and reused for the entire step.
        """

        # Apply motor calibration scales.
        left = raw_left * self.left_motor_scale
        right = raw_right * self.right_motor_scale

        # If both are basically zero, stop.
        if abs(left) < 1e-6 and abs(right) < 1e-6:
            return 0.0, 0.0

        # First clamp absurd raw values by common scaling if needed.
        max_abs = max(abs(left), abs(right), 1e-6)

        if max_abs > self.max_pwm:
            scale_down = self.max_pwm / max_abs
            left *= scale_down
            right *= scale_down

        # Enforce min PWM using common scaling, preserving ratio.
        nonzero_values = [abs(v) for v in (left, right) if abs(v) > 1e-6]

        if nonzero_values:
            min_abs_nonzero = min(nonzero_values)

            if min_abs_nonzero < self.min_pwm:
                scale_up = self.min_pwm / max(min_abs_nonzero, 1e-6)
                left *= scale_up
                right *= scale_up

        # If min scaling pushed one side too high, scale both down together.
        max_abs = max(abs(left), abs(right), 1e-6)

        if max_abs > self.max_pwm:
            scale_down = self.max_pwm / max_abs
            left *= scale_down
            right *= scale_down

        # Final numerical clamp only.
        left = self.clamp_max_only(left)
        right = self.clamp_max_only(right)

        return left, right

    def clamp_max_only(self, value: float) -> float:
        return max(-self.max_pwm, min(self.max_pwm, value))

    def clamp_with_min(self, value: float) -> float:
        if abs(value) < 1e-6:
            return 0.0

        value = self.clamp_max_only(value)

        if abs(value) < self.min_pwm:
            value = math.copysign(self.min_pwm, value)

        return value

    def safe_ratio(self, left: float, right: float) -> float:
        if abs(right) < 1e-6:
            return 999.0
        return left / right

    # ---------------- Motor GPIO helpers ----------------

    def set_motor_outputs_direct(self, left: float, right: float) -> None:
        """
        Apply final motor outputs directly.

        This function does NOT do ratio-changing min PWM logic.
        It assumes locked PWM commands have already been prepared.
        """
        left = self.clamp_max_only(left)
        right = self.clamp_max_only(right)

        self.set_one_motor(left, self.left_pwm, self.left_in1, self.left_in2)
        self.set_one_motor(right, self.right_pwm, self.right_in1, self.right_in2)

    def set_one_motor(
        self,
        pwm_value: float,
        pwm_dev: PWMOutputDevice,
        in1: DigitalOutputDevice,
        in2: DigitalOutputDevice,
    ) -> None:
        pwm_value = self.clamp_max_only(pwm_value)

        if abs(pwm_value) < 1e-6:
            in1.off()
            in2.off()
            pwm_dev.value = 0.0
            return

        if pwm_value > 0.0:
            in1.on()
            in2.off()
        else:
            in1.off()
            in2.on()

        pwm_dev.value = min(abs(pwm_value), 1.0)

    def stop_motors(self) -> None:
        self.left_pwm.value = 0.0
        self.right_pwm.value = 0.0

        self.left_in1.off()
        self.left_in2.off()

        self.right_in1.off()
        self.right_in2.off()

    def destroy_node(self) -> bool:
        self.stop_motors()
        self.stby.off()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotorInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()