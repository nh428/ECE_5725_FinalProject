#!/usr/bin/env python3
"""
robot_debug.py  â€”  Run this in a second terminal while run_robot.py is active.

Usage:
    source /opt/ros/jazzy/setup.bash
    source /home/pi/ros2_ws/install/setup.bash
    python3 robot_debug.py

Shows a live dashboard of:
  - /scan  â†’ what the lidar sees (center, left, right distances + gap direction)
  - /cmd_vel â†’ what reactive_nav is commanding (speed + turn)
  - /cmd_vel â†’ what the motors are actually receiving (derived PWM values)
  - Node heartbeat: warns if topics go silent
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import math
import time
import os

# â”€â”€ tunables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
WHEEL_BASE       = 0.101   # metres  â€” keep in sync with motor_interface.py
MAX_WHEEL_SPEED  = 0.15    # m/s     â€” keep in sync with motor_interface.py
MIN_PWM          = 0.45    # stiction threshold â€” keep in sync with motor_interface.py
SILENT_TIMEOUT   = 2.0     # seconds before "NO DATA" warning
FOV_DEGREES      = 140     # keep in sync with reactive_nav.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def clear():
    os.system('clear')

def bar(value, max_val, width=20, fill='â–ˆ', empty='â–‘'):
    """Simple ASCII bar, value can be negative (shows direction)."""
    proportion = abs(value) / max_val if max_val != 0 else 0
    filled = int(proportion * width)
    b = fill * filled + empty * (width - filled)
    if value < 0:
        return f'â—€ {b}'
    elif value > 0:
        return f'{b} â–¶'
    else:
        return f'{empty * width}'

def arrow(angle_deg):
    """Return a directional arrow based on angle."""
    if abs(angle_deg) < 10:
        return 'â†‘  STRAIGHT'
    elif angle_deg > 45:
        return 'â†—  TURN RIGHT (sharp)'
    elif angle_deg > 10:
        return 'â†—  TURN RIGHT'
    elif angle_deg < -45:
        return 'â†™  TURN LEFT (sharp)'
    else:
        return 'â†–  TURN LEFT'

def pwm_display(pwm):
    """Colour-code PWM: stopped / low / normal / high."""
    if pwm == 0:
        return f'  STOPPED  '
    sign = '+' if pwm > 0 else '-'
    pct = int(abs(pwm) * 100)
    if abs(pwm) <= MIN_PWM + 0.01:
        tag = '(at min)'
    elif abs(pwm) > 0.85:
        tag = '(HIGH)'
    else:
        tag = ''
    return f'{sign}{pct:3d}% {tag}'


class DebugMonitor(Node):
    def __init__(self):
        super().__init__('robot_debug_monitor')

        # State populated by callbacks
        self.scan_time    = None
        self.cmd_time     = None

        self.center_dist  = 0.0
        self.left_dist    = 0.0
        self.right_dist   = 0.0
        self.max_gap      = 0.0
        self.gap_angle    = 0.0   # degrees

        self.linear_x     = 0.0
        self.angular_z    = 0.0
        self.pwm_left     = 0.0
        self.pwm_right    = 0.0

        self.scan_count   = 0
        self.cmd_count    = 0

        # Subscriptions
        self.create_subscription(LaserScan, '/scan',    self.scan_cb,    qos_profile_sensor_data)
        self.create_subscription(Twist,     '/cmd_vel', self.cmd_vel_cb, 10)

        # Refresh display at 10 Hz regardless of message rate
        self.create_timer(0.1, self.display)

        self.start_time = time.time()
        self.get_logger().info('Debug monitor started. Waiting for topics...')

    # â”€â”€ callbacks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def scan_cb(self, msg):
        self.scan_time = time.time()
        self.scan_count += 1

        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges), 5.0, ranges)
        ranges = np.clip(ranges, 0, 5.0)

        if math.isnan(msg.angle_increment) or msg.angle_increment == 0:
            return

        fov_rad = math.radians(FOV_DEGREES)
        angle_increment = msg.angle_increment
        num_indices_fov = int((fov_rad / 2.0) / angle_increment)

        if msg.angle_min >= 0:
            right_arc = ranges[:num_indices_fov]
            left_arc  = ranges[-num_indices_fov:]
            front     = np.concatenate((left_arc, right_arc))
        else:
            center_index = len(ranges) // 2
            front = ranges[center_index - num_indices_fov : center_index + num_indices_fov]

        angles = np.linspace(-fov_rad / 2, fov_rad / 2, len(front))
        window = 15
        smoothed = np.convolve(front, np.ones(window) / window, mode='same')

        n = len(smoothed)
        self.center_dist = float(np.mean(smoothed[n//2 - 5 : n//2 + 5]))
        self.left_dist   = float(np.mean(smoothed[:n // 3]))
        self.right_dist  = float(np.mean(smoothed[2 * n // 3:]))
        self.max_gap     = float(np.max(smoothed))
        self.gap_angle   = math.degrees(float(angles[np.argmax(smoothed)]))

    def cmd_vel_cb(self, msg):
        self.cmd_time  = time.time()
        self.cmd_count += 1
        self.linear_x  = msg.linear.x
        self.angular_z = msg.angular.z

        # Derive PWM exactly as motor_interface.py does
        v     = msg.linear.x
        omega = msg.angular.z
        vl = v - (omega * WHEEL_BASE / 2.0)
        vr = v + (omega * WHEEL_BASE / 2.0)

        def to_pwm(wheel_v):
            p = wheel_v / MAX_WHEEL_SPEED
            p = max(min(p, 1.0), -1.0)
            if p > 0:   return max(p, MIN_PWM)
            elif p < 0: return min(p, -MIN_PWM)
            return 0.0

        self.pwm_left  = to_pwm(vl)
        self.pwm_right = to_pwm(vr)

    # â”€â”€ display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def display(self):
        now      = time.time()
        uptime   = int(now - self.start_time)
        scan_age = (now - self.scan_time) if self.scan_time else None
        cmd_age  = (now - self.cmd_time)  if self.cmd_time  else None

        scan_ok = scan_age is not None and scan_age < SILENT_TIMEOUT
        cmd_ok  = cmd_age  is not None and cmd_age  < SILENT_TIMEOUT

        clear()
        print("â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
        print("â•‘           MAZE BOT  â€”  LIVE DEBUG MONITOR           â•‘")
        print(f"â•‘  uptime: {uptime:>5}s    scans: {self.scan_count:<6}  cmds: {self.cmd_count:<6}  â•‘")
        print("â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£")

        # â”€â”€ LIDAR section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("â•‘  LIDAR  /scan                                        â•‘")
        if not scan_ok:
            status = "WAITING FOR DATA..." if self.scan_time is None else f"NO DATA  ({scan_age:.1f}s ago)"
            print(f"â•‘    âš   {status:<45} â•‘")
        else:
            # Distance bars (max display range = 4m)
            def dist_bar(d):
                filled = int(min(d, 4.0) / 4.0 * 16)
                return 'â–ˆ' * filled + 'â–‘' * (16 - filled)

            print(f"â•‘    LEFT   {self.left_dist:4.2f}m  [{dist_bar(self.left_dist)}]          â•‘")
            print(f"â•‘    CENTER {self.center_dist:4.2f}m  [{dist_bar(self.center_dist)}]          â•‘")
            print(f"â•‘    RIGHT  {self.right_dist:4.2f}m  [{dist_bar(self.right_dist)}]          â•‘")
            print(f"â•‘    MAX GAP {self.max_gap:4.2f}m                                   â•‘")
            danger = self.center_dist < 0.35
            center_label = '  âš  OBSTACLE!' if danger else ''
            print(f"â•‘    Gap direction: {arrow(self.gap_angle):<20} ({self.gap_angle:+.1f}Â°){center_label:<12}â•‘")

        print("â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£")

        # â”€â”€ CMD_VEL section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("â•‘  REACTIVE NAV  /cmd_vel                              â•‘")
        if not cmd_ok:
            status = "WAITING FOR DATA..." if self.cmd_time is None else f"NO DATA  ({cmd_age:.1f}s ago)"
            print(f"â•‘    âš   {status:<45} â•‘")
        else:
            spd_bar = bar(self.linear_x, MAX_WHEEL_SPEED)
            trn_bar = bar(self.angular_z, 1.0)
            print(f"â•‘    Linear   {self.linear_x:+.3f} m/s   {spd_bar:<26}â•‘")
            print(f"â•‘    Angular  {self.angular_z:+.3f} r/s   {trn_bar:<26}â•‘")

            if self.linear_x == 0 and self.angular_z != 0:
                mode = 'âŸ³  SPINNING IN PLACE'
            elif abs(self.angular_z) < 0.05:
                mode = 'â†‘  DRIVING STRAIGHT'
            elif self.linear_x > 0 and self.angular_z > 0.3:
                mode = 'â†—  CURVING RIGHT'
            elif self.linear_x > 0 and self.angular_z < -0.3:
                mode = 'â†–  CURVING LEFT'
            elif self.linear_x == 0 and self.angular_z == 0:
                mode = 'â–   STOPPED'
            else:
                mode = '~  ADJUSTING'
            print(f"â•‘    Mode:    {mode:<41}â•‘")

        print("â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£")

        # â”€â”€ MOTOR section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("â•‘  MOTORS  (derived from /cmd_vel)                     â•‘")
        if not cmd_ok:
            print(f"â•‘    âš   No /cmd_vel data â€” motor state unknown          â•‘")
        else:
            lbar = bar(self.pwm_left,  1.0, width=16)
            rbar = bar(self.pwm_right, 1.0, width=16)
            print(f"â•‘    LEFT  PWM: {pwm_display(self.pwm_left):<14}  {lbar:<20}â•‘")
            print(f"â•‘    RIGHT PWM: {pwm_display(self.pwm_right):<14}  {rbar:<20}â•‘")

            # Diagnose common problems
            both_min = (abs(self.pwm_left) == MIN_PWM and abs(self.pwm_right) == MIN_PWM
                        and self.linear_x != 0)
            same_dir = (self.pwm_left > 0 and self.pwm_right > 0) or \
                       (self.pwm_left < 0 and self.pwm_right < 0)

            if both_min and same_dir and abs(self.angular_z) > 0.3:
                print("â•‘    âš   DIAG: Both motors at min PWM despite large turn  â•‘")
                print("â•‘       â†’ wheel_base or max_wheel_speed may be wrong     â•‘")
            elif both_min and self.linear_x > 0:
                print("â•‘    âš   DIAG: Both motors clamped to min PWM             â•‘")
                print("â•‘       â†’ try lowering max_wheel_speed in motor_interface â•‘")
            elif not same_dir and abs(self.angular_z) > 0.5:
                print("â•‘    âœ“  Differential drive active (motors counter-rotate) â•‘")
            else:
                print("â•‘                                                      â•‘")

        print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
        print("  Ctrl+C to quit")


def main():
    rclpy.init()
    node = DebugMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\nDebug monitor stopped.")


if __name__ == '__main__':
    main()