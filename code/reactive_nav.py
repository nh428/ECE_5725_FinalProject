#!/usr/bin/env python3
"""
reactive_nav.py

Reactive LiDAR navigation for obstacle-course behavior.

Main behavior:
- Uses /scan from SLLIDAR.
- Continuously computes live RViz markers from the newest scan.
- start / step_run / run / auto = continuous stepping.
- step = one locked movement step.
- During each movement step:
    1. Calculate the best open ray / gap once.
    2. Convert that target angle into fixed left/right PWM.
    3. Publish /motor_pwm_lock:
           Float32MultiArray [left_pwm, right_pwm, duration_sec]
    4. Do NOT change motor PWM during the step.
    5. Recalculate only after the step finishes.
- If an obstacle is too close directly ahead:
    1. Compare long rays on left vs. right.
    2. Lock a turn-in-place toward the more open side.
    3. After the turn, recalculate using the newest scan.

Updated:
- Navigation view cone is now -60 deg to +60 deg, 120 deg total.
- Candidate rays require at least min_cluster_points nearby scan points.
- Default min_cluster_points is now 15.

Live RViz marker topics:
- /nav_filtered_profile
- /nav_target_cluster
- /nav_top_candidates
- /nav_target_marker
- /nav_locked_target_marker
- /nav_fov_bounds_marker
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker


RayPoint = Tuple[float, float, float, float, float]
# angle_rad_robot_frame, range_m, x_m, y_m, front_bumper_clearance_m


class ReactiveNav(Node):
    def __init__(self) -> None:
        super().__init__("reactive_nav")

        # ---------------- ROS parameters ----------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("nav_mode_topic", "/nav_mode")

        # LiDAR geometry/orientation
        self.declare_parameter("lidar_scan_offset_deg", 0.0)
        self.declare_parameter("lidar_to_front_offset_m", 0.203)

        # Direction signs
        self.declare_parameter("steering_sign", 1.0)
        self.declare_parameter("linear_sign", 1.0)

        # Navigation FOV: 120 degrees total
        # -60 degrees to +60 degrees.
        self.declare_parameter("fov_left_deg", 60.0)
        self.declare_parameter("fov_right_deg", -60.0)

        # Scan filtering
        self.declare_parameter("min_range_m", 0.12)
        self.declare_parameter("max_range_m", 5.0)

        # Candidate selectivity
        # A candidate ray must have at least this many nearby points in its
        # local angular window to be considered stable enough.
        self.declare_parameter("min_cluster_points", 15)

        # Step timing
        self.declare_parameter("drive_step_duration_sec", 2.75)
        self.declare_parameter("avoid_turn_duration_sec", 0.85)

        # PWM behavior
        self.declare_parameter("min_pwm", 0.50)
        self.declare_parameter("max_forward_pwm", 0.85)
        self.declare_parameter("calibration_pwm", 0.80)
        self.declare_parameter("turn_pwm_gain", 0.42)

        # Avoidance behavior
        self.declare_parameter("close_obstacle_clearance_m", 0.25)
        self.declare_parameter("front_clearance_window_deg", 18.0)
        self.declare_parameter("side_ray_min_deg", 25.0)
        self.declare_parameter("side_ray_max_deg", 60.0)
        self.declare_parameter("avoid_turn_pwm", 0.62)

        # Open-ray scoring
        self.declare_parameter("candidate_downsample", 2)
        self.declare_parameter("local_window_deg", 12.0)
        self.declare_parameter("long_ray_threshold_m", 1.00)
        self.declare_parameter("target_marker_range_m", 1.8)

        # Update loop
        self.declare_parameter("control_rate_hz", 10.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.nav_mode_topic = str(self.get_parameter("nav_mode_topic").value)

        self.lidar_scan_offset_rad = math.radians(
            float(self.get_parameter("lidar_scan_offset_deg").value)
        )
        self.lidar_to_front_offset_m = float(
            self.get_parameter("lidar_to_front_offset_m").value
        )

        self.steering_sign = float(self.get_parameter("steering_sign").value)
        self.linear_sign = float(self.get_parameter("linear_sign").value)

        self.fov_left_rad = math.radians(float(self.get_parameter("fov_left_deg").value))
        self.fov_right_rad = math.radians(float(self.get_parameter("fov_right_deg").value))
        self.fov_min_rad = min(self.fov_left_rad, self.fov_right_rad)
        self.fov_max_rad = max(self.fov_left_rad, self.fov_right_rad)

        self.min_range_m = float(self.get_parameter("min_range_m").value)
        self.max_range_m = float(self.get_parameter("max_range_m").value)

        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)

        self.drive_step_duration_sec = float(
            self.get_parameter("drive_step_duration_sec").value
        )
        self.avoid_turn_duration_sec = float(
            self.get_parameter("avoid_turn_duration_sec").value
        )

        self.min_pwm = float(self.get_parameter("min_pwm").value)
        self.max_forward_pwm = float(self.get_parameter("max_forward_pwm").value)
        self.calibration_pwm = float(self.get_parameter("calibration_pwm").value)
        self.turn_pwm_gain = float(self.get_parameter("turn_pwm_gain").value)

        self.close_obstacle_clearance_m = float(
            self.get_parameter("close_obstacle_clearance_m").value
        )
        self.front_clearance_window_rad = math.radians(
            float(self.get_parameter("front_clearance_window_deg").value)
        )
        self.side_ray_min_rad = math.radians(
            float(self.get_parameter("side_ray_min_deg").value)
        )
        self.side_ray_max_rad = math.radians(
            float(self.get_parameter("side_ray_max_deg").value)
        )
        self.avoid_turn_pwm = float(self.get_parameter("avoid_turn_pwm").value)

        self.candidate_downsample = max(
            1,
            int(self.get_parameter("candidate_downsample").value),
        )
        self.local_window_rad = math.radians(
            float(self.get_parameter("local_window_deg").value)
        )
        self.long_ray_threshold_m = float(
            self.get_parameter("long_ray_threshold_m").value
        )
        self.target_marker_range_m = float(
            self.get_parameter("target_marker_range_m").value
        )

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / max(control_rate_hz, 1.0)

        # Marker QoS
        self.marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ---------------- ROS I/O ----------------
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.nav_mode_sub = self.create_subscription(
            String,
            self.nav_mode_topic,
            self.nav_mode_callback,
            10,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.motor_pwm_lock_pub = self.create_publisher(
            Float32MultiArray,
            "/motor_pwm_lock",
            10,
        )

        self.filtered_profile_pub = self.create_publisher(
            Marker,
            "/nav_filtered_profile",
            self.marker_qos,
        )
        self.target_cluster_pub = self.create_publisher(
            Marker,
            "/nav_target_cluster",
            self.marker_qos,
        )
        self.top_candidates_pub = self.create_publisher(
            Marker,
            "/nav_top_candidates",
            self.marker_qos,
        )
        self.target_marker_pub = self.create_publisher(
            Marker,
            "/nav_target_marker",
            self.marker_qos,
        )
        self.locked_target_marker_pub = self.create_publisher(
            Marker,
            "/nav_locked_target_marker",
            self.marker_qos,
        )
        self.fov_bounds_pub = self.create_publisher(
            Marker,
            "/nav_fov_bounds_marker",
            self.marker_qos,
        )

        self.timer = self.create_timer(self.control_period_sec, self.control_loop)

        # ---------------- State ----------------
        self.latest_scan: Optional[LaserScan] = None

        self.state = "STOPPED"
        self.continuous_step_run = False

        self.move_end_time = 0.0

        self.locked_target_xy: Optional[Tuple[float, float]] = None
        self.locked_left_pwm = 0.0
        self.locked_right_pwm = 0.0
        self.locked_action = "NONE"

        self.last_no_scan_log_time = 0.0
        self.last_hold_log_time = 0.0

        self.get_logger().info("Reactive nav started: selective open-ray obstacle-course version.")
        self.get_logger().info(f"Subscribing to scan topic: {self.scan_topic}")
        self.get_logger().info(
            "FOV: "
            f"{math.degrees(self.fov_min_rad):.1f} deg to "
            f"{math.degrees(self.fov_max_rad):.1f} deg "
            f"({math.degrees(self.fov_max_rad - self.fov_min_rad):.1f} deg total)"
        )
        self.get_logger().info(
            f"Minimum nearby points for candidate ray: {self.min_cluster_points}"
        )
        self.get_logger().info(
            f"Close obstacle turn threshold: {self.close_obstacle_clearance_m:.2f} m"
        )
        self.get_logger().info(
            "Behavior: close obstacle -> locked turn toward longer side ray; "
            "otherwise locked curved step toward best open ray."
        )

    # ---------------- Callbacks ----------------

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def nav_mode_callback(self, msg: String) -> None:
        cmd = msg.data.strip().lower()

        if cmd in ("start", "step_run", "run", "auto"):
            self.continuous_step_run = True

            if self.state != "MOVING":
                self.state = "CALCULATE"

            self.get_logger().info("Navigation command: START continuous stepping")

        elif cmd == "step":
            self.continuous_step_run = False

            if self.state != "MOVING":
                self.state = "CALCULATE"

            self.get_logger().info("Navigation command: STEP once")

        elif cmd in ("stop", "halt", "off"):
            self.continuous_step_run = False
            self.state = "STOPPED"

            self.locked_target_xy = None
            self.locked_left_pwm = 0.0
            self.locked_right_pwm = 0.0
            self.locked_action = "NONE"

            self.publish_stop()
            self.publish_zero_pwm_lock()
            self.delete_locked_target_marker()

            self.get_logger().info("Navigation command: STOP")

        else:
            self.get_logger().warn(f"Unknown /nav_mode command: {cmd}")

    # ---------------- Main loop ----------------

    def control_loop(self) -> None:
        now = time.monotonic()

        if self.latest_scan is None:
            self.publish_fov_bounds()

            if now - self.last_no_scan_log_time > 2.0:
                self.get_logger().warn("No /scan received yet.")
                self.last_no_scan_log_time = now

            if self.state != "STOPPED":
                self.publish_stop()

            return

        # Always recompute marker/candidate data from the newest scan.
        # This gives real-time RViz updates even while moving.
        processed = self.process_scan(self.latest_scan)
        self.publish_live_markers(processed)

        if self.state == "STOPPED":
            self.publish_stop()
            return

        if self.state == "MOVING":
            # Critical: do not recompute or republish motor PWM while moving.
            # The motor interface holds the locked PWM until duration ends.
            if now >= self.move_end_time:
                self.publish_stop()

                finished_action = self.locked_action
                finished_left = self.locked_left_pwm
                finished_right = self.locked_right_pwm

                self.locked_target_xy = None
                self.locked_left_pwm = 0.0
                self.locked_right_pwm = 0.0
                self.locked_action = "NONE"
                self.delete_locked_target_marker()

                if self.continuous_step_run:
                    self.state = "CALCULATE"
                    self.get_logger().info(
                        f"Continuous mode: {finished_action} step finished "
                        f"(L={finished_left:.2f}, R={finished_right:.2f}); "
                        "recalculating."
                    )
                else:
                    self.state = "STOPPED"
                    self.get_logger().info(
                        f"Single {finished_action} step finished "
                        f"(L={finished_left:.2f}, R={finished_right:.2f}); "
                        "returning to STOPPED."
                    )

            return

        if self.state == "CALCULATE":
            self.calculate_and_start_locked_step(processed)
            return

        self.get_logger().warn(f"Unknown state {self.state}; stopping.")
        self.state = "STOPPED"
        self.continuous_step_run = False
        self.publish_stop()
        self.publish_zero_pwm_lock()

    # ---------------- Scan processing ----------------

    def process_scan(self, scan: LaserScan) -> Dict[str, Any]:
        rays = self.extract_open_rays(scan)
        front_clearance = self.compute_front_clearance(rays)
        candidates = self.score_open_rays(rays)
        side_info = self.compute_side_opening(rays)

        return {
            "rays": rays,
            "front_clearance": front_clearance,
            "candidates": candidates,
            "side_info": side_info,
        }

    def extract_open_rays(self, scan: LaserScan) -> List[RayPoint]:
        rays: List[RayPoint] = []

        angle = scan.angle_min

        for raw_r in scan.ranges:
            if math.isinf(raw_r):
                r = self.max_range_m
            elif math.isfinite(raw_r):
                r = float(raw_r)
            else:
                angle += scan.angle_increment
                continue

            if r < self.min_range_m:
                angle += scan.angle_increment
                continue

            r = min(r, self.max_range_m)

            a = angle + self.lidar_scan_offset_rad
            a = math.atan2(math.sin(a), math.cos(a))

            if self.fov_min_rad <= a <= self.fov_max_rad:
                x = r * math.cos(a)
                y = r * math.sin(a)

                if x > 0.0:
                    front_bumper_clearance = max(0.0, x - self.lidar_to_front_offset_m)
                    rays.append((a, r, x, y, front_bumper_clearance))

            angle += scan.angle_increment

        rays.sort(key=lambda p: p[0])
        return rays

    def compute_front_clearance(self, rays: List[RayPoint]) -> float:
        front_rays = [p for p in rays if abs(p[0]) <= self.front_clearance_window_rad]

        if not front_rays:
            return self.max_range_m

        return min(p[4] for p in front_rays)

    def compute_side_opening(self, rays: List[RayPoint]) -> Dict[str, float]:
        left_rays = [
            p for p in rays
            if self.side_ray_min_rad <= p[0] <= self.side_ray_max_rad
        ]
        right_rays = [
            p for p in rays
            if -self.side_ray_max_rad <= p[0] <= -self.side_ray_min_rad
        ]

        left_score = self.side_open_score(left_rays)
        right_score = self.side_open_score(right_rays)

        return {
            "left_score": left_score,
            "right_score": right_score,
        }

    def side_open_score(self, rays: List[RayPoint]) -> float:
        if not rays:
            return 0.0

        ranges = [p[1] for p in rays]
        long_ranges = [r for r in ranges if r >= self.long_ray_threshold_m]

        max_r = max(ranges)

        if long_ranges:
            avg_long = sum(long_ranges) / len(long_ranges)
        else:
            avg_long = sum(ranges) / len(ranges)

        return 0.65 * max_r + 0.35 * avg_long

    def score_open_rays(self, rays: List[RayPoint]) -> List[dict]:
        candidates = []

        if not rays:
            return candidates

        for i, ray in enumerate(rays):
            if i % self.candidate_downsample != 0:
                continue

            angle, r, x, y, front_clearance = ray

            local_rays = [
                p for p in rays
                if abs(p[0] - angle) <= self.local_window_rad
            ]

            # This is the new selectivity requirement.
            # Do not consider isolated/noisy rays as candidates.
            if len(local_rays) < self.min_cluster_points:
                continue

            local_ranges = [p[1] for p in local_rays]
            local_mean_range = sum(local_ranges) / len(local_ranges)
            long_count = sum(1 for p in local_rays if p[1] >= self.long_ray_threshold_m)
            local_long_fraction = long_count / max(1, len(local_rays))

            range_score = min(r / self.max_range_m, 1.0)
            clearance_score = min(front_clearance / self.max_range_m, 1.0)

            # Prefer forward, but still allow side rays when they are much more open.
            center_score = 1.0 - min(abs(angle) / max(abs(self.fov_max_rad), 1e-6), 1.0)

            local_range_score = min(local_mean_range / self.max_range_m, 1.0)

            # Main goal: find a long, stable ray/gap.
            score = (
                3.0 * range_score
                + 2.0 * clearance_score
                + 1.2 * local_range_score
                + 1.0 * local_long_fraction
                + 0.8 * center_score
            )

            marker_dist = min(r, self.target_marker_range_m)
            target_x = marker_dist * math.cos(angle)
            target_y = marker_dist * math.sin(angle)

            candidates.append(
                {
                    "score": score,
                    "angle": angle,
                    "range": r,
                    "target_xy": (target_x, target_y),
                    "raw_xy": (x, y),
                    "front_bumper_clearance": front_clearance,
                    "local_mean_range": local_mean_range,
                    "local_long_fraction": local_long_fraction,
                    "local_point_count": len(local_rays),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    # ---------------- Step action selection ----------------

    def calculate_and_start_locked_step(self, processed: Dict[str, Any]) -> None:
        now = time.monotonic()

        rays: List[RayPoint] = processed["rays"]
        front_clearance: float = processed["front_clearance"]
        candidates: List[dict] = processed["candidates"]
        side_info: Dict[str, float] = processed["side_info"]

        if not rays:
            self.publish_stop()

            if now - self.last_hold_log_time > 1.0:
                self.get_logger().warn("No valid open rays. Holding position.")
                self.last_hold_log_time = now

            self.state = "CALCULATE" if self.continuous_step_run else "STOPPED"
            return

        if front_clearance <= self.close_obstacle_clearance_m:
            action, left_pwm, right_pwm, duration, locked_xy = self.make_avoid_turn_command(
                side_info
            )
            self.start_locked_action(
                action=action,
                left_pwm=left_pwm,
                right_pwm=right_pwm,
                duration=duration,
                target_xy=locked_xy,
                extra_log=(
                    f"front_clearance={front_clearance:.2f}, "
                    f"left_open={side_info['left_score']:.2f}, "
                    f"right_open={side_info['right_score']:.2f}"
                ),
            )
            return

        if not candidates:
            self.publish_stop()

            if now - self.last_hold_log_time > 1.0:
                self.get_logger().warn(
                    f"No open-ray candidates with at least {self.min_cluster_points} local points. "
                    "Holding position."
                )
                self.last_hold_log_time = now

            self.state = "CALCULATE" if self.continuous_step_run else "STOPPED"
            return

        best = candidates[0]
        target_angle = float(best["angle"])
        target_x, target_y = best["target_xy"]

        left_pwm, right_pwm = self.target_angle_to_locked_pwm(target_angle)

        self.start_locked_action(
            action="TRACK_RAY",
            left_pwm=left_pwm,
            right_pwm=right_pwm,
            duration=self.drive_step_duration_sec,
            target_xy=(target_x, target_y),
            extra_log=(
                f"angle={math.degrees(target_angle):.1f} deg, "
                f"ray_range={best['range']:.2f}, "
                f"front_clearance={front_clearance:.2f}, "
                f"local_points={best['local_point_count']}, "
                f"score={best['score']:.2f}"
            ),
        )

    def start_locked_action(
        self,
        action: str,
        left_pwm: float,
        right_pwm: float,
        duration: float,
        target_xy: Tuple[float, float],
        extra_log: str,
    ) -> None:
        self.locked_action = action
        self.locked_left_pwm = left_pwm
        self.locked_right_pwm = right_pwm
        self.locked_target_xy = target_xy

        self.publish_locked_target_marker(target_xy[0], target_xy[1])

        msg = Float32MultiArray()
        msg.data = [float(left_pwm), float(right_pwm), float(duration)]
        self.motor_pwm_lock_pub.publish(msg)

        self.move_end_time = time.monotonic() + duration
        self.state = "MOVING"

        mode_text = "continuous" if self.continuous_step_run else "single-step"

        self.get_logger().info(
            f"Locked {action} step started [{mode_text}]: "
            f"L={left_pwm:.2f}, R={right_pwm:.2f}, dur={duration:.2f}s, "
            f"{extra_log}"
        )

    def make_avoid_turn_command(
        self,
        side_info: Dict[str, float],
    ) -> Tuple[str, float, float, float, Tuple[float, float]]:
        left_score = side_info["left_score"]
        right_score = side_info["right_score"]

        turn_left = left_score >= right_score

        p = max(self.min_pwm, min(self.max_forward_pwm, self.avoid_turn_pwm))

        if self.steering_sign < 0.0:
            turn_left = not turn_left

        if turn_left:
            left_pwm = -p
            right_pwm = p
            action = "AVOID_TURN_LEFT"
            target_angle = math.radians(60.0)
        else:
            left_pwm = p
            right_pwm = -p
            action = "AVOID_TURN_RIGHT"
            target_angle = math.radians(-60.0)

        marker_x = self.target_marker_range_m * math.cos(target_angle)
        marker_y = self.target_marker_range_m * math.sin(target_angle)

        return action, left_pwm, right_pwm, self.avoid_turn_duration_sec, (marker_x, marker_y)

    def target_angle_to_locked_pwm(self, target_angle: float) -> Tuple[float, float]:
        """
        Convert target angle to a fixed curved PWM pair.

        This is intentionally calculated only at the beginning of the step.
        During MOVING, this function is not called again.
        """
        base_pwm = max(self.min_pwm, min(self.max_forward_pwm, self.calibration_pwm))

        angle_norm = target_angle / math.radians(60.0)
        angle_norm = max(-1.0, min(1.0, angle_norm))

        turn = self.steering_sign * self.turn_pwm_gain * angle_norm
        turn = max(-0.45, min(0.45, turn))

        raw_left = 1.0 - turn
        raw_right = 1.0 + turn

        max_raw = max(abs(raw_left), abs(raw_right), 1e-6)

        ratio_left = raw_left / max_raw
        ratio_right = raw_right / max_raw

        left = base_pwm * ratio_left * self.linear_sign
        right = base_pwm * ratio_right * self.linear_sign

        left, right = self.enforce_min_max_preserve_ratio(left, right)

        return left, right

    def enforce_min_max_preserve_ratio(self, left: float, right: float) -> Tuple[float, float]:
        nonzero_values = [abs(v) for v in (left, right) if abs(v) > 1e-6]

        if nonzero_values:
            min_abs_nonzero = min(nonzero_values)

            if min_abs_nonzero < self.min_pwm:
                scale_up = self.min_pwm / max(min_abs_nonzero, 1e-6)
                left *= scale_up
                right *= scale_up

        max_abs = max(abs(left), abs(right), 1e-6)

        if max_abs > self.max_forward_pwm:
            scale_down = self.max_forward_pwm / max_abs
            left *= scale_down
            right *= scale_down

        left = max(-self.max_forward_pwm, min(self.max_forward_pwm, left))
        right = max(-self.max_forward_pwm, min(self.max_forward_pwm, right))

        return left, right

    # ---------------- Command publishers ----------------

    def publish_stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def publish_zero_pwm_lock(self) -> None:
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0]
        self.motor_pwm_lock_pub.publish(msg)

    # ---------------- Marker publishing ----------------

    def publish_live_markers(self, processed: Dict[str, Any]) -> None:
        rays: List[RayPoint] = processed["rays"]
        candidates: List[dict] = processed["candidates"]

        self.publish_filtered_profile(rays)
        self.publish_target_clusters(rays)
        self.publish_top_candidates(candidates[:8])
        self.publish_fov_bounds()

        if candidates:
            best = candidates[0]
            x, y = best["target_xy"]
            self.publish_target_marker(x, y)
        else:
            self.delete_target_marker()

        if self.locked_target_xy is not None:
            self.publish_locked_target_marker(
                self.locked_target_xy[0],
                self.locked_target_xy[1],
            )
        else:
            self.delete_locked_target_marker()

    def make_marker(self, ns: str, marker_id: int, marker_type: int) -> Marker:
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 0
        return m

    def publish_filtered_profile(self, rays: List[RayPoint]) -> None:
        m = self.make_marker("nav_filtered_profile", 0, Marker.POINTS)
        m.scale.x = 0.035
        m.scale.y = 0.035
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0

        for _, _, x, y, _ in rays:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.02
            m.points.append(p)

        self.filtered_profile_pub.publish(m)

    def publish_target_clusters(self, rays: List[RayPoint]) -> None:
        """
        Blue points represent open-ray endpoints used for navigation.
        """
        m = self.make_marker("nav_target_cluster", 0, Marker.POINTS)
        m.scale.x = 0.055
        m.scale.y = 0.055
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.2
        m.color.b = 1.0

        for _, _, x, y, _ in rays[:: self.candidate_downsample]:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            m.points.append(p)

        self.target_cluster_pub.publish(m)

    def publish_top_candidates(self, candidates: List[dict]) -> None:
        m = self.make_marker("nav_top_candidates", 0, Marker.SPHERE_LIST)
        m.scale.x = 0.12
        m.scale.y = 0.12
        m.scale.z = 0.12
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.55
        m.color.b = 0.0

        for item in candidates:
            x, y = item["target_xy"]
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.10
            m.points.append(p)

        self.top_candidates_pub.publish(m)

    def publish_target_marker(self, x: float, y: float) -> None:
        m = self.make_marker("nav_target_marker", 0, Marker.ARROW)

        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.12

        end = Point()
        end.x = float(x)
        end.y = float(y)
        end.z = 0.12

        m.points.append(start)
        m.points.append(end)

        m.scale.x = 0.045
        m.scale.y = 0.14
        m.scale.z = 0.22

        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 1.0

        self.target_marker_pub.publish(m)

    def delete_target_marker(self) -> None:
        m = self.make_marker("nav_target_marker", 0, Marker.ARROW)
        m.action = Marker.DELETE
        self.target_marker_pub.publish(m)

    def publish_locked_target_marker(self, x: float, y: float) -> None:
        m = self.make_marker("nav_locked_target_marker", 0, Marker.ARROW)

        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.20

        end = Point()
        end.x = float(x)
        end.y = float(y)
        end.z = 0.20

        m.points.append(start)
        m.points.append(end)

        m.scale.x = 0.060
        m.scale.y = 0.18
        m.scale.z = 0.28

        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0

        self.locked_target_marker_pub.publish(m)

    def delete_locked_target_marker(self) -> None:
        m = self.make_marker("nav_locked_target_marker", 0, Marker.ARROW)
        m.action = Marker.DELETE
        self.locked_target_marker_pub.publish(m)

    def publish_fov_bounds(self) -> None:
        fov_min = math.radians(-60.0)
        fov_max = math.radians(60.0)

        length = 2.0
        z = 0.08

        rays = self.make_marker("nav_fov_bounds_marker", 0, Marker.LINE_LIST)
        rays.scale.x = 0.055
        rays.color.a = 1.0
        rays.color.r = 0.0
        rays.color.g = 1.0
        rays.color.b = 1.0

        for a in [fov_min, 0.0, fov_max]:
            p0 = Point()
            p0.x = 0.0
            p0.y = 0.0
            p0.z = z

            p1 = Point()
            p1.x = float(length * math.cos(a))
            p1.y = float(length * math.sin(a))
            p1.z = z

            rays.points.append(p0)
            rays.points.append(p1)

        self.fov_bounds_pub.publish(rays)

        arc = self.make_marker("nav_fov_bounds_marker", 1, Marker.LINE_STRIP)
        arc.scale.x = 0.050
        arc.color.a = 1.0
        arc.color.r = 0.0
        arc.color.g = 1.0
        arc.color.b = 1.0

        num_arc_points = 41

        for i in range(num_arc_points):
            t = i / float(num_arc_points - 1)
            a = fov_min + t * (fov_max - fov_min)

            p = Point()
            p.x = float(length * math.cos(a))
            p.y = float(length * math.sin(a))
            p.z = z

            arc.points.append(p)

        self.fov_bounds_pub.publish(arc)

        fan = self.make_marker("nav_fov_bounds_marker", 2, Marker.TRIANGLE_LIST)
        fan.scale.x = 1.0
        fan.scale.y = 1.0
        fan.scale.z = 1.0
        fan.color.a = 0.18
        fan.color.r = 0.0
        fan.color.g = 1.0
        fan.color.b = 1.0

        num_triangles = 40

        for i in range(num_triangles):
            a0 = fov_min + (i / float(num_triangles)) * (fov_max - fov_min)
            a1 = fov_min + ((i + 1) / float(num_triangles)) * (fov_max - fov_min)

            center = Point()
            center.x = 0.0
            center.y = 0.0
            center.z = z - 0.01

            p0 = Point()
            p0.x = float(length * math.cos(a0))
            p0.y = float(length * math.sin(a0))
            p0.z = z - 0.01

            p1 = Point()
            p1.x = float(length * math.cos(a1))
            p1.y = float(length * math.sin(a1))
            p1.z = z - 0.01

            fan.points.append(center)
            fan.points.append(p0)
            fan.points.append(p1)

        self.fov_bounds_pub.publish(fan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReactiveNav()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.publish_zero_pwm_lock()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
