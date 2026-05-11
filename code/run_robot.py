#!/usr/bin/env bash

set -e

# ============================================================
# run_robot
# Clean launch script for robot mapping/navigation
#
# Fix included:
#   - Suppresses noisy sllidar / tf / slam logs
#   - Creates a corrected /scan topic by rotating LiDAR scan data
#     180 degrees in the LaserScan message itself
#   - Uses yaw = 0 for the LiDAR transform
#   - Starts robot nodes in a controlled order
# ============================================================

# -----------------------------
# User settings
# -----------------------------

WORKSPACE="$HOME/ros2_ws"

# Update these if your package/executable names are different
REACTIVE_PKG="reactive_nav"
REACTIVE_EXE="reactive_navigator"

MOTOR_PKG="motor_interface"
MOTOR_EXE="motor_interface"

# LiDAR settings
LIDAR_PORT="/dev/ttyUSB0"
LIDAR_BAUD="115200"

# Raw scan from sllidar
RAW_SCAN_TOPIC="/scan_raw"

# Corrected scan used by the rest of the robot
CORRECTED_SCAN_TOPIC="/scan"

# Frames
BASE_FRAME="base_link"
LIDAR_FRAME="laser"

# LiDAR pose relative to base_link
# Keep yaw = 0 because we are correcting the scan data directly.
LIDAR_X="0.0"
LIDAR_Y="0.0"
LIDAR_Z="0.12"
LIDAR_ROLL="0.0"
LIDAR_PITCH="0.0"
LIDAR_YAW="0.0"

# SLAM config
SLAM_PARAMS="$WORKSPACE/src/slam_toolbox/config/mapper_params_online_async.yaml"

# Log directory
LOG_DIR="$HOME/robot_logs"
mkdir -p "$LOG_DIR"

# -----------------------------
# Helper functions
# -----------------------------

cleanup() {
    echo ""
    echo "[run_robot] Shutting down robot stack..."

    pkill -f "sllidar_node" 2>/dev/null || true
    pkill -f "static_transform_publisher" 2>/dev/null || true
    pkill -f "async_slam_toolbox_node" 2>/dev/null || true
    pkill -f "scan_180_corrector.py" 2>/dev/null || true
    pkill -f "$REACTIVE_EXE" 2>/dev/null || true
    pkill -f "$MOTOR_EXE" 2>/dev/null || true

    echo "[run_robot] Done."
}

trap cleanup EXIT

wait_for_topic() {
    local topic_name="$1"
    local timeout_sec="$2"
    local elapsed=0

    echo "[run_robot] Waiting for topic $topic_name..."

    while [ "$elapsed" -lt "$timeout_sec" ]; do
        if ros2 topic list 2>/dev/null | grep -qx "$topic_name"; then
            echo "[run_robot] Found $topic_name."
            return 0
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "[run_robot] ERROR: Timed out waiting for $topic_name."
    return 1
}

print_build_time() {
    local pkg="$1"
    local exe="$2"

    local path
    path="$(ros2 pkg prefix "$pkg" 2>/dev/null || true)"

    if [ -z "$path" ]; then
        echo "[run_robot] $pkg not found."
        return
    fi

    local full_exe="$path/lib/$pkg/$exe"

    if [ -f "$full_exe" ]; then
        echo "[run_robot] $pkg/$exe built: $(stat -c '%y' "$full_exe")"
    else
        echo "[run_robot] Could not find executable: $full_exe"
    fi
}

# -----------------------------
# Start clean
# -----------------------------

echo "[run_robot] Cleaning up old robot processes..."

pkill -f "sllidar_node" 2>/dev/null || true
pkill -f "static_transform_publisher" 2>/dev/null || true
pkill -f "async_slam_toolbox_node" 2>/dev/null || true
pkill -f "scan_180_corrector.py" 2>/dev/null || true
pkill -f "$REACTIVE_EXE" 2>/dev/null || true
pkill -f "$MOTOR_EXE" 2>/dev/null || true

sleep 2

# -----------------------------
# Source ROS workspace
# -----------------------------

if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "[run_robot] ERROR: Could not find /opt/ros/humble/setup.bash"
    exit 1
fi

if [ -f "$WORKSPACE/install/setup.bash" ]; then
    source "$WORKSPACE/install/setup.bash"
else
    echo "[run_robot] ERROR: Could not find $WORKSPACE/install/setup.bash"
    exit 1
fi

echo "[run_robot] ROS environment loaded."

print_build_time "$REACTIVE_PKG" "$REACTIVE_EXE"
print_build_time "$MOTOR_PKG" "$MOTOR_EXE"

# -----------------------------
# Check LiDAR port
# -----------------------------

if [ ! -e "$LIDAR_PORT" ]; then
    echo "[run_robot] ERROR: LiDAR port $LIDAR_PORT does not exist."
    echo "[run_robot] Available serial devices:"
    ls -l /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null || true
    exit 1
fi

echo "[run_robot] Using LiDAR port: $LIDAR_PORT"

# -----------------------------
# Create temporary scan corrector node
# -----------------------------

SCAN_CORRECTOR="/tmp/scan_180_corrector.py"

cat > "$SCAN_CORRECTOR" << 'EOF'
#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class Scan180Corrector(Node):
    def __init__(self):
        super().__init__('scan_180_corrector')

        self.declare_parameter('input_topic', '/scan_raw')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('frame_id', 'laser')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.sub = self.create_subscription(
            LaserScan,
            input_topic,
            self.scan_callback,
            10
        )

        self.pub = self.create_publisher(
            LaserScan,
            output_topic,
            10
        )

        self.get_logger().info(
            f'Correcting LaserScan by 180 degrees: {input_topic} -> {output_topic}'
        )

    def scan_callback(self, msg):
        out = LaserScan()

        out.header = msg.header
        out.header.frame_id = self.frame_id

        out.angle_min = self.normalize_angle(msg.angle_min + math.pi)
        out.angle_max = self.normalize_angle(msg.angle_max + math.pi)

        # If adding pi makes angle_min greater than angle_max, keep the scan
        # continuous by shifting the interval back into a valid increasing range.
        if out.angle_min > out.angle_max:
            out.angle_min -= 2.0 * math.pi

        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max

        # Important:
        # Do NOT reverse the ranges here.
        # We are rotating the scan angles by 180 degrees while preserving
        # each range's angular index relationship.
        out.ranges = list(msg.ranges)
        out.intensities = list(msg.intensities)

        self.pub.publish(out)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = Scan180Corrector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
EOF

chmod +x "$SCAN_CORRECTOR"

# -----------------------------
# Start LiDAR driver quietly
# -----------------------------

echo "[run_robot] Starting SLLIDAR quietly on $RAW_SCAN_TOPIC..."

ros2 run sllidar_ros2 sllidar_node \
    --ros-args \
    -r __node:=sllidar_node \
    -r scan:="$RAW_SCAN_TOPIC" \
    -p serial_port:="$LIDAR_PORT" \
    -p serial_baudrate:="$LIDAR_BAUD" \
    --log-level ERROR \
    > "$LOG_DIR/sllidar.log" 2>&1 &

sleep 3

wait_for_topic "$RAW_SCAN_TOPIC" 15

# -----------------------------
# Start corrected scan publisher
# -----------------------------

echo "[run_robot] Starting 180-degree LaserScan corrector..."

python3 "$SCAN_CORRECTOR" \
    --ros-args \
    -p input_topic:="$RAW_SCAN_TOPIC" \
    -p output_topic:="$CORRECTED_SCAN_TOPIC" \
    -p frame_id:="$LIDAR_FRAME" \
    --log-level WARN \
    > "$LOG_DIR/scan_corrector.log" 2>&1 &

sleep 2

wait_for_topic "$CORRECTED_SCAN_TOPIC" 15

# -----------------------------
# Static transform
# -----------------------------

echo "[run_robot] Starting static transform: $BASE_FRAME -> $LIDAR_FRAME"

ros2 run tf2_ros static_transform_publisher \
    "$LIDAR_X" "$LIDAR_Y" "$LIDAR_Z" \
    "$LIDAR_ROLL" "$LIDAR_PITCH" "$LIDAR_YAW" \
    "$BASE_FRAME" "$LIDAR_FRAME" \
    --ros-args --log-level ERROR \
    > "$LOG_DIR/static_tf.log" 2>&1 &

sleep 1

# -----------------------------
# Start SLAM
# -----------------------------

echo "[run_robot] Starting SLAM toolbox..."

if [ -f "$SLAM_PARAMS" ]; then
    ros2 launch slam_toolbox online_async_launch.py \
        slam_params_file:="$SLAM_PARAMS" \
        use_sim_time:=false \
        > "$LOG_DIR/slam_toolbox.log" 2>&1 &
else
    echo "[run_robot] WARNING: SLAM params file not found:"
    echo "[run_robot] $SLAM_PARAMS"
    echo "[run_robot] Starting SLAM toolbox with default params."

    ros2 launch slam_toolbox online_async_launch.py \
        use_sim_time:=false \
        > "$LOG_DIR/slam_toolbox.log" 2>&1 &
fi

sleep 4

# -----------------------------
# Start reactive navigator
# -----------------------------

echo "[run_robot] Starting reactive navigator..."

ros2 run "$REACTIVE_PKG" "$REACTIVE_EXE" \
    --ros-args \
    -p scan_topic:="$CORRECTED_SCAN_TOPIC" \
    -p lidar_scan_offset_deg:=0.0 \
    -p steering_sign:=1.0 \
    -p linear_sign:=1.0 \
    --log-level INFO \
    > "$LOG_DIR/reactive_navigator.log" 2>&1 &

sleep 2

# -----------------------------
# Start motor interface
# -----------------------------

echo "[run_robot] Starting motor interface..."

ros2 run "$MOTOR_PKG" "$MOTOR_EXE" \
    --ros-args --log-level INFO \
    > "$LOG_DIR/motor_interface.log" 2>&1 &

sleep 2

# -----------------------------
# Summary
# -----------------------------

echo ""
echo "[run_robot] Robot stack started."
echo "[run_robot] Raw LiDAR topic:       $RAW_SCAN_TOPIC"
echo "[run_robot] Corrected scan topic:  $CORRECTED_SCAN_TOPIC"
echo "[run_robot] LiDAR TF yaw:          $LIDAR_YAW rad"
echo "[run_robot] LiDAR scan offset:     applied in LaserScan data, not TF"
echo "[run_robot] Logs saved in:         $LOG_DIR"
echo ""
echo "[run_robot] To check scans:"
echo "  ros2 topic echo $CORRECTED_SCAN_TOPIC"
echo ""
echo "[run_robot] To stop, press Ctrl+C."
echo ""

while true; do
    sleep 1
done