
#!/usr/bin/env python3
"""
BNO055 PDR prototype + Wi-Fi live position server.

Features:
- acceleration-magnitude step signal
- rolling average smoothing
- slow baseline removal
- event-based peak capture
- cooldown between steps
- minimum absolute peak requirement
- press Enter to start
- press Enter again to stop early
- starting facing direction becomes "north"
- heading debug plot
- 90-degree discrete heading bins: N, E, S, W
- direction hysteresis / direction locking
- vertical IMU mount support using a body-forward sensor axis
- fixed 0.61 m step length, about 2 feet
- live Wi-Fi server so Pi 4 can receive x/y/person tracking data

Pi 4 should poll:
    http://PI_ZERO_IP:8000/position

Useful endpoints:
    /position
    /status
    /reset

Outputs:
1. XY trajectory plot
2. ax, ay, az vs time plot
3. step_signal vs time with detected steps marked
4. heading debug plot
5. trajectory CSV with heading data
"""

import csv
import json
import math
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import board
import adafruit_bno055
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# WIFI SERVER SETTINGS
# --------------------------------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000


# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
SAMPLE_HZ = 50.0
DT_TARGET = 1.0 / SAMPLE_HZ
RUN_TIME_SEC = 1000.0


# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------
# Change this to compare modes:
# "IMUPLUS" = gyro + accel fusion, no magnetometer.
# "NDOF"    = gyro + accel + magnetometer fusion.
BNO_MODE_NAME = "IMUPLUS"

if BNO_MODE_NAME == "IMUPLUS":
    BNO_MODE = adafruit_bno055.IMUPLUS_MODE
elif BNO_MODE_NAME == "NDOF":
    BNO_MODE = adafruit_bno055.NDOF_MODE
else:
    raise ValueError("BNO_MODE_NAME must be either 'IMUPLUS' or 'NDOF'")


# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------
OUTPUT_DIR = "/home/kaelembent/Final_Project/images"
OUTPUT_TAG = f"vertical_{BNO_MODE_NAME.lower()}_cardinal_locked_maze"

OUTPUT_PNG_XY = f"{OUTPUT_DIR}/pdr_track_xy_{OUTPUT_TAG}.png"
OUTPUT_PNG_ACC = f"{OUTPUT_DIR}/pdr_acc_xyz_{OUTPUT_TAG}.png"
OUTPUT_PNG_SIGNAL = f"{OUTPUT_DIR}/pdr_step_signal_{OUTPUT_TAG}.png"
OUTPUT_PNG_HEADING = f"{OUTPUT_DIR}/pdr_heading_debug_{OUTPUT_TAG}.png"
OUTPUT_CSV = f"{OUTPUT_DIR}/pdr_track_{OUTPUT_TAG}.csv"


# --------------------------------------------------
# IMU MOUNTING SETTING
# --------------------------------------------------
# Since the BNO055 is mounted vertically on the body, heading should be computed
# from the sensor axis that points in the person's walking direction.
#
# Try "+Z" first if the board face points forward away from your body.
# Try "-Z" if the board face points into your body.
#
# Other options:
#   "+X", "-X", "+Y", "-Y", "+Z", "-Z"
BODY_FORWARD_AXIS = "+Z"


# --------------------------------------------------
# PDR SETTINGS
# --------------------------------------------------

# heading smoothing
HEADING_LPF_ALPHA = 0.25

# Optional heading correction.
# Use this if the whole path is rotated by a fixed amount.
# Try 0, 90, -90, or 180 if the axes look wrong.
HEADING_OFFSET_DEG = 0.0
HEADING_OFFSET_RAD = math.radians(HEADING_OFFSET_DEG)

# If right turns become left turns, flip this.
USE_FLIPPED_HEADING_SIGN = True

# If your x-direction is still mirrored after trying USE_FLIPPED_HEADING_SIGN,
# set this to True.
FLIP_X_UPDATE = False

# 90-degree discrete heading mode
USE_CARDINAL_HEADING = True

# Direction locking / hysteresis
# This prevents N/E/N/E flickering when heading is near a boundary.
DIRECTION_SWITCH_TO_NEW_DEG = 30.0
DIRECTION_LEAVE_CURRENT_DEG = 55.0
DIRECTION_CONFIRM_STEPS = 2

# step-signal construction
ROLLING_WINDOW = 5
BASELINE_WINDOW = 60

# event-based detection thresholds
STEP_EVENT_START = 2.2
STEP_EVENT_END = 0.5
MIN_STEP_PROMINENCE = 2.0
MIN_STEP_INTERVAL = 0.45

# absolute peak threshold
MIN_STEP_PEAK = 5.0

# fixed step length
FIXED_STEP_LEN_M = 0.61  # about 2 feet
MIN_STEP_LEN = 0.0
MAX_STEP_LEN = 1.10

# plot options
LABEL_STEP_TIMES = True

# moving status timeout
MOVING_TIMEOUT_SEC = 1.5


# --------------------------------------------------
# Stop control
# --------------------------------------------------
stop_event = threading.Event()


def wait_for_stop_input():
    input("\nPress Enter again at any time to stop recording early...\n")
    stop_event.set()


# --------------------------------------------------
# Shared live state for Wi-Fi server
# --------------------------------------------------
state_lock = threading.Lock()

live_state = {
    "ok": True,
    "running": False,
    "x": 0.0,
    "y": 0.0,
    "steps_total": 0,
    "steps": 0,
    "direction": "N",
    "moving": False,
    "t": 0.0,
    "last_step_time": None,
    "heading_deg": 0.0,
    "heading_used_deg": 0.0,
    "instant_direction": "N",
    "instant_heading_deg": 0.0,
    "axis_heading_deg": 0.0,
    "original_yaw_deg": 0.0,
    "cal_sys": 0,
    "cal_gyro": 0,
    "cal_acc": 0,
    "cal_mag": 0,
    "mode": BNO_MODE_NAME,
    "body_forward_axis": BODY_FORWARD_AXIS,
}


def reset_live_state():
    with state_lock:
        live_state["x"] = 0.0
        live_state["y"] = 0.0
        live_state["steps_total"] = 0
        live_state["steps"] = 0
        live_state["direction"] = "N"
        live_state["moving"] = False
        live_state["t"] = 0.0
        live_state["last_step_time"] = None
        live_state["heading_deg"] = 0.0
        live_state["heading_used_deg"] = 0.0
        live_state["instant_direction"] = "N"
        live_state["instant_heading_deg"] = 0.0
        live_state["axis_heading_deg"] = 0.0
        live_state["original_yaw_deg"] = 0.0


class PositionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/position" or path == "/status":
            with state_lock:
                data = dict(live_state)
            self.send_json(data)
            return

        if path == "/reset":
            reset_live_state()
            self.send_json({"ok": True, "message": "Pi Zero tracking state reset"})
            return

        self.send_json({
            "ok": False,
            "error": "Valid endpoints are /position, /status, /reset"
        })

    def send_json(self, data):
        payload = json.dumps(data).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def start_wifi_server():
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), PositionHandler)
    print(f"Pi Zero Wi-Fi server running on http://0.0.0.0:{SERVER_PORT}")
    print("Pi 4 should poll:")
    print(f"  http://<PI_ZERO_IP>:{SERVER_PORT}/position")
    print("Available endpoints:")
    print("  /position")
    print("  /status")
    print("  /reset")
    print()
    server.serve_forever()


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_axis_vector(axis_name):
    """
    Return a unit vector for the selected sensor/body axis.
    """
    axes = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }

    if axis_name not in axes:
        raise ValueError("BODY_FORWARD_AXIS must be one of +X, -X, +Y, -Y, +Z, -Z")

    return axes[axis_name]


def quat_to_rotation_matrix(q):
    """
    Convert BNO055 quaternion (w, x, y, z) to a rotation matrix.
    """
    w, x, y, z = q

    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])


def quat_to_heading_from_forward_axis(q, forward_axis_name):
    """
    Compute walking heading from the sensor axis that points forward on the body.

    This is better than plain yaw when the IMU is mounted vertically.
    It rotates the chosen sensor-forward axis into the world frame, projects that
    vector onto the horizontal XY plane, and calculates the angle of that vector.
    """
    q = np.array(q, dtype=float)

    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return None

    q = q / norm

    forward_sensor = get_axis_vector(forward_axis_name)
    R = quat_to_rotation_matrix(q)

    forward_world = R @ forward_sensor

    fx = forward_world[0]
    fy = forward_world[1]

    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        return None

    return math.atan2(fy, fx)


def quat_to_yaw(q):
    """
    Original yaw calculation, kept for optional debugging/reference.
    The main code now uses quat_to_heading_from_forward_axis().
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def wrap_deg(deg):
    """Wrap degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def angle_lerp(prev, new, alpha):
    """Low-pass filter angle while handling wraparound."""
    diff = wrap_angle(new - prev)
    return wrap_angle(prev + alpha * diff)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_current_heading(sensor):
    """
    Read current BNO055 quaternion and compute heading using the selected
    body-forward sensor axis.
    """
    q = sensor.quaternion
    if q is None or None in q:
        return None

    return quat_to_heading_from_forward_axis(q, BODY_FORWARD_AXIS)


CARDINAL_DEGS = {
    "N": 0.0,
    "E": 90.0,
    "S": 180.0,
    "W": -90.0,
}


def angle_error_deg(a, b):
    """
    Smallest signed difference from b to a in degrees.
    Returns value in [-180, 180).
    """
    return (a - b + 180.0) % 360.0 - 180.0


def angle_error_abs_deg(a, b):
    return abs(angle_error_deg(a, b))


def nearest_cardinal_dir(heading_deg):
    heading_deg = wrap_deg(heading_deg)
    return min(
        CARDINAL_DEGS.keys(),
        key=lambda d: angle_error_abs_deg(heading_deg, CARDINAL_DEGS[d])
    )


def cardinal_dir_to_rad(direction):
    return math.radians(CARDINAL_DEGS[direction])


def quantize_heading_cardinal(heading_rad):
    """
    Instant quantization to one of four cardinal directions.
    This is still logged for debugging, but the actual PDR update uses
    the direction-locked accepted direction.
    """
    deg = math.degrees(heading_rad)
    deg = wrap_deg(deg)
    direction = nearest_cardinal_dir(deg)
    return cardinal_dir_to_rad(direction), CARDINAL_DEGS[direction], direction


# --------------------------------------------------
# Start Wi-Fi server before sensor setup
# --------------------------------------------------
server_thread = threading.Thread(target=start_wifi_server, daemon=True)
server_thread.start()


# --------------------------------------------------
# Sensor setup
# --------------------------------------------------
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

time.sleep(1.0)
sensor.mode = BNO_MODE
time.sleep(0.1)

print("BNO055 started")
print("Selected mode:", BNO_MODE_NAME)
print("Sensor mode value:", sensor.mode)
print("Body forward axis:", BODY_FORWARD_AXIS)
print("Calibration status is (sys, gyro, accel, mag)")
print("Current calibration:", sensor.calibration_status)
print()


# --------------------------------------------------
# Wait for user to start
# --------------------------------------------------
print("Mount the IMU vertically in the same orientation you will use while walking.")
print("Face the direction you want to define as NORTH.")
print("Hold the IMU still for a moment.")
input("Press Enter to start recording PDR data and serving live position...\n")

# Capture the starting heading and define it as north.
time.sleep(0.2)

north_heading = None
for _ in range(30):
    north_heading = get_current_heading(sensor)
    if north_heading is not None:
        break
    time.sleep(0.05)

if north_heading is None:
    raise RuntimeError("Could not read initial heading for north reference.")

print(f"Reference north heading captured: {north_heading:.3f} rad")
print("This starting facing direction is now treated as north.")
print(f"Heading offset: {HEADING_OFFSET_DEG:.1f} deg")
print(f"Flipped heading sign: {USE_FLIPPED_HEADING_SIGN}")
print(f"Cardinal heading mode: {USE_CARDINAL_HEADING}")
print(f"Flip X update: {FLIP_X_UPDATE}")
print(f"Direction switch to new threshold: {DIRECTION_SWITCH_TO_NEW_DEG:.1f} deg")
print(f"Direction leave current threshold: {DIRECTION_LEAVE_CURRENT_DEG:.1f} deg")
print(f"Direction confirm steps: {DIRECTION_CONFIRM_STEPS}")
print(f"Fixed step length: {FIXED_STEP_LEN_M:.2f} m")
print()

with state_lock:
    live_state["running"] = True
    live_state["mode"] = BNO_MODE_NAME
    live_state["body_forward_axis"] = BODY_FORWARD_AXIS

# Start background listener for stop.
stop_thread = threading.Thread(target=wait_for_stop_input, daemon=True)
stop_thread.start()


# --------------------------------------------------
# State
# --------------------------------------------------
x = 0.0
y = 0.0
steps_total = 0

times = []
xs = []
ys = []

step_events = []

# raw accel logs
acc_t = []
acc_x = []
acc_y = []
acc_z = []

# step-signal logs
sig_t = []
sig_raw_log = []
sig_filt_log = []
baseline_log = []
step_signal_log = []

# heading logs
heading_t = []
raw_heading_deg_log = []
rel_heading_deg_log = []
smooth_heading_deg_log = []

# optional original yaw logs
original_yaw_deg_log = []

# instant quantized heading logs
quant_heading_deg_log = []
quant_heading_dir_log = []

# accepted/locked heading logs
accepted_heading_deg_log = []
accepted_heading_dir_log = []

# calibration logs
cal_sys_log = []
cal_gyro_log = []
cal_acc_log = []
cal_mag_log = []

# buffers
sig_window = deque(maxlen=ROLLING_WINDOW)
baseline_window = deque(maxlen=BASELINE_WINDOW)

last_step_time = -999.0
prev_heading = None

# Direction locking state
accepted_heading_dir = "N"
pending_heading_dir = None
pending_heading_count = 0

# event detector state
in_step_event = False
event_peak_val = -1e9
event_peak_time = None
event_start_valley = 0.0

start_time = time.perf_counter()
prev_time = start_time


# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    now = time.perf_counter()
    elapsed = now - start_time

    if elapsed >= RUN_TIME_SEC:
        print("\nReached maximum run time.")
        break

    if stop_event.is_set():
        print("\nStopped early by user.")
        break

    dt = now - prev_time
    prev_time = now

    lin = sensor.linear_acceleration
    quat = sensor.quaternion
    gyro = sensor.gyro
    cal = sensor.calibration_status

    if (
        not lin or not quat or not gyro
        or None in lin or None in quat or None in gyro
    ):
        time.sleep(0.005)
        continue

    q = np.array(quat, dtype=float)
    if np.linalg.norm(q) < 1e-6:
        time.sleep(0.005)
        continue

    a = np.array(lin, dtype=float)

    # log raw acceleration
    acc_t.append(elapsed)
    acc_x.append(a[0])
    acc_y.append(a[1])
    acc_z.append(a[2])

    # --------------------------------------------------
    # Heading relative to starting north direction
    # --------------------------------------------------
    raw_heading = quat_to_heading_from_forward_axis(q, BODY_FORWARD_AXIS)

    if raw_heading is None:
        time.sleep(0.005)
        continue

    # Optional original yaw for comparison/debugging.
    original_yaw = quat_to_yaw(q)

    if USE_FLIPPED_HEADING_SIGN:
        raw_heading_relative = wrap_angle(north_heading - raw_heading + HEADING_OFFSET_RAD)
    else:
        raw_heading_relative = wrap_angle(raw_heading - north_heading + HEADING_OFFSET_RAD)

    if prev_heading is None:
        heading = raw_heading_relative
    else:
        heading = angle_lerp(prev_heading, raw_heading_relative, HEADING_LPF_ALPHA)

    prev_heading = heading

    # Instant quantization, used only for debugging now.
    heading_q, heading_q_deg, heading_q_dir = quantize_heading_cardinal(heading)

    # log heading for debugging
    heading_t.append(elapsed)
    raw_heading_deg_log.append(math.degrees(raw_heading))
    rel_heading_deg_log.append(math.degrees(raw_heading_relative))
    smooth_heading_deg_log.append(math.degrees(heading))
    original_yaw_deg_log.append(math.degrees(original_yaw))
    quant_heading_deg_log.append(heading_q_deg)
    quant_heading_dir_log.append(heading_q_dir)
    accepted_heading_deg_log.append(CARDINAL_DEGS[accepted_heading_dir])
    accepted_heading_dir_log.append(accepted_heading_dir)

    cal_sys_log.append(cal[0])
    cal_gyro_log.append(cal[1])
    cal_acc_log.append(cal[2])
    cal_mag_log.append(cal[3])

    # --------------------------------------------------
    # Step signal
    # --------------------------------------------------
    acc_mag = float(np.linalg.norm(a))
    sig_window.append(acc_mag)
    sig_filt = float(np.mean(sig_window))

    baseline_window.append(sig_filt)
    baseline = float(np.mean(baseline_window))

    step_signal = sig_filt - baseline

    # log step-signal info
    sig_t.append(elapsed)
    sig_raw_log.append(acc_mag)
    sig_filt_log.append(sig_filt)
    baseline_log.append(baseline)
    step_signal_log.append(step_signal)

    # --------------------------------------------------
    # Event-based step detection
    # --------------------------------------------------
    if (not in_step_event) and (step_signal > STEP_EVENT_START):
        in_step_event = True
        event_peak_val = step_signal
        event_peak_time = elapsed
        event_start_valley = (
            min(step_signal_log[-10:])
            if len(step_signal_log) >= 10
            else step_signal
        )

    if in_step_event:
        if step_signal > event_peak_val:
            event_peak_val = step_signal
            event_peak_time = elapsed

        if step_signal < STEP_EVENT_END:
            prominence = event_peak_val - event_start_valley
            cooldown_ok = (event_peak_time - last_step_time) >= MIN_STEP_INTERVAL
            prominence_ok = prominence >= MIN_STEP_PROMINENCE
            peak_ok = event_peak_val >= MIN_STEP_PEAK

            if cooldown_ok and prominence_ok and peak_ok:
                t_step = event_peak_time

                step_len = FIXED_STEP_LEN_M
                step_len = clamp(step_len, MIN_STEP_LEN, MAX_STEP_LEN)

                continuous_heading_deg = math.degrees(heading)
                nearest_dir = nearest_cardinal_dir(continuous_heading_deg)

                # --------------------------------------------------
                # Direction hysteresis / direction locking
                # --------------------------------------------------
                if USE_CARDINAL_HEADING:
                    if nearest_dir == accepted_heading_dir:
                        pending_heading_dir = None
                        pending_heading_count = 0
                    else:
                        err_to_new = angle_error_abs_deg(
                            continuous_heading_deg,
                            CARDINAL_DEGS[nearest_dir]
                        )
                        err_from_current = angle_error_abs_deg(
                            continuous_heading_deg,
                            CARDINAL_DEGS[accepted_heading_dir]
                        )

                        should_consider_switch = (
                            err_to_new <= DIRECTION_SWITCH_TO_NEW_DEG
                            and err_from_current >= DIRECTION_LEAVE_CURRENT_DEG
                        )

                        if should_consider_switch:
                            if pending_heading_dir == nearest_dir:
                                pending_heading_count += 1
                            else:
                                pending_heading_dir = nearest_dir
                                pending_heading_count = 1

                            if pending_heading_count >= DIRECTION_CONFIRM_STEPS:
                                print(
                                    f"DIRECTION SWITCH: {accepted_heading_dir} -> {nearest_dir} "
                                    f"at t={t_step:.2f}s, heading={continuous_heading_deg:.1f} deg"
                                )
                                accepted_heading_dir = nearest_dir
                                pending_heading_dir = None
                                pending_heading_count = 0
                        else:
                            pending_heading_dir = None
                            pending_heading_count = 0

                    heading_used_dir = accepted_heading_dir
                    heading_used_deg = CARDINAL_DEGS[accepted_heading_dir]
                    heading_used = cardinal_dir_to_rad(accepted_heading_dir)

                else:
                    heading_used = heading
                    heading_used_deg = math.degrees(heading)
                    heading_used_dir = "continuous"

                # PDR update.
                # heading = 0 means movement in +Y direction.
                dx = step_len * math.sin(heading_used)
                dy = step_len * math.cos(heading_used)

                if FLIP_X_UPDATE:
                    dx = -dx

                x += dx
                y += dy
                steps_total += 1

                last_step_time = t_step

                step_events.append({
                    "t": t_step,
                    "signal_peak": event_peak_val,
                    "prominence": prominence,
                    "step_len": step_len,
                    "raw_heading_deg": math.degrees(raw_heading),
                    "original_yaw_deg": math.degrees(original_yaw),
                    "heading_rad": heading,
                    "heading_deg": continuous_heading_deg,
                    "instant_heading_dir": heading_q_dir,
                    "instant_heading_deg": heading_q_deg,
                    "heading_used_rad": heading_used,
                    "heading_used_deg": heading_used_deg,
                    "heading_used_dir": heading_used_dir,
                    "pending_heading_dir": pending_heading_dir if pending_heading_dir else "",
                    "pending_heading_count": pending_heading_count,
                    "x": x,
                    "y": y,
                    "dx": dx,
                    "dy": dy,
                    "steps_total": steps_total,
                    "cal_sys": cal[0],
                    "cal_gyro": cal[1],
                    "cal_acc": cal[2],
                    "cal_mag": cal[3],
                })

                print(
                    f"STEP {steps_total:03d} @ {t_step:5.2f}s | "
                    f"peak={event_peak_val: .3f} | prom={prominence: .3f} | "
                    f"len={step_len: .3f} m | "
                    f"axis_heading={math.degrees(raw_heading): .1f} deg | "
                    f"rel_heading={continuous_heading_deg: .1f} deg | "
                    f"instant={heading_q_dir} | "
                    f"used={heading_used_dir}({heading_used_deg: .0f} deg) | "
                    f"pending={pending_heading_dir}:{pending_heading_count} | "
                    f"dx={dx: .3f}, dy={dy: .3f} | "
                    f"pos=({x: .3f}, {y: .3f}) | cal={cal}"
                )

            else:
                print(
                    f"REJECT @ {event_peak_time:5.2f}s | "
                    f"peak={event_peak_val: .3f} | prom={prominence: .3f} | "
                    f"cooldown_ok={cooldown_ok} | peak_ok={peak_ok} | prom_ok={prominence_ok}"
                )

            in_step_event = False
            event_peak_val = -1e9
            event_peak_time = None
            event_start_valley = 0.0

    times.append(elapsed)
    xs.append(x)
    ys.append(y)

    # --------------------------------------------------
    # Update Wi-Fi live state for Pi 4
    # --------------------------------------------------
    moving = (elapsed - last_step_time) <= MOVING_TIMEOUT_SEC if last_step_time >= 0 else False

    with state_lock:
        live_state["ok"] = True
        live_state["running"] = True
        live_state["x"] = round(float(x), 4)
        live_state["y"] = round(float(y), 4)
        live_state["steps_total"] = int(steps_total)
        live_state["steps"] = int(steps_total)
        live_state["direction"] = str(accepted_heading_dir)
        live_state["moving"] = bool(moving)
        live_state["t"] = round(float(elapsed), 3)
        live_state["last_step_time"] = None if last_step_time < 0 else round(float(last_step_time), 3)
        live_state["heading_deg"] = round(float(math.degrees(heading)), 2)
        live_state["heading_used_deg"] = round(float(CARDINAL_DEGS[accepted_heading_dir]), 2)
        live_state["instant_direction"] = str(heading_q_dir)
        live_state["instant_heading_deg"] = round(float(heading_q_deg), 2)
        live_state["axis_heading_deg"] = round(float(math.degrees(raw_heading)), 2)
        live_state["original_yaw_deg"] = round(float(math.degrees(original_yaw)), 2)
        live_state["cal_sys"] = int(cal[0])
        live_state["cal_gyro"] = int(cal[1])
        live_state["cal_acc"] = int(cal[2])
        live_state["cal_mag"] = int(cal[3])
        live_state["mode"] = BNO_MODE_NAME
        live_state["body_forward_axis"] = BODY_FORWARD_AXIS

    sleep_time = DT_TARGET - (time.perf_counter() - now)
    if sleep_time > 0:
        time.sleep(sleep_time)


with state_lock:
    live_state["running"] = False
    live_state["moving"] = False


# --------------------------------------------------
# Save CSV
# --------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time_s",
        "x_m",
        "y_m",
        "axis_heading_deg",
        "original_yaw_deg",
        "relative_heading_deg",
        "smoothed_heading_deg",
        "instant_quantized_heading_deg",
        "instant_quantized_heading_dir",
        "accepted_heading_deg",
        "accepted_heading_dir",
        "cal_sys",
        "cal_gyro",
        "cal_acc",
        "cal_mag",
    ])

    n = min(len(times), len(xs), len(ys), len(heading_t))
    for i in range(n):
        writer.writerow([
            times[i],
            xs[i],
            ys[i],
            raw_heading_deg_log[i],
            original_yaw_deg_log[i],
            rel_heading_deg_log[i],
            smooth_heading_deg_log[i],
            quant_heading_deg_log[i],
            quant_heading_dir_log[i],
            accepted_heading_deg_log[i],
            accepted_heading_dir_log[i],
            cal_sys_log[i],
            cal_gyro_log[i],
            cal_acc_log[i],
            cal_mag_log[i],
        ])


# --------------------------------------------------
# XY trajectory plot
# --------------------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(xs, ys, linewidth=2, label="trajectory")

if xs and ys:
    plt.scatter([xs[0]], [ys[0]], s=70, label="start")
    plt.scatter([xs[-1]], [ys[-1]], s=70, label="end")

if step_events:
    step_xs = [e["x"] for e in step_events]
    step_ys = [e["y"] for e in step_events]
    plt.scatter(step_xs, step_ys, s=25, label="steps")

    if LABEL_STEP_TIMES:
        for e in step_events:
            label = f'{e["t"]:.1f}s {e["heading_used_dir"]}'
            plt.text(e["x"], e["y"], label, fontsize=7)

plt.title(f"BNO055 PDR XY Trajectory, Vertical IMU, Cardinal Locked, {BNO_MODE_NAME}")
plt.xlabel("X position (m), East/West relative")
plt.ylabel("Y position (m), North/South relative")
plt.grid(True)
plt.legend()
plt.savefig(OUTPUT_PNG_XY, dpi=150, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Acceleration vs time plot
# --------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(acc_t, acc_x)
axs[0].set_ylabel("ax (m/s^2)")
axs[0].grid(True)
axs[0].set_title(f"BNO055 Linear Acceleration vs Time, {BNO_MODE_NAME}")

axs[1].plot(acc_t, acc_y)
axs[1].set_ylabel("ay (m/s^2)")
axs[1].grid(True)

axs[2].plot(acc_t, acc_z)
axs[2].set_ylabel("az (m/s^2)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_PNG_ACC, dpi=150, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Step-signal plot with detected steps
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(sig_t, sig_filt_log, label="filtered |a|")
plt.plot(sig_t, baseline_log, label="baseline")
plt.plot(sig_t, step_signal_log, label="step_signal")

if step_events:
    step_ts = [e["t"] for e in step_events]
    step_vals = []

    for ts in step_ts:
        idx = min(range(len(sig_t)), key=lambda i: abs(sig_t[i] - ts))
        step_vals.append(step_signal_log[idx])

    plt.scatter(step_ts, step_vals, s=30, label="detected steps")

    if LABEL_STEP_TIMES:
        for e, sv in zip(step_events, step_vals):
            plt.text(e["t"], sv, f'{e["t"]:.1f}s {e["heading_used_dir"]}', fontsize=7)

plt.axhline(STEP_EVENT_START, linestyle="--", linewidth=1, label="step event start")
plt.axhline(MIN_STEP_PEAK, linestyle=":", linewidth=1, label="min step peak")
plt.title(f"BNO055 Step Signal with Detected Steps, {BNO_MODE_NAME}")
plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.grid(True)
plt.legend()
plt.savefig(OUTPUT_PNG_SIGNAL, dpi=150, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Heading debug plot
# --------------------------------------------------
plt.figure(figsize=(11, 6))

plt.plot(heading_t, raw_heading_deg_log, label=f"axis heading from {BODY_FORWARD_AXIS}")
plt.plot(heading_t, original_yaw_deg_log, label="original yaw formula, reference only", alpha=0.7)
plt.plot(heading_t, rel_heading_deg_log, label="relative heading, start = north")
plt.plot(heading_t, smooth_heading_deg_log, label="smoothed heading before quantization")
plt.step(heading_t, quant_heading_deg_log, where="post", label="instant quantized heading")
plt.step(heading_t, accepted_heading_deg_log, where="post", label="accepted locked direction")

if step_events:
    step_ts = [e["t"] for e in step_events]
    step_headings = [e["heading_deg"] for e in step_events]
    step_instant = [e["instant_heading_deg"] for e in step_events]
    step_used = [e["heading_used_deg"] for e in step_events]

    plt.scatter(step_ts, step_headings, s=30, label="continuous heading at steps")
    plt.scatter(step_ts, step_instant, s=40, marker="x", label="instant quantized at steps")
    plt.scatter(step_ts, step_used, s=45, marker="s", label="accepted locked at steps")

plt.axhline(0, linestyle="--", linewidth=1, label="north reference")
plt.axhline(90, linestyle=":", linewidth=1, label="+90 deg, East")
plt.axhline(-90, linestyle=":", linewidth=1, label="-90 deg, West")
plt.axhline(180, linestyle=":", linewidth=1, label="180 deg, South")
plt.axhline(-180, linestyle=":", linewidth=1)

plt.title(f"BNO055 Heading Debug Plot, Vertical IMU, Axis={BODY_FORWARD_AXIS}, {BNO_MODE_NAME}")
plt.xlabel("Time (s)")
plt.ylabel("Heading (degrees)")
plt.grid(True)
plt.legend()
plt.savefig(OUTPUT_PNG_HEADING, dpi=150, bbox_inches="tight")
plt.close()


print(f"\nSaved XY plot to:       {OUTPUT_PNG_XY}")
print(f"Saved accel plot to:    {OUTPUT_PNG_ACC}")
print(f"Saved step signal to:   {OUTPUT_PNG_SIGNAL}")
print(f"Saved heading plot to:  {OUTPUT_PNG_HEADING}")
print(f"Saved CSV to:           {OUTPUT_CSV}")
print(f"Detected steps:         {len(step_events)}")
print(f"Mode used:              {BNO_MODE_NAME}")
print(f"Body forward axis used: {BODY_FORWARD_AXIS}")
print(f"Heading offset used:    {HEADING_OFFSET_DEG:.1f} deg")
print(f"Flipped heading sign:   {USE_FLIPPED_HEADING_SIGN}")
print(f"Cardinal heading used:  {USE_CARDINAL_HEADING}")
print(f"Flip X update:          {FLIP_X_UPDATE}")
print(f"Direction switch to new:{DIRECTION_SWITCH_TO_NEW_DEG:.1f} deg")
print(f"Direction leave current:{DIRECTION_LEAVE_CURRENT_DEG:.1f} deg")
print(f"Direction confirm steps:{DIRECTION_CONFIRM_STEPS}")

print("\nFinal live state being served to Pi 4:")
with state_lock:
    print(json.dumps(live_state, indent=2))

print("\nThe Wi-Fi server is still alive until you press Ctrl+C.")
print(f"Pi 4 can still read: http://<PI_ZERO_IP>:{SERVER_PORT}/position")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nExiting Pi Zero PDR Wi-Fi server.")