#!/usr/bin/env python3

"""
Pi 4 path plotting + Pi Zero person tracking receiver.

What this does:
1. Loads the robot planned path from a CSV file.
2. Polls the Pi Zero over Wi-Fi for person/PDR coordinates.
3. Stores the received person coordinates.
4. When stopped, saves a plot showing:
   - planned robot path
   - person tracked path
   - start/end markers
5. Saves received person tracking data to a CSV file.

Expected Pi Zero endpoint:
    http://PI_ZERO_IP:8000/position

Expected JSON from Pi Zero:
    {
        "x": 0.0,
        "y": 0.0,
        "steps_total": 0,
        "direction": "N",
        "moving": false,
        "t": 0.0
    }
"""

import csv
import json
import os
import time
import threading
import urllib.request
import urllib.error

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIGURATION
# ============================================================

# Planned robot path CSV from mapping/path planning
PLANNED_PATH_CSV = "/tmp/planned_path.csv"

# Output files
OUTPUT_IMAGE = "/home/pi/ros2_ws/src/images/path_plot_with_person3.png"
PERSON_LOG_CSV = "/home/pi/ros2_ws/src/images/person_tracking_log3.csv"

# Pi Zero server info
# CHANGE THIS to the Pi Zero's actual IP address
PI_ZERO_IP = "10.49.245.84"
PI_ZERO_PORT = 8000
PI_ZERO_POSITION_ENDPOINT = f"http://{PI_ZERO_IP}:{PI_ZERO_PORT}/position"

# How often Pi 4 asks Pi Zero for person tracking data
POLL_PERIOD_SEC = 0.25

# Optional coordinate scaling/offsets
# Use these if the Pi Zero PDR coordinate frame needs to be aligned to the map.
PERSON_X_SCALE = 1.0
PERSON_Y_SCALE = 1.0
PERSON_X_OFFSET = 0.0
PERSON_Y_OFFSET = 0.0

# If the person path is mirrored, change these to True
FLIP_PERSON_X = False
FLIP_PERSON_Y = False

# If the person path is rotated relative to the planned path,
# use this. Try 0, 90, -90, or 180.
PERSON_ROTATION_DEG = 0.0


# ============================================================
# STOP CONTROL
# ============================================================

stop_event = threading.Event()


def wait_for_stop():
    input("\nPress Enter to stop recording and save the final plot...\n")
    stop_event.set()


# ============================================================
# HELPERS
# ============================================================

def transform_person_xy(x, y):
    """
    Apply simple transform to align Pi Zero PDR coordinates with the plotted map.

    Order:
    1. optional flip x/y
    2. rotation
    3. scale
    4. offset
    """
    import math

    x = float(x)
    y = float(y)

    if FLIP_PERSON_X:
        x = -x

    if FLIP_PERSON_Y:
        y = -y

    theta = math.radians(PERSON_ROTATION_DEG)

    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)

    xr = PERSON_X_SCALE * xr + PERSON_X_OFFSET
    yr = PERSON_Y_SCALE * yr + PERSON_Y_OFFSET

    return xr, yr


def load_planned_path(csv_path):
    """
    Load planned path CSV.

    Supports:
    - CSV with x,y columns
    - CSV with any first two numeric columns
    """
    if not os.path.exists(csv_path):
        print(f"WARNING: Planned path CSV not found: {csv_path}")
        print("The final plot will only show the person path.")
        return None, None, None

    df = pd.read_csv(csv_path)

    if len(df.columns) < 2:
        raise ValueError("Planned path CSV must have at least two columns.")

    cols = df.columns.tolist()

    x_col = "x" if "x" in cols else cols[0]
    y_col = "y" if "y" in cols else cols[1]

    print(f"Loaded planned path from: {csv_path}")
    print(f"Using planned path columns: {x_col}, {y_col}")
    print(f"Planned path points: {len(df)}")

    return df, x_col, y_col


def fetch_pi_zero_position():
    """
    Poll Pi Zero /position endpoint and return parsed JSON.
    """
    with urllib.request.urlopen(PI_ZERO_POSITION_ENDPOINT, timeout=2.0) as response:
        raw = response.read().decode("utf-8")
        data = json.loads(raw)
        return data


def save_person_log(person_records, output_csv):
    """
    Save received Pi Zero tracking data to CSV.
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "pi4_time_s",
            "pi_zero_time_s",
            "x_raw",
            "y_raw",
            "x_plot",
            "y_plot",
            "steps_total",
            "direction",
            "moving",
            "last_step_time",
        ])

        for r in person_records:
            writer.writerow([
                r.get("pi4_time_s", ""),
                r.get("pi_zero_time_s", ""),
                r.get("x_raw", ""),
                r.get("y_raw", ""),
                r.get("x_plot", ""),
                r.get("y_plot", ""),
                r.get("steps_total", ""),
                r.get("direction", ""),
                r.get("moving", ""),
                r.get("last_step_time", ""),
            ])

    print(f"Person tracking log saved to: {output_csv}")


def save_final_plot(planned_df, x_col, y_col, person_records, output_image):
    """
    Save final plot with planned path and person path.
    """
    plt.figure(figsize=(9, 9))

    # ------------------------------------------------------------
    # Plot planned robot path
    # ------------------------------------------------------------
    if planned_df is not None and len(planned_df) > 0:
        plt.plot(
            planned_df[x_col],
            planned_df[y_col],
            "b-",
            alpha=0.6,
            linewidth=2,
            label="Planned Robot Path"
        )

        plt.scatter(
            planned_df[x_col],
            planned_df[y_col],
            c=range(len(planned_df)),
            cmap="viridis",
            s=12,
            alpha=0.8,
            label="Planned Path Points"
        )

        plt.plot(
            planned_df[x_col].iloc[0],
            planned_df[y_col].iloc[0],
            "go",
            markersize=10,
            label="Robot Path Start"
        )

        plt.plot(
            planned_df[x_col].iloc[-1],
            planned_df[y_col].iloc[-1],
            "ro",
            markersize=10,
            label="Robot Path End"
        )

    # ------------------------------------------------------------
    # Plot person/PDR path
    # ------------------------------------------------------------
    if len(person_records) > 0:
        person_x = [r["x_plot"] for r in person_records]
        person_y = [r["y_plot"] for r in person_records]

        plt.plot(
            person_x,
            person_y,
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Person Tracked Path"
        )

        plt.scatter(
            person_x,
            person_y,
            c=range(len(person_x)),
            cmap="plasma",
            s=18,
            alpha=0.9,
            label="Person Tracking Points"
        )

        plt.plot(
            person_x[0],
            person_y[0],
            marker="o",
            markersize=10,
            color="lime",
            label="Person Start"
        )

        plt.plot(
            person_x[-1],
            person_y[-1],
            marker="x",
            markersize=12,
            color="red",
            label="Person End"
        )

        # Label every few points so the path direction is visible
        label_every = max(1, len(person_records) // 10)

        for i, r in enumerate(person_records):
            if i % label_every == 0 or i == len(person_records) - 1:
                label = str(r.get("direction", ""))
                plt.text(
                    r["x_plot"],
                    r["y_plot"],
                    label,
                    fontsize=8
                )

    else:
        print("WARNING: No person records were received. Plot will only show planned path.")

    # ------------------------------------------------------------
    # Plot formatting
    # ------------------------------------------------------------
    plt.title("Planned Robot Path with Pi Zero Person Tracking Overlay")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.axis("equal")
    plt.legend(loc="best")

    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Final plot saved to: {output_image}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("==============================================")
    print("  Pi 4 Planned Path + Person Tracking Plotter")
    print("==============================================")
    print(f"Planned path CSV: {PLANNED_PATH_CSV}")
    print(f"Pi Zero endpoint: {PI_ZERO_POSITION_ENDPOINT}")
    print(f"Output image:     {OUTPUT_IMAGE}")
    print(f"Person log CSV:   {PERSON_LOG_CSV}")
    print()

    planned_df, x_col, y_col = load_planned_path(PLANNED_PATH_CSV)

    person_records = []

    stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
    stop_thread.start()

    print("Starting Pi Zero polling.")
    print("Walk with the Pi Zero/BNO055 now.")
    print("Press Enter when finished to save the plot.")
    print()

    start_time = time.perf_counter()
    last_good_time = None
    poll_count = 0
    error_count = 0

    while not stop_event.is_set():
        loop_start = time.perf_counter()
        pi4_time = loop_start - start_time

        try:
            data = fetch_pi_zero_position()
            poll_count += 1
            last_good_time = pi4_time

            # Pull position from Pi Zero JSON
            x_raw = float(data.get("x", 0.0))
            y_raw = float(data.get("y", 0.0))

            x_plot, y_plot = transform_person_xy(x_raw, y_raw)

            record = {
                "pi4_time_s": round(pi4_time, 3),
                "pi_zero_time_s": data.get("t", data.get("time_s", "")),
                "x_raw": x_raw,
                "y_raw": y_raw,
                "x_plot": x_plot,
                "y_plot": y_plot,
                "steps_total": data.get("steps_total", data.get("steps", "")),
                "direction": data.get("direction", data.get("heading_used_dir", "")),
                "moving": data.get("moving", ""),
                "last_step_time": data.get("last_step_time", ""),
            }

            person_records.append(record)

            print(
                f"RX {poll_count:04d} | "
                f"x={x_raw: .2f}, y={y_raw: .2f} "
                f"-> plot=({x_plot: .2f}, {y_plot: .2f}) | "
                f"steps={record['steps_total']} | "
                f"dir={record['direction']} | "
                f"moving={record['moving']}"
            )

        except urllib.error.URLError as e:
            error_count += 1
            print(f"Connection error #{error_count}: Could not reach Pi Zero: {e}")

        except json.JSONDecodeError as e:
            error_count += 1
            print(f"JSON error #{error_count}: Pi Zero returned invalid JSON: {e}")

        except Exception as e:
            error_count += 1
            print(f"Error #{error_count}: {e}")

        sleep_time = POLL_PERIOD_SEC - (time.perf_counter() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print()
    print("Stopping receiver.")
    print(f"Total good polls: {poll_count}")
    print(f"Total errors:     {error_count}")

    if last_good_time is not None:
        print(f"Last good packet: {last_good_time:.2f} s after start")
    else:
        print("No successful packets were received from Pi Zero.")

    save_person_log(person_records, PERSON_LOG_CSV)
    save_final_plot(planned_df, x_col, y_col, person_records, OUTPUT_IMAGE)

    print()
    print("Done.")
    print(f"Copy/view this image from your computer:")
    print(f"  {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()