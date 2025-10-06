import serial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import os
import sys
import threading
from collections import deque

# ----------------------------
# Serial Setup
# ----------------------------
PORT = "COM3"
BAUD = 115200
print(f"Connecting on {PORT} at {BAUD}...")
ser = serial.Serial(PORT, BAUD, timeout=1)

# ----------------------------
# CSV Setup
# ----------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"logs/rssi_log_{timestamp}.csv"

columns = ["time_ms", "rssi", "var", "status", "label"]
with open(csv_file, "w") as f:
    f.write(",".join(columns) + "\n")
print(f"Logging to {csv_file}")

# ----------------------------
# Shared state
# ----------------------------
buffer = []
plot_window = 500
times = deque(maxlen=plot_window)
rssis = deque(maxlen=plot_window)
vars_ = deque(maxlen=plot_window)
current_label = "Unlabeled"

# ----------------------------
# Keypress listener
# ----------------------------
def key_listener():
    global current_label
    print("\nLabeling controls: 1=Sit, 2=Walk, 3=Leave\n")
    while True:
        key = sys.stdin.read(1)
        if key == "1":
            current_label = "Sit"
        elif key == "2":
            current_label = "Walk"
        elif key == "3":
            current_label = "Leave"
        print(f"[Label changed] → {current_label}")

threading.Thread(target=key_listener, daemon=True).start()

# ----------------------------
# Live Plot Setup
# ----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))
line_rssi, = ax1.plot([], [], lw=2, label="RSSI")
line_var, = ax2.plot([], [], lw=2, color="orange", label="Variance")

ax1.set_ylabel("RSSI (dBm)")
ax1.set_title("ESP8266 WiFi Motion Detection (Live)")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("VAR")
ax2.grid(True)
ax2.legend()

def update(frame):
    global buffer
    try:
        line_raw = ser.readline().decode(errors="ignore").strip()
        if not line_raw:
            return line_rssi, line_var

        parts = line_raw.split(",")
        if len(parts) != 4:
            return line_rssi, line_var

        time_ms, rssi, var, status = parts
        row = {
            "time_ms": int(time_ms),
            "rssi": int(rssi),
            "var": float(var),
            "status": status,
            "label": current_label
        }

        times.append(row["time_ms"])
        rssis.append(row["rssi"])
        vars_.append(row["var"])
        buffer.append(",".join(map(str, row.values())) + "\n")

        if len(buffer) >= 50:
            with open(csv_file, "a") as f:
                f.writelines(buffer)
            buffer.clear()

        # Update both plots
        line_rssi.set_data(times, rssis)
        line_var.set_data(times, vars_)
        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

    except Exception as e:
        print("Parse error:", e)

    return line_rssi, line_var

ani = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
plt.show()

if buffer:
    with open(csv_file, "a") as f:
        f.writelines(buffer)
