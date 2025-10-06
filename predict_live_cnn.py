# predict_live_cnn.py
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
import joblib
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# Optimized CNN Model
# -------------------------
class SmallCNN1D(nn.Module):
    def __init__(self, in_channels=2, n_classes=4):
        super().__init__()
        # Larger filters, more stable gradients, faster convergence
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # Feature extractor
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.max_pool1d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.max_pool1d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = self.pool(x).squeeze(-1)

        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# Load trained model and labels
# -------------------------
labels = joblib.load("cnn_labels.joblib")
num_classes = len(labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallCNN1D(in_channels=2, n_classes=num_classes).to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()
print(f"✅ Model loaded successfully | Classes: {labels}")

# -------------------------
# Serial config
# -------------------------
PORT = "COM3"         # change if needed
BAUD = 115200
WINDOW_SIZE = 40       # must match training
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"✅ Serial connected on {PORT} @ {BAUD} baud")

# -------------------------
# Buffers for live data
# -------------------------
buffer = deque(maxlen=WINDOW_SIZE)
rssis = deque(maxlen=300)
preds = deque(maxlen=300)

# -------------------------
# Preprocess function
# -------------------------
def preprocess_window(df):
    arr = df[["rssi", "var"]].values.T  # shape (2, window)
    # Normalize channels per window
    arr[0] = (arr[0] - np.mean(arr[0])) / (np.std(arr[0]) + 1e-6)
    arr[1] = (arr[1] - np.mean(arr[1])) / (np.std(arr[1]) + 1e-6)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor

# -------------------------
# Matplotlib setup
# -------------------------
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

line_rssi, = ax1.plot([], [], label="RSSI", lw=1.5)
ax1.set_ylabel("RSSI (dBm)")
ax1.grid(True)
ax1.legend()

ax2.set_ylabel("Predicted Label")
ax2.set_xlabel("Time Steps")
ax2.set_ylim(-0.5, num_classes - 0.5)
ax2.set_yticks(range(num_classes))
ax2.set_yticklabels(labels)
ax2.grid(True)
pred_line, = ax2.plot([], [], "ro-", markersize=4, label="Prediction")
ax2.legend()

# -------------------------
# Live update function
# -------------------------
def update(frame):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return line_rssi, pred_line

        # Expecting: time_ms,rssi,var,...
        parts = line.split(",")
        if len(parts) < 3:
            return line_rssi, pred_line

        rssi = int(parts[1])
        var = float(parts[2])

        buffer.append({"rssi": rssi, "var": var, "time": time.time()})
        rssis.append(rssi)

        # Only predict when window is full
        if len(buffer) == WINDOW_SIZE:
            df_window = pd.DataFrame(buffer)
            X = preprocess_window(df_window)

            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_label = labels[pred_idx]

            preds.append(pred_idx)
            print(f"Predicted: {pred_label} | Probabilities: {dict(zip(labels, probs.round(2)))}")

        # Update RSSI line
        line_rssi.set_data(range(len(rssis)), list(rssis))
        ax1.set_xlim(0, len(rssis))
        ax1.set_ylim(min(rssis)-5, max(rssis)+5)

        # Update prediction line
        if len(preds) > 0:
            pred_line.set_data(range(len(preds)), list(preds))
            ax2.set_xlim(0, len(preds))

        return line_rssi, pred_line

    except Exception as e:
        print("⚠️ Error:", e)
        return line_rssi, pred_line

# -------------------------
# Run animation
# -------------------------
ani = FuncAnimation(fig, update, interval=200)
plt.tight_layout()
plt.show()
