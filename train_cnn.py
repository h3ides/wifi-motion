import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -------------------------
# Parameters
# -------------------------
WINDOW = 40       # samples per window
STEP = 10         # overlap step
BATCH_SIZE = 32
EPOCHS = 35
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization strength

# -------------------------
# CNN Model (Improved)
# -------------------------
class EnhancedCNN1D(nn.Module):
    def __init__(self, in_channels=2, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------
# Data Preparation
# -------------------------
def windows_from_df(df):
    X, y = [], []
    if "var" not in df.columns:
        df["var"] = (df["rssi"].diff().fillna(0)**2)

    # optional small smoothing to reduce noise
    df["rssi"] = df["rssi"].rolling(window=3, min_periods=1).mean()

    for start in range(0, len(df) - WINDOW + 1, STEP):
        win = df.iloc[start:start + WINDOW]
        label = win["label"].iloc[-1]
        if label == "Unlabeled":
            continue

        ch1 = win["rssi"].values.astype(np.float32)
        ch2 = win["var"].values.astype(np.float32)

        # normalize each channel
        ch1 = (ch1 - np.mean(ch1)) / (np.std(ch1) + 1e-6)
        ch2 = (ch2 - np.mean(ch2)) / (np.std(ch2) + 1e-6)

        # light noise augmentation for robustness
        if np.random.rand() < 0.3:
            ch1 += np.random.normal(0, 0.05, size=ch1.shape)
            ch2 += np.random.normal(0, 0.05, size=ch2.shape)

        X.append(np.stack([ch1, ch2], axis=0))
        y.append(label)

    return np.array(X), np.array(y)


Xs, ys = [], []
for f in glob.glob("logs/*.csv"):
    df = pd.read_csv(f)
    if "label" not in df.columns:
        continue
    Xf, yf = windows_from_df(df)
    if len(Xf) > 0:
        Xs.append(Xf)
        ys.append(yf)

if len(Xs) == 0:
    raise RuntimeError("No valid labeled data found!")

X = np.concatenate(Xs, axis=0)
y = np.concatenate(ys, axis=0)

# preserve natural label order (don’t sort alphabetically)
labels = list(dict.fromkeys(y))
label2idx = {l: i for i, l in enumerate(labels)}
y_idx = np.array([label2idx[v] for v in y])

print(f"Dataset: {X.shape[0]} windows | Labels: {labels}")

# -------------------------
# Torch Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_idx, test_size=0.2, stratify=y_idx, random_state=42
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# -------------------------
# Training Loop
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN1D(in_channels=2, n_classes=len(labels)).to(device)
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
crit = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device).float(), yb.to(device).long()
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    # Eval
    model.eval()
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            ys_pred.append(preds)
            ys_true.append(yb.numpy())

    ys_pred = np.concatenate(ys_pred)
    ys_true = np.concatenate(ys_true)
    acc = (ys_pred == ys_true).mean()
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss={total_loss/len(train_loader):.4f} | Test acc={acc:.3f}")

# -------------------------
# Final Evaluation
# -------------------------
print("\nClassification Report:")
print(classification_report(ys_true, ys_pred, target_names=labels))

# -------------------------
# Save Model and Labels
# -------------------------
torch.save(model.state_dict(), "cnn_model.pth")
joblib.dump(labels, "cnn_labels.joblib")
print("✅ Saved cnn_model.pth and cnn_labels.joblib")
