# 🧠 WiFi-Motion: RSSI-Based Human Activity Recognition

This project demonstrates **WiFi-based motion detection** using **RSSI (Received Signal Strength Indicator)** values from an ESP8266 module.  
It applies **machine learning and deep learning (1D CNN)** to classify motion states such as **Sit, Walk, and Leave** — using only signal variations, with no cameras or wearable sensors.

---

## 📁 Repository Overview

| File | Description |
|------|--------------|
| **wifi.ino** | Arduino sketch for ESP8266. Continuously scans WiFi RSSI and transmits readings via serial. |
| **logger.py** | Python script for real-time serial logging and live visualization. It allows keyboard-based labeling (1=Still, 2=Sit, 3=Walk, 4=Leave) while saving to CSV in the `logs/` folder. |
| **train_cnn.py** | Trains a lightweight 1D CNN model on labeled RSSI data. Saves the model (`cnn_model.pth`) and label map (`cnn_labels.joblib`) for inference. |
| **predict_live_cnn.py** | Runs real-time inference from the ESP8266 serial stream using the trained CNN. Displays live signal and predicted activity using Matplotlib. |
| **logs/** | Folder where logged CSV training data is automatically stored. |
| **requirements.txt** | Python dependencies for model training and live inference. |
| **Wifi Motion.pdf** | Summary documentation and experiment overview. |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/h3ides/wifi-motion.git
cd wifi-motion
2️⃣ Create Virtual Environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Requirements
pip install -r requirements.txt
4️⃣ Flash ESP8266
Open wifi.ino in Arduino IDE.

Set the correct COM port and board (e.g. NodeMCU 1.0).

Upload to your ESP8266.

🧩 Data Logging
Run:
python logger.py
View live RSSI graph in real-time.

Label data using keys:
1 → Sit

2 → Walk

3 → Leave

Data will automatically save in logs/data_<timestamp>.csv.

🧠 Training the CNN
Run:
python train_cnn.py
The script loads all CSVs from logs/

Trains a compact 1D CNN on windowed RSSI data.

Saves:

cnn_model.pth → Trained model weights

cnn_labels.joblib → Label names

🔮 Real-Time Prediction
Once the model is trained:
python predict_live_cnn.py
Reads live RSSI from ESP8266 (COM3 by default).

Continuously predicts activity.

Displays:

Live RSSI graph.

Real-time predicted activity ( Sit / Walk / Leave).

🧰 Requirements
All dependencies are listed in requirements.txt:
torch
pandas
numpy
matplotlib
joblib
pyserial
📡 How It Works
WiFi Signal Source: ESP8266 continuously scans local RSSI.

Data Logging: Python captures RSSI + variance through serial.

Labeling: Human provides activity label with keypress.

Feature Extraction / CNN: Model learns temporal patterns in RSSI.

Live Inference: The trained model predicts motion state in real-time.

📊 Activity Labels
Label	Meaning
1	Still
2	Sit
3	Walk
4	Leave

🧩 Example Use-Cases
Indoor crowd presence monitoring

Smart-room automation

Privacy-preserving motion detection

Low-cost occupancy sensing



📜 License
This project is released under the MIT License — free for academic and personal use.