#include <ESP8266WiFi.h>

// ===== CONFIG =====
const char* ssid = "bnb";     // hotspot name
const char* password = "";    // no password

const int windowSize = 20;    // rolling window size
const int sampleDelay = 200;  // ms between samples
const float motionThreshold = 5.0; // tune for your room

// ===== STATE =====
int rssiWindow[windowSize];
int indexWin = 0;
bool filled = false;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ WiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // CSV header with labels
  Serial.println("time_ms,rssi,var,status");
}

void loop() {
  static unsigned long lastPrint = 0;
  unsigned long now = millis();

  if (now - lastPrint >= sampleDelay) {
    int32_t rssi = WiFi.RSSI();

    // push RSSI into rolling window
    rssiWindow[indexWin] = rssi;
    indexWin = (indexWin + 1) % windowSize;
    if (indexWin == 0) filled = true;

    // compute variance
    float var = 0.0;
    if (filled) {
      float mean = 0;
      for (int i = 0; i < windowSize; i++) mean += rssiWindow[i];
      mean /= windowSize;

      for (int i = 0; i < windowSize; i++) {
        float diff = rssiWindow[i] - mean;
        var += diff * diff;
      }
      var /= windowSize;
    }

    // classify
    String status = (var > motionThreshold) ? "MOTION" : "Still";

    // clean CSV row with label
    Serial.printf("%lu,%d,%.2f,%s\n", now, rssi, var, status.c_str());

    lastPrint = now;
  }
}
