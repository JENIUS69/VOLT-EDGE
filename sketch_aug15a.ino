#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <DHT.h>

const char* ssid = "VoltEdge_System";

// 8 Appliance Relays (D2 through D9)
const int lightPins[4] = {D2, D4, D6, D8};
const int fanPins[4]   = {D3, D5, D7, D9};

// Reassigned Serial Pins for Siren and Sensor
const int buzzerPin    = D0; // RX pin as Buzzer Relay
#define DHTPIN D1            // TX pin as DHT11 Data Pin
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);
ESP8266WebServer server(80);

void setup() {
  // Serial.begin() is intentionally omitted so D0 and D1 act as standard GPIOs
  dht.begin();

  // Active-LOW Relays for 4 Zones (Set HIGH to keep OFF on boot)
  for (int i = 0; i < 4; i++) {
    pinMode(lightPins[i], OUTPUT);
    digitalWrite(lightPins[i], HIGH);
    
    pinMode(fanPins[i], OUTPUT);
    digitalWrite(fanPins[i], HIGH);
  }

  // Active-LOW Relay for Buzzer Siren (Set HIGH to keep OFF on boot)
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, HIGH);

  WiFi.softAP(ssid);

  // Sensor endpoint
  server.on("/sensor", []() {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
      server.send(500, "application/json", "{\"temp\":\"--\",\"humidity\":\"--\"}");
      return;
    }
    String json = "{\"temp\":" + String(t, 1) + ",\"humidity\":" + String(h, 1) + "}";
    server.send(200, "application/json", json);
  });

  server.onNotFound(handleRequest);
  server.begin();
}

void loop() {
  server.handleClient();
}

void handleRequest() {
  String uri = server.uri();

  // Active-LOW Siren Controls
  if (uri.indexOf("/alarm/on") != -1) {
    digitalWrite(buzzerPin, LOW);   // Relay closes -> Siren ON
    server.send(200, "text/plain", "Alarm ON");
    return;
  } else if (uri.indexOf("/alarm/off") != -1) {
    digitalWrite(buzzerPin, HIGH);  // Relay opens -> Siren OFF
    server.send(200, "text/plain", "Alarm OFF");
    return;
  }

  // Active-LOW Relay Parsing for 4 Zones (Light & Fan)
  for (int i = 1; i <= 4; i++) {
    String zStr = "/zone" + String(i);
    if (uri.indexOf(zStr) != -1) {
      int idx = i - 1;
      if (uri.indexOf("/light/on") != -1)       digitalWrite(lightPins[idx], LOW);
      else if (uri.indexOf("/light/off") != -1)  digitalWrite(lightPins[idx], HIGH);
      else if (uri.indexOf("/fan/on") != -1)     digitalWrite(fanPins[idx], LOW);
      else if (uri.indexOf("/fan/off") != -1)    digitalWrite(fanPins[idx], HIGH);
      
      server.send(200, "text/plain", "Zone " + String(i) + " Updated");
      return;
    }
  }

  server.send(404, "text/plain", "Not Found");
}