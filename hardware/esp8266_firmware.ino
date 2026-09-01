#include <ESP8266WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* DESKTOP_HOST  = "192.168.1.100";
const uint16_t TCP_PORT   = 5555;
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
#define LED_RED_PIN   16
#define LED_GREEN_PIN 12
#define LED_BLUE_PIN  2
#define BUTTON_PIN    0
#define SAMPLE_RATE      16000
#define CHUNK_SAMPLES    800
int16_t audioBuffer[CHUNK_SAMPLES];
WiFiClient tcpClient;
bool isRecording = true;
unsigned long lastDisplayUpdate = 0;
String currentStatus = "LISTENING";
int professionalPct = 80;
int storedItems = 0;
void setLEDColor(uint8_t r, uint8_t g, uint8_t b) {
    analogWrite(LED_RED_PIN, r);
    analogWrite(LED_GREEN_PIN, g);
    analogWrite(LED_BLUE_PIN, b);
}
void showStatusLED(String status) {
    if (status == "PROFESSIONAL") {
        setLEDColor(0, 255, 0);
    } else if (status == "PERSONAL" || status == "DISCARD") {
        setLEDColor(255, 0, 0);
    } else if (status == "ANALYZING") {
        setLEDColor(255, 200, 0);
    } else if (status == "SAVING") {
        setLEDColor(0, 100, 255);
    } else {
        setLEDColor(0, 150, 0);
    }
}
void renderOLED() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("== MEETING FILTER =="));
    display.setCursor(0, 14);
    display.print(F("State : "));
    display.println(currentStatus);
    display.setCursor(0, 26);
    display.print(F("Prof. : "));
    display.print(professionalPct);
    display.println(F("%"));
    display.setCursor(0, 38);
    display.print(F("Stored: "));
    display.print(storedItems);
    display.println(F(" items"));
    display.setCursor(0, 52);
    if (tcpClient.connected()) {
        display.println(F("WiFi: LINKED [TCP]"));
    } else {
        display.println(F("WiFi: CONNECTING..."));
    }
    display.display();
}
void captureAudioChunk() {
    for (int i = 0; i < CHUNK_SAMPLES; i++) {
        int16_t sample = (int16_t)(analogRead(A0) - 512) * 64;
        audioBuffer[i] = sample;
    }
}
void setup() {
    Serial.begin(115200);
    delay(200);
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_GREEN_PIN, OUTPUT);
    pinMode(LED_BLUE_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    setLEDColor(0, 0, 255);
    Wire.begin(4, 5);
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 20);
        display.println(F("Connecting WiFi..."));
        display.display();
    }
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(300);
        Serial.print(".");
    }
    Serial.println(F("\nWiFi Connected. IP: "));
    Serial.println(WiFi.localIP());
    setLEDColor(0, 255, 0);
}
void loop() {
    if (!tcpClient.connected()) {
        currentStatus = "RECONNECTING";
        setLEDColor(255, 100, 0);
        tcpClient.connect(DESKTOP_HOST, TCP_PORT);
        delay(500);
        return;
    }
    if (digitalRead(BUTTON_PIN) == LOW) {
        isRecording = !isRecording;
        delay(300);
    }
    if (isRecording) {
        captureAudioChunk();
        tcpClient.write((uint8_t*)audioBuffer, CHUNK_SAMPLES * sizeof(int16_t));
        if (tcpClient.available()) {
            String feedback = tcpClient.readStringUntil('\n');
            feedback.trim();
            if (feedback.startsWith("PROF:")) {
                currentStatus = "PROFESSIONAL";
                storedItems++;
                showStatusLED("PROFESSIONAL");
            } else if (feedback.startsWith("DISC:")) {
                currentStatus = "DISCARDED";
                showStatusLED("PERSONAL");
            } else if (feedback.startsWith("ANALYZING")) {
                currentStatus = "ANALYZING";
                showStatusLED("ANALYZING");
            }
        }
    } else {
        currentStatus = "PAUSED";
        setLEDColor(200, 200, 0);
    }
    if (millis() - lastDisplayUpdate > 300) {
        lastDisplayUpdate = millis();
        renderOLED();
    }
}
