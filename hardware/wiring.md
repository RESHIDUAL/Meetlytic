# ESP8266 Hardware Wiring & Architecture Guide

## 1. System Architecture Overview

The standalone prototype uses an **ESP8266 NodeMCU** micro-controller as the physical meeting-table node. It acquires conversational audio, provides visible LED status indication, renders meeting state on an OLED screen, and streams PCM chunks over local TCP WiFi to the zero-cloud processing pipeline.

```
+----------------------------------------------------------------------+
|                     STANDALONE ESP8266 TABLE NODE                    |
|                                                                      |
|  [INMP441 I2S Mic]   --->  [ESP8266 NodeMCU]  --->  [SSD1306 OLED]   |
|                                  |                                   |
|                             [RGB LED]                                |
|                                  | (Local WiFi TCP Socket)           |
+----------------------------------|-----------------------------------+
                                   v
+----------------------------------------------------------------------+
|                LOCAL DESKTOP / EDGE PROCESSING PIPELINE              |
|                                                                      |
|  [VAD] -> [MFCC/DTW] -> [Context Engine] -> [Classifier] -> [Privacy]|
|                                                              |       |
|                                                     [SQLite Storage] |
+----------------------------------------------------------------------+
```

---

## 2. Hardware Pin Connections (ESP8266 NodeMCU v2/v3)

| Component | Pin Label | ESP8266 Physical / BCM Pin | Description / Electrical Spec |
| :--- | :--- | :--- | :--- |
| **INMP441 MEMS Mic** | `VDD` | **3V3 (3.3V)** | Power Supply (3.3V, < 2 mA) |
| | `GND` | **GND** | System Ground |
| | `SD` (Data) | **GPIO 13 (D7 / HMOSI)** | I2S Serial Audio Data |
| | `SCK` (Clock)| **GPIO 14 (D5 / HSCLK)** | I2S Bit Clock |
| | `WS` (Word) | **GPIO 15 (D8 / HCS)** | I2S Word Select / LR Clock |
| | `L/R` (Sel) | **GND** | Left Channel Select |
| **SSD1306 0.96" OLED**| `VCC` | **3V3** | Display 3.3V Power |
| | `GND` | **GND** | Display Ground |
| | `SDA` | **GPIO 4 (D2)** | I2C Data Line (Internal Pullup) |
| | `SCL` | **GPIO 5 (D1)** | I2C Clock Line (Internal Pullup)|
| **Common-Cathode RGB LED**| `Red Anode` | **GPIO 16 (D0)** | Connected via **330 Ω** resistor |
| | `Green Anode` | **GPIO 12 (D6)** | Connected via **220 Ω** resistor |
| | `Blue Anode` | **GPIO 2 (D4)** | Connected via **220 Ω** resistor |
| | `Cathode` | **GND** | Shared System Ground |
| **Push Button** | `Pin 1` | **GPIO 0 (D3 / FLASH)** | Active LOW (Internal Pullup) |
| | `Pin 2` | **GND** | Ground |

---

## 3. Physical Breadboard Schematic (ASCII Art)

```
        +---------------------------------------------+
        |                ESP8266 NodeMCU              |
        |                                             |
        | [3V3] ----+---------------------+           |
        |           |                     |           |
        | [GND] ----+---------+           |           |
        |           |         |           |           |
        | [D1] (GPIO5)  ------+-----+     |           |
        | [D2] (GPIO4)  ------+---+ |     |           |
        |                     |   | |     |           |
        | [D5] (GPIO14) ------|-+ | |     |           |
        | [D7] (GPIO13) ----+ | | | |     |           |
        | [D8] (GPIO15) --+ | | | | |     |           |
        |                 | | | | | |     |           |
        +-----------------|-|-|-|-|-|-|-----+---------+
                          | | | | | | |     |
                          | | | | | | |     +--> OLED VCC (3.3V)
                          | | | | | | +--------> OLED SCL
                          | | | | | +----------> OLED SDA
                          | | | | +------------> GND (All modules)
                          | | | |
                          | | | +--------------> INMP441 SCK
                          | | +----------------> INMP441 SD
                          +--------------------> INMP441 WS
```

---

## 4. Status Indicator LED States

To ensure full transparency during meetings and prevent covert recording, the LED provides real-time state feedback:

* **GREEN:** Professional / Work-related conversation detected and stored.
* **YELLOW:** Turn currently being analyzed through the DSP / VAD pipeline.
* **RED:** Non-professional / Private conversation detected - **actively discarded and wiped from RAM**.
* **BLUE:** Saving extracted decisions, action items, or generating summary.

---

## 5. Bill of Materials (BOM) & Cost Breakdown

| Component | Model / Part | Source | Approx. Unit Cost |
| :--- | :--- | :--- | :--- |
| **Micro-controller** | ESP8266 NodeMCU ESP-12E Module | Amazon / AliExpress | ~$3.50 |
| **Microphone** | INMP441 I2S Digital MEMS Microphone | InvenSense | ~$2.80 |
| **Display** | 0.96" I2C 128x64 Monochrome OLED | SSD1306 | ~$2.50 |
| **Status LED** | 5mm Common-Cathode RGB LED + Resistors | Generic | ~$0.40 |
| **Control Switch** | 6mm Momentary Tactile Push Button | Generic | ~$0.20 |
| **Battery / Power** | 3.7V 1200mAh LiPo + TP4056 USB Charger | Generic | ~$4.50 |
| **Enclosure** | 3D-Printed Mini Table Puck Case | PLA / PETG | ~$1.50 |
| **TOTAL HARDWARE COST**| | | **~$15.40** |

---

## 6. Flash and Deployment Instructions

1. Install the **Arduino IDE** (or VS Code + PlatformIO).
2. Under **Board Manager**, install `esp8266 by ESP8266 Community`.
3. Under **Library Manager**, install:
   - `Adafruit SSD1306`
   - `Adafruit GFX Library`
4. Open `hardware/esp8266_firmware.ino`.
5. Enter your local **WiFi SSID**, **Password**, and the **Desktop Server IP**.
6. Select Board: `NodeMCU 1.0 (ESP-12E Module)` and Upload over micro-USB.
7. Run `python -m hardware.wifi_bridge` on your desktop to begin streaming.
