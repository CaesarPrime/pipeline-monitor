# ESP32 field-node firmware

`pipeline_node/pipeline_node.ino` is the firmware listed in Section 4.3 of the project report.
It is provided as the reference design for the physical node and is **not** needed to run the
simulation dashboard.

To build it:

1. Install the Arduino IDE 2.x and add the ESP32 board package
   (`https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`).
2. Install the TensorFlow Lite for Microcontrollers, MCCI LoRaWAN LMIC and TinyGPS++ libraries.
3. Run `python train.py` in the repository root and copy `models/model_data.h` into this folder.
4. Select **ESP32 Dev Module** and upload.

The pin assignments follow Tables 4.1 to 4.5 of the report.
