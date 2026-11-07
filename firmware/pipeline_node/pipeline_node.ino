// ====================================================================
// Pipeline Monitoring Edge AI System
// Version: 1.0
// Target: ESP32 (ESP32-WROOM-32 DevKitC)
// Author: Ogechukwu Marycynthia Nwankwo
// ====================================================================
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <I2S.h>
#include <TinyGPS++.h>
#include <MCCI_LoRaWAN.h>
// ====================================================================
// PIN DEFINITIONS
// ====================================================================
// Vibration Sensor (ADXL356)
#define PIN_CS_ADXL 5
#define PIN_SCK_ADXL 18
#define PIN_MISO_ADXL 19
#define PIN_MOSI_ADXL 23
// Acoustic Sensor (SPH0645)
#define I2S_BCLK 26
#define I2S_DATA 33
#define I2S_LRCLK 25
// Light Sensors (TEMD5510FX01)
#define PIN_LIGHT_1 36
#define PIN_LIGHT_2 39
#define PIN_LIGHT_3 34
#define PIN_LIGHT_4 35
// GPS
#define PIN_GPS_TX 16
#define PIN_GPS_RX 17
// LoRaWAN (SX1262)
#define PIN_LORA_NSS 15
#define PIN_LORA_SCK 14
#define PIN_LORA_MOSI 13
#define PIN_LORA_MISO 12
#define PIN_LORA_DIO1 27
#define PIN_LORA_BUSY 4
// Power Management
#define PIN_BATTERY 32
// ====================================================================
// OBJECTS
// ====================================================================
SPISettings spiSettings(1000000, MSBFIRST, SPI_MODE0);
TinyGPSPlus gps;
LoRaWAN lorawan;
// ====================================================================
// CONSTANTS
// ====================================================================
#define SAMPLE_RATE 12800
#define BUFFER_SIZE 25600 // 2 seconds
#define WINDOW_SIZE 3276 // 256 ms
#define OVERLAP 0.5
#define FEATURE_SIZE 128
#define NUM_CLASSES 5
#define CONFIDENCE_THRESHOLD 0.7
// ====================================================================
// BUFFERS
// ====================================================================
int16_t vibBuffer[BUFFER_SIZE];
int32_t audioBuffer[BUFFER_SIZE];
int lightBuffer[BUFFER_SIZE];
float features[FEATURE_SIZE];
// ====================================================================
// TENSORFLOW LITE MICRO MODEL
// ====================================================================
// Include quantized model (generated from training)
#include "model_data.h"
// Tensor arena
constexpr int kTensorArenaSize = 200 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
// ====================================================================
// FUNCTION PROTOTYPES
// ====================================================================
void initSensors();
void acquireData();
void preprocessData();
void extractFeatures();
void runInference();
void generateAlert(int class_idx, float confidence);
void sendAlert(struct alert_t alert);
void managePower();
// ====================================================================
// SETUP
// ====================================================================
void setup() {
Serial.begin(115200);
Serial.println("Pipeline Monitoring Edge AI System");
Serial.println("Version 1.0");
// Initialize sensors
initSensors();
// Initialize LoRaWAN
initLoRaWAN();
// Initialize power management
initPowerManagement();
Serial.println("System ready");
}
// ====================================================================
// MAIN LOOP
// ====================================================================
void loop() {
// Acquire sensor data
acquireData();
// Preprocess data
preprocessData();
// Extract features
extractFeatures();
// Run inference
runInference();
// Manage power
managePower();
// Small delay to prevent watchdog
delay(10);
}
// ====================================================================
// SENSOR INITIALIZATION
// ====================================================================
void initSensors() {
// Initialize SPI
SPI.begin(PIN_SCK_ADXL, PIN_MISO_ADXL, PIN_MOSI_ADXL, PIN_CS_ADXL);
// Initialize ADXL356
pinMode(PIN_CS_ADXL, OUTPUT);
digitalWrite(PIN_CS_ADXL, HIGH);
// Configure ADXL356
SPI.beginTransaction(spiSettings);
digitalWrite(PIN_CS_ADXL, LOW);
SPI.transfer(0x00); // Register address
SPI.transfer(0x0A); // Value: 100 Hz bandwidth
digitalWrite(PIN_CS_ADXL, HIGH);
SPI.endTransaction();
// Initialize I2S for acoustic
I2S.begin(I2S_BCLK, I2S_DATA, I2S_LRCLK);
I2S.setFrequency(SAMPLE_RATE);
// Initialize light sensor pins
pinMode(PIN_LIGHT_1, INPUT);
pinMode(PIN_LIGHT_2, INPUT);
pinMode(PIN_LIGHT_3, INPUT);
pinMode(PIN_LIGHT_4, INPUT);
// Initialize I2C for humidity sensor
Wire.begin();
// Initialize GPS
Serial2.begin(9600, SERIAL_8N1, PIN_GPS_TX, PIN_GPS_RX);
Serial.println("Sensors initialized");
}
// ====================================================================
// DATA ACQUISITION
// ====================================================================
void acquireData() {
unsigned long startTime = micros();
unsigned long samplePeriod = 1000000 / SAMPLE_RATE;
for (int i = 0; i < BUFFER_SIZE; i++) {
// Read vibration
digitalWrite(PIN_CS_ADXL, LOW);
SPI.beginTransaction(spiSettings);
SPI.transfer(0x08); // X-axis register
int16_t x = (SPI.transfer(0x00) << 8) | SPI.transfer(0x00);
SPI.endTransaction();
digitalWrite(PIN_CS_ADXL, HIGH);
vibBuffer[i] = x;
// Read acoustic
audioBuffer[i] = I2S.read();
// Read light (averaged over 4 photodiodes)
int l1 = analogRead(PIN_LIGHT_1);
int l2 = analogRead(PIN_LIGHT_2);
int l3 = analogRead(PIN_LIGHT_3);
int l4 = analogRead(PIN_LIGHT_4);
lightBuffer[i] = (l1 + l2 + l3 + l4) / 4;
// Wait for next sample
while (micros() - startTime < samplePeriod) {}
startTime += samplePeriod;
}
}
// ====================================================================
// SIGNAL PROCESSING
// ====================================================================
void preprocessData() {
// Bandpass filter (vibration: 10 Hz - 2 kHz)
// 4th-order Butterworth filter coefficients
// b0, b1, b2, a1, a2
float b[3] = {0.0215, 0.0430, 0.0215};
float a[2] = {-1.7730, 0.8590};
static float vibState[2] = {0, 0};
for (int i = 0; i < BUFFER_SIZE; i++) {
// Apply filter
float filtered = b[0] * vibBuffer[i] +
b[1] * vibState[0] +
b[2] * vibState[1] -
a[0] * vibState[0] -
a[1] * vibState[1];
vibState[1] = vibState[0];
vibState[0] = vibBuffer[i];
vibBuffer[i] = (int16_t)filtered;
}
}
// ====================================================================
// FEATURE EXTRACTION
// ====================================================================
void extractFeatures() {
// Extract 128 features
int featureIdx = 0;
int step = WINDOW_SIZE * (1 - OVERLAP);
int numWindows = (BUFFER_SIZE - WINDOW_SIZE) / step + 1;
// Window 1 (take first window for inference)
int startIdx = 0;
int endIdx = startIdx + WINDOW_SIZE;
// 1. Vibration spectral features (32 bands)
computeVibrationSpectrum(&vibBuffer[startIdx], WINDOW_SIZE, &features[featureIdx]);
featureIdx += 32;
// 2. Acoustic MFCCs (40 coefficients)
computeMFCC(&audioBuffer[startIdx], WINDOW_SIZE, &features[featureIdx]);
featureIdx += 40;
// 3. Light statistics (16 features)
computeLightStats(&lightBuffer[startIdx], WINDOW_SIZE, &features[featureIdx]);
featureIdx += 16;
// 4. Cross-modal features (40 features)
computeCrossModal(&vibBuffer[startIdx], &audioBuffer[startIdx], WINDOW_SIZE, &features[featureIdx]);
featureIdx += 40;
}
// ====================================================================
// INFERENCE
// ====================================================================
void runInference() {
// Load TFLM model
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
error_reporter->Report("Model schema version mismatch");
return;
}
// Build interpreter
static tflite::MicroInterpreter static_interpreter(
model, resolver, tensor_arena, kTensorArenaSize);
tflite::MicroInterpreter* interpreter = &static_interpreter;
// Get input tensor
TfLiteTensor* input = interpreter->input(0);
for (int i = 0; i < FEATURE_SIZE; i++) {
input->data.f[i] = features[i];
}
// Run inference
TfLiteStatus invoke_status = interpreter->Invoke();
if (invoke_status != kTfLiteOk) {
error_reporter->Report("Inference failed");
return;
}
// Get output
TfLiteTensor* output = interpreter->output(0);
int maxIdx = 0;
float maxConf = output->data.f[0];
for (int i = 1; i < NUM_CLASSES; i++) {
if (output->data.f[i] > maxConf) {
maxConf = output->data.f[i];
maxIdx = i;
}
}
// Generate alert if confidence above threshold
if (maxConf > CONFIDENCE_THRESHOLD && maxIdx > 0) {
generateAlert(maxIdx, maxConf);
}
}
// ====================================================================
// ALERT GENERATION
// ====================================================================
struct alert_t {
int class_id;
float confidence;
double latitude;
double longitude;
float battery_soc;
float temperature;
float humidity;
unsigned long timestamp;
};
void generateAlert(int class_idx, float confidence) {
alert_t alert;
alert.class_id = class_idx;
alert.confidence = confidence;
// Get GPS
alert.latitude = gps.location.lat();
alert.longitude = gps.location.lng();
// Get battery status
alert.battery_soc = getBatterySOC();
// Get temperature and humidity
alert.temperature = getTemperature();
alert.humidity = getHumidity();
// Get timestamp
alert.timestamp = millis();
// Send alert
sendAlert(alert);
}
// ====================================================================
// COMMUNICATION
// ====================================================================
void sendAlert(struct alert_t alert) {
// Prepare LoRaWAN message
uint8_t buffer[32];
size_t len = encodeAlert(alert, buffer);
// Try LoRaWAN
bool sent = lorawan.send(buffer, len);
if (!sent) {
// Fallback to satellite
sendSatelliteAlert(alert);
}
}
void sendSatelliteAlert(struct alert_t alert) {
// Prepare message
char msg[128];
snprintf(msg, sizeof(msg),
"ALERT:%d,CONF:%.2f,LAT:%.6f,LON:%.6f,BAT:%.1f,TEMP:%.1f",
alert.class_id, alert.confidence,
alert.latitude, alert.longitude,
alert.battery_soc, alert.temperature);
// Send via Iridium
Serial1.println(msg); // Iridium modem on UART1
Serial.printf("Satellite alert sent: %s\n", msg);
}
// ====================================================================
// POWER MANAGEMENT
// ====================================================================
void managePower() {
static unsigned long lastCheck = 0;
if (millis() - lastCheck < 60000) return; // Check every minute
lastCheck = millis();
float soc = getBatterySOC();
if (soc < 30) {
// Low battery: enter deep sleep
Serial.printf("Low battery: %.1f%%, entering sleep\n", soc);
esp_deep_sleep(3600000000ULL); // Sleep for 1 hour (microseconds)
}
}
float getBatterySOC() {
int adc = analogRead(PIN_BATTERY);
float voltage = adc * 3.3 / 4095.0 * 3.0; // 12-bit ADC, 3:1 voltage divider
float soc = (voltage - 2.8) / (3.6 - 2.8) * 100.0;
return constrain(soc, 0, 100);
}
// ====================================================================
// END OF FIRMWARE
