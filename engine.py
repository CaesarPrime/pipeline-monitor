"""
engine.py
---------
Simulation engine behind the dashboard. Each call to step() advances a
simulated clock, updates the power state of the 15 nodes (solar harvest,
consumption, battery), generates sensor events, classifies them with the
trained model exactly as the field node would, and writes alerts and health
logs to the database.
"""

import os
from datetime import datetime, timedelta

import joblib
import numpy as np

from database import connect, init_db
from features import extract
from simulator import CLASSES, THREAT_CLASSES, generate_event

CONF_THRESHOLD = 0.7
BATTERY_WH = 57.6          # 18,000 mAh x 3.2 V LiFePO4
LOAD_W = 0.148             # average consumption (148 mW)
PANEL_PEAK_W = 2.2         # effective peak harvest of the 15 W panel under cloud cover


class Classifier:
    """Loads the deployed model: the TCN if TensorFlow is available, otherwise the Random Forest."""

    def __init__(self):
        self.scaler = joblib.load("models/scaler.joblib")
        self.name, self._keras = "Random Forest", None
        if os.path.exists("models/tcn.keras"):
            try:
                from tensorflow import keras
                self._keras = keras.models.load_model("models/tcn.keras")
                self.name = "TCN"
            except Exception:
                self._keras = None
        if self._keras is None:
            self._rf = joblib.load("models/rf.joblib")
            self._rf.set_params(n_jobs=1)

    def predict_proba(self, features):
        x = self.scaler.transform(features.reshape(1, -1))
        if self._keras is not None:
            return np.asarray(self._keras(x[..., None], training=False))[0]
        return self._rf.predict_proba(x)[0]


class Simulation:
    def __init__(self, seed=None, reset_db=True):
        self.rng = np.random.default_rng(seed)
        self.con = init_db(reset=reset_db)
        self.model = Classifier()
        self.clock = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
        self.devices = [dict(r) for r in self.con.execute("SELECT * FROM device")]
        for d in self.devices:
            d["soc"] = float(self.rng.uniform(70, 95))
            d["temp"] = float(self.rng.uniform(28, 32))
            d["hum"] = float(self.rng.uniform(80, 92))
        self.last_event = None
        self.tick_minutes = 10
        self.event_rate = 1.5          # expected events per tick across the network
        self.threat_share = 0.35       # share of events that are real threats

    # -------------------------------------------------------------- power model
    def _solar_w(self):
        h = self.clock.hour + self.clock.minute / 60
        if 7 <= h <= 18:
            sun = np.sin(np.pi * (h - 7) / 11)
            cloud = self.rng.uniform(0.3, 1.0)          # monsoon cloud cover
            return PANEL_PEAK_W * sun * cloud
        return 0.0

    def _update_devices(self):
        hours = self.tick_minutes / 60
        solar = self._solar_w()
        h = self.clock.hour
        for d in self.devices:
            if d["status"] == "Offline":
                continue
            net_wh = (solar - LOAD_W) * hours
            d["soc"] = float(np.clip(d["soc"] + net_wh / BATTERY_WH * 100, 0, 100))
            day = 7 <= h <= 18
            d["temp"] = float(np.clip(d["temp"] + self.rng.normal(0.15 if day else -0.15, 0.2), 24, 42))
            d["hum"] = float(np.clip(d["hum"] + self.rng.normal(-0.3 if day else 0.3, 0.5), 60, 98))
            d["status"] = "Low power (sleep)" if d["soc"] < 30 else "Online"

    def _log_health(self):
        t = self.clock.isoformat(sep=" ", timespec="minutes")
        self.con.executemany(
            "INSERT INTO health_log (device_id, battery_soc, temperature, humidity, log_time) VALUES (?,?,?,?,?)",
            [(d["device_id"], round(d["soc"], 1), round(d["temp"], 1), round(d["hum"], 1), t) for d in self.devices])

    # -------------------------------------------------------------- events
    def _random_class(self):
        if self.rng.random() > self.threat_share:
            return 0
        return int(self.rng.choice([1, 2, 3, 4], p=[0.35, 0.2, 0.15, 0.3]))

    def process_event(self, device, true_class):
        night = not (7 <= self.clock.hour < 18)
        ev = generate_event(true_class, self.rng, night=night)
        probs = self.model.predict_proba(extract(ev))
        pred, conf = int(np.argmax(probs)), float(np.max(probs))
        alerted = pred in THREAT_CLASSES and conf > CONF_THRESHOLD
        t = (self.clock + timedelta(minutes=float(self.rng.uniform(0, self.tick_minutes)))).isoformat(
            sep=" ", timespec="seconds")
        self.con.execute(
            "INSERT INTO event_log (device_id, true_class, predicted_class, confidence, alerted, event_time) "
            "VALUES (?,?,?,?,?,?)", (device["device_id"], true_class, pred, conf, int(alerted), t))
        if alerted:
            channel = "LoRaWAN" if self.rng.random() < 0.93 else "Satellite (backup)"
            self.con.execute(
                "INSERT INTO alert (device_id, class_id, confidence, latitude, longitude, channel, event_time, true_class) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (device["device_id"], pred, round(conf, 3), device["latitude"], device["longitude"],
                 channel, t, true_class))
        self.last_event = {"event": ev, "device": device["device_id"], "true": true_class,
                           "pred": pred, "conf": conf, "probs": probs.tolist(), "alerted": alerted, "time": t}
        return self.last_event

    def step(self):
        self.clock += timedelta(minutes=self.tick_minutes)
        self._update_devices()
        self._log_health()
        active = [d for d in self.devices if d["status"] == "Online"]
        for _ in range(self.rng.poisson(self.event_rate)):
            if active:
                self.process_event(active[self.rng.integers(len(active))], self._random_class())
        self.con.commit()

    def inject(self, device_id, true_class):
        """Manually trigger an event at a chosen node (demonstration button)."""
        dev = next(d for d in self.devices if d["device_id"] == device_id)
        out = self.process_event(dev, true_class)
        self.con.commit()
        return out


def class_name(i):
    return CLASSES[int(i)]
