"""
database.py
-----------
SQLite database that follows the entity-relationship design in Section 3.5:
SITE, DEVICE, EVENT_CLASS, ALERT, HEALTH_LOG, USER and RESPONSE.
"""

import hashlib
import sqlite3
from datetime import datetime

from simulator import CLASSES

DB_PATH = "pipeline.db"

# 15 nodes in five clusters along a pipeline corridor in Rivers State.
# Coordinates are approximate and are used for simulation only.
SITES = [
    (1, "Bodo Creek", 1, "Mangrove", "Water"),
    (2, "Okrika", 2, "Urban/riverine", "Road and water"),
    (3, "Eleme", 3, "Industrial", "Road"),
    (4, "Ogoni", 4, "Agricultural", "Road"),
    (5, "Bonny", 5, "Coastal", "Water"),
]
SITE_CENTRES = {1: (4.620, 7.270), 2: (4.742, 7.083), 3: (4.790, 7.120), 4: (4.670, 7.370), 5: (4.450, 7.170)}
PRIORITY = {0: "None", 1: "Critical", 2: "High", 3: "Critical", 4: "Medium"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS site (
    site_id INTEGER PRIMARY KEY, site_name TEXT, cluster_no INTEGER,
    terrain_type TEXT, access_mode TEXT);
CREATE TABLE IF NOT EXISTS device (
    device_id TEXT PRIMARY KEY, dev_eui TEXT, site_id INTEGER REFERENCES site(site_id),
    latitude REAL, longitude REAL, firmware_version TEXT, install_date TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS event_class (
    class_id INTEGER PRIMARY KEY, class_name TEXT, priority_level TEXT);
CREATE TABLE IF NOT EXISTS alert (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT REFERENCES device(device_id),
    class_id INTEGER REFERENCES event_class(class_id), confidence REAL,
    latitude REAL, longitude REAL, channel TEXT, event_time TEXT,
    verification_status TEXT DEFAULT 'Pending', true_class INTEGER);
CREATE TABLE IF NOT EXISTS health_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT REFERENCES device(device_id),
    battery_soc REAL, temperature REAL, humidity REAL, log_time TEXT);
CREATE TABLE IF NOT EXISTS user (
    user_id INTEGER PRIMARY KEY, full_name TEXT, role TEXT, phone_number TEXT, password_hash TEXT);
CREATE TABLE IF NOT EXISTS response (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER REFERENCES alert(alert_id),
    user_id INTEGER REFERENCES user(user_id), action_taken TEXT, response_time TEXT);
CREATE TABLE IF NOT EXISTS event_log (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT, true_class INTEGER,
    predicted_class INTEGER, confidence REAL, alerted INTEGER, event_time TEXT);
"""


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def init_db(reset=False):
    con = connect()
    if reset:
        for t in ["response", "alert", "health_log", "event_log", "device", "site", "event_class", "user"]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM site").fetchone()[0] == 0:
        con.executemany("INSERT INTO site VALUES (?,?,?,?,?)", SITES)
        con.executemany("INSERT INTO event_class VALUES (?,?,?)",
                        [(i, c, PRIORITY[i]) for i, c in enumerate(CLASSES)])
        today = datetime.now().strftime("%Y-%m-%d")
        n = 0
        for site_id, (lat, lon) in SITE_CENTRES.items():
            for k in range(3):
                n += 1
                con.execute("INSERT INTO device VALUES (?,?,?,?,?,?,?,?)",
                            (f"PIM-{n:02d}", f"70B3D57ED00{n:05X}", site_id,
                             lat + 0.006 * (k - 1), lon + 0.008 * (k - 1), "1.0", today, "Online"))
        con.executemany("INSERT INTO user VALUES (?,?,?,?,?)", [
            (1, "Duty Officer", "Security Personnel", "0800-000-0001", hash_password("security123")),
            (2, "Field Technician", "Field Technician", "0800-000-0002", hash_password("tech123")),
        ])
        con.commit()
    return con


def query(con, sql, params=()):
    import pandas as pd
    return pd.read_sql_query(sql, con, params=params)
