import os
import random
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

SCHEMA_SQL = """
DROP TABLE IF EXISTS raw_gate_events, payment_transactions, local_events, weather_logs, lot_config CASCADE;

CREATE TABLE lot_config (
    lot_id VARCHAR(20) PRIMARY KEY,
    capacity INT NOT NULL,
    entry_gate_offset_sec INT DEFAULT 0,
    exit_gate_offset_sec INT DEFAULT 0
);

CREATE TABLE raw_gate_events (
    event_id SERIAL PRIMARY KEY,
    vehicle_plate VARCHAR(15) NOT NULL,
    event_type VARCHAR(10) NOT NULL,
    gate_timestamp TIMESTAMP NOT NULL,
    gate_id VARCHAR(10) NOT NULL
);

CREATE TABLE payment_transactions (
    tx_id SERIAL PRIMARY KEY,
    vehicle_plate VARCHAR(15) NOT NULL,
    paid_at TIMESTAMP NOT NULL,
    amount DECIMAL(6, 2) NOT NULL
);

CREATE TABLE local_events (
    event_date DATE PRIMARY KEY,
    event_name VARCHAR(100) NOT NULL,
    expected_attendance INT NOT NULL
);

CREATE TABLE weather_logs (
    log_hour TIMESTAMP PRIMARY KEY,
    condition VARCHAR(20) NOT NULL,
    temp_c DECIMAL(4, 1) NOT NULL
);
"""

def seed():
    print("Connecting to Neon DB...")
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
    cur = conn.cursor()
    print("Creating tables...")
    cur.execute(SCHEMA_SQL)

    cur.execute(
        "INSERT INTO lot_config VALUES (%s, %s, %s, %s)",
        ("LOT_MAIN", 80, 0, -240)
    )

    start_date = datetime(2026, 11, 1, 0, 0, 0)
    event_days = {
        3: ("Tech Expo 2026", 1500),
        7: ("City Championship Match", 4000),
        10: ("Symphony Night", 900),
        13: ("Food & Music Fest", 2500),
    }

    for day_offset, (name, att) in event_days.items():
        d = (start_date + timedelta(days=day_offset)).date()
        cur.execute("INSERT INTO local_events VALUES (%s, %s, %s)", (d, name, att))

    plates_pool = [f"MH12AB{1000 + i}" for i in range(600)]
    gate_events = []
    payments = []
    weather = []

    for day in range(14):
        curr_day = start_date + timedelta(days=day)
        is_event = day in event_days
        daily_volume = random.randint(155, 195) if is_event else random.randint(85, 115)

        for h in range(24):
            ts = curr_day + timedelta(hours=h)
            cond = random.choices(["Clear", "Cloudy", "Rain"], weights=[0.65, 0.25, 0.10])[0]
            temp = round(random.uniform(21.0, 31.0), 1)
            weather.append((ts, cond, temp))

        for _ in range(daily_volume):
            plate = random.choice(plates_pool)
            if is_event and random.random() < 0.45:
                entry_hour = random.choices([17, 18, 19], weights=[0.4, 0.4, 0.2])[0]
            else:
                entry_hour = random.choices([8, 9, 10, 11, 13, 14, 16], weights=[0.1, 0.25, 0.25, 0.15, 0.1, 0.1, 0.05])[0]

            true_entry = curr_day + timedelta(hours=entry_hour, minutes=random.randint(0, 59))
            duration_mins = random.randint(45, 240)
            true_exit = true_entry + timedelta(minutes=duration_mins)

            anomaly = random.random()
            if anomaly < 0.06:
                skewed_exit = true_exit + timedelta(seconds=-240)
                gate_events.append((plate, "EXIT", skewed_exit, "GATE_OUT"))
            elif anomaly < 0.11:
                gate_events.append((plate, "ENTRY", true_entry, "GATE_IN"))
                payments.append((plate, true_exit - timedelta(minutes=3), round(duration_mins * 0.5, 2)))
            else:
                if random.random() < 0.03:
                    true_exit = true_entry + timedelta(minutes=2)
                skewed_exit = true_exit + timedelta(seconds=-240)
                gate_events.append((plate, "ENTRY", true_entry, "GATE_IN"))
                gate_events.append((plate, "EXIT", skewed_exit, "GATE_OUT"))
                payments.append((plate, true_exit - timedelta(minutes=2), round(max(10.0, duration_mins * 0.5), 2)))

    print("Bulk inserting records...")
    execute_values(
        cur,
        "INSERT INTO raw_gate_events (vehicle_plate, event_type, gate_timestamp, gate_id) VALUES %s",
        gate_events
    )
    execute_values(
        cur,
        "INSERT INTO payment_transactions (vehicle_plate, paid_at, amount) VALUES %s",
        payments
    )
    execute_values(
        cur,
        "INSERT INTO weather_logs (log_hour, condition, temp_c) VALUES %s",
        weather
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done! Seeded {len(gate_events)} gate events and {len(payments)} payments into Neon DB.")

if __name__ == "__main__":
    seed()