import numpy as np
import pandas as pd
import time
import threading
from queue import Queue
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Load model & scalers
model = load_model("../models/battery_lstm_model.keras")
scaler_x = joblib.load("../models/input_scaler.pkl")
scaler_y = joblib.load("../models/target_scaler.pkl")

FEATURES = ['Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah', 'Power', 'Remaining Capacity']
BATTERY_CAPACITY = 85
RUNTIME = 60
PREDICT_EVERY = 1

TIME_STEPS = 1
cumulative_ah = 0
last_prediction = float('inf')  # Monotonic constraint tracker

data_queue = Queue()

# ---------------- Simulate realistic battery usage ---------------- #
def simulate_realistic_battery_data():
    print("[\U0001f4e1] Starting realistic battery simulation...")
    base_current = 15
    base_voltage = 12.4

    for t in range(RUNTIME):
        current = base_current + 3 * np.sin(t / 10) + np.random.normal(0, 1)
        voltage = base_voltage + 0.05 * np.sin(t / 15) + np.random.normal(0, 0.01)
        current = max(5, min(current, 50))
        voltage = max(11.5, min(voltage, 12.6))

        ah_out = current / 3600
        global cumulative_ah
        cumulative_ah += ah_out
        power = current * voltage
        remaining = max(BATTERY_CAPACITY - cumulative_ah, 0)

        row = {
            'Current': round(current, 2),
            'Voltage': round(voltage, 2),
            'Ah Out': round(ah_out, 5),
            'Cumulative Actual Disch Ah': round(cumulative_ah, 5),
            'Power': round(power, 2),
            'Remaining Capacity': round(remaining, 2)
        }

        data_queue.put(row)
        time.sleep(1)

    print("[✅] Simulation finished.")

# ---------------- Predict every second ---------------- #
def predict_battery_life_every_second():
    print("[🔁] LSTM prediction every second...\n")
    buffer = []
    global last_prediction

    for _ in range(RUNTIME):
        if not data_queue.empty():
            row = data_queue.get()
            buffer.append(row)

        if len(buffer) >= TIME_STEPS:
            df = pd.DataFrame(buffer[-TIME_STEPS:])[FEATURES]
            df = df.rolling(window=3, min_periods=1).mean()
            X_scaled = scaler_x.transform(df)
            X_input = np.expand_dims(X_scaled, axis=0)

            y_pred_scaled = model.predict(X_input, verbose=0)
            y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
            time_remaining = np.expm1(y_pred_log[0][0])

            # ✅ Enforce non-increasing prediction
            time_remaining = min(time_remaining, last_prediction)
            last_prediction = time_remaining

            h = int(time_remaining // 3600)
            m = int((time_remaining % 3600) // 60)
            s = int(time_remaining % 60)
            now = datetime.now().strftime('%H:%M:%S')
            current = row['Current']
            voltage = row['Voltage']
            print(f"[{now}] ⚡ Current: {current:.2f}A | Voltage: {voltage:.2f}V | 🔋 Estimated Remaining Time: {h:02d}:{m:02d}:{s:02d} (HH:MM:SS)")

        time.sleep(PREDICT_EVERY)

# ---------------- Run Threads ---------------- #
if __name__ == "__main__":
    publisher_thread = threading.Thread(target=simulate_realistic_battery_data)
    subscriber_thread = threading.Thread(target=predict_battery_life_every_second)

    publisher_thread.start()
    subscriber_thread.start()

    publisher_thread.join()
    subscriber_thread.join()

    print("[🛑] Session completed.")
