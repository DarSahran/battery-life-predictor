import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------ #
MODEL_PATH = "../models/battery_lstm_model.keras"
SCALER_X_PATH = "../models/input_scaler.pkl"
SCALER_Y_PATH = "../models/target_scaler.pkl"
FEATURES = ['Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah', 'Power', 'Remaining Capacity']
EXTRA_FEATURES = ['Capacity', 'Charged_Upto'] + [f"Battery_Type_{t}" for t in ['b1', 'b2', 'b3', 'tn1', 'b5']]
ALL_FEATURES = FEATURES + EXTRA_FEATURES
TIME_STEPS = 30
RUNTIME = 180

# ------------------ Load model and scalers ------------------ #
model = load_model(MODEL_PATH)
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# ------------------ Streamlit UI ------------------ #
st.set_page_config(layout="wide")
st.title("🔋 Real-Time Battery Life Monitor")

battery_type = st.selectbox("Select Battery Type:", [
    "B1 - 81.28 Ah",
    "B2 - 85 Ah",
    "B3 - 88.35 Ah",
    "TN1 - 85 Ah",
    "B5 - 85 Ah"
])

battery_capacity = float(battery_type.split("-")[1].strip().split()[0])
battery_code = battery_type.split("-")[0].strip().lower()

if st.button("Start Simulation"):
    st.subheader(f"Live Battery Predictions for {battery_type}")

    base_current = 15
    base_voltage = 12.4
    cumulative_ah = 0
    last_prediction = float('inf')
    predictions = []
    timestamps = []
    buffer = []

    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    battery_type_map = {"b1": 0, "b2": 1, "b3": 2, "tn1": 3, "b5": 4}

    for t in range(RUNTIME):
        current = base_current + 3 * np.sin(t / 10) + np.random.normal(0, 1)
        voltage = base_voltage + 0.05 * np.sin(t / 15) + np.random.normal(0, 0.01)
        current = max(5, min(current, 50))
        voltage = max(9.3, min(voltage, 13.0))

        ah_out = current / 3600
        cumulative_ah += ah_out
        power = current * voltage
        remaining = max(battery_capacity - cumulative_ah, 0)

        base_row = {
            'Current': current,
            'Voltage': voltage,
            'Ah Out': ah_out,
            'Cumulative Actual Disch Ah': cumulative_ah,
            'Power': power,
            'Remaining Capacity': remaining,
            'Capacity': battery_capacity,
            'Charged_Upto': battery_capacity,
        }

        for bt in battery_type_map:
            base_row[f"Battery_Type_{bt}"] = 1 if battery_code == bt else 0

        buffer.append(base_row)

        if len(buffer) >= TIME_STEPS:
            df = pd.DataFrame(buffer)
            df_smoothed = df.rolling(window=3, min_periods=1).mean()
            df_ordered = df_smoothed[scaler_x.feature_names_in_]  # Ensure correct column order
            X_scaled = scaler_x.transform(df_ordered)
            X_input = np.expand_dims(X_scaled[-TIME_STEPS:], axis=0)

            y_pred_scaled = model.predict(X_input, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            discharge_percent = y_pred[0][0]
            discharge_percent = np.clip(discharge_percent, 0, 100)
            time_remaining = (discharge_percent / 100) * (battery_capacity * 3600 / base_current)

            h = int(time_remaining // 3600)
            m = int((time_remaining % 3600) // 60)
            s = int(time_remaining % 60)

            predictions.append(time_remaining / 3600)
            timestamps.append(datetime.now().strftime('%H:%M:%S'))

            chart_placeholder.line_chart(pd.DataFrame({"Battery Hours Left": predictions}, index=timestamps))
            table_placeholder.markdown(
                f"**[{timestamps[-1]}]** ⚡ Current: `{current:.2f}` A | 🔋 Voltage: `{voltage:.2f}` V | ⏳ Time Left: `{h:02d}:{m:02d}:{s:02d}` (HH:MM:SS)"
            )

            buffer.pop(0)

        time.sleep(1)

    st.success("✅ Simulation completed!")