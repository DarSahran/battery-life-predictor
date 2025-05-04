import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
import xgboost as xgb

# ------------------ CONFIG ------------------ #
MODEL_PATH = "../models/battery_xgboost_model.json"
TIME_STEPS = 10

# ------------------ Load model and scalers ------------------ #
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

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
battery_code = battery_type.split("-")[0].strip()

charged_ah = st.number_input(
    "Enter Current Charged Value (in Ah):", 
    min_value=0.0, 
    max_value=battery_capacity, 
    value=battery_capacity, 
    step=0.1
)

if st.button("Start Simulation"):
    st.subheader(f"Live Battery Predictions for {battery_type}")

    base_current = 15
    base_voltage = 12.4
    cumulative_ah = charged_ah  # initialize from user input
    predictions = []
    timestamps = []
    buffer = []
    start_time = datetime.now()

    battery_order = {
        "B1": 8,
        "B2": 9,
        "B3": 10,
        "TN1": 12,
        "B5": 11
    }

    battery_type_map = {"B1": 0, "B2": 1, "B3": 2, "TN1": 3, "B5": 4}

    # KPIs setup (using empty containers for dynamic updates)
    col1, col2, col3, col4 = st.columns(4)
    current_kpi = col1.empty()
    voltage_kpi = col2.empty()
    remaining_kpi = col3.empty()
    elapsed_kpi = col4.empty()

    chart_placeholder = st.empty()
    start_time = datetime.now()

    while True:
        battery_order_code = battery_order[battery_code]
        t = (datetime.now() - start_time).seconds

        current = base_current + 3 * np.sin(t / 10) + np.random.normal(0, 1)
        voltage = base_voltage + 0.05 * np.sin(t / 15) + np.random.normal(0, 0.01)
        current = max(5, min(current, 50))
        voltage = max(9.0, min(voltage, 13.0))

        if voltage < 9.4:
            break

        ah_out = current / 3600
        cumulative_ah += ah_out
        power = current * voltage
        remaining = max(battery_capacity - cumulative_ah, 0)

        base_row = [
            float(current),
            float(voltage),
            float(ah_out),
            float(cumulative_ah),
            float(power),
            float(remaining),
            float(battery_capacity),
            float(battery_capacity),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        base_row[battery_order_code] = 1.0

        buffer.append(base_row)

        if len(buffer) >= TIME_STEPS:
            X_input = np.array(buffer, dtype=np.float32)
            y_pred = model.predict(X_input)
            discharge_percent = np.clip(y_pred[0], 0, 100)
            time_remaining = (discharge_percent / 100) * (battery_capacity * 3600 / base_current)

            # Update KPIs
            current_kpi.metric("⚡ Current (A)", f"{current:.2f}")
            voltage_kpi.metric("🔋 Voltage (V)", f"{voltage:.2f}")
            h = int(time_remaining // 3600)
            m = int((time_remaining % 3600) // 60)
            s = int(time_remaining % 60)
            remaining_kpi.metric("⏳ Remaining Time", f"{h:02d}:{m:02d}:{s:02d}")
            elapsed_kpi.metric("🕒 Running Time", str(datetime.now() - start_time).split('.')[0])

            predictions.append(time_remaining / 3600)
            timestamps.append(datetime.now().strftime('%H:%M:%S'))
            chart_placeholder.line_chart(pd.DataFrame({"Battery Hours Left": predictions}, index=timestamps))

            buffer.pop(0)

        time.sleep(1)

    st.success("✅ Battery voltage dropped below threshold. Simulation completed.")
