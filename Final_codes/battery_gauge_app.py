import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import Predict

Predictor = Predict.prediction()
BATTERY_TYPE_MAPPING = {'tn1': 0, 'b1': 1, 'b2': 2, 'b3': 3, 'b5': 4}

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def format_time(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        return f"{seconds//60} min {seconds%60} sec"
    else:
        return f"{seconds//3600} hr {((seconds%3600)//60)} min"

def apply_smoothing(x, window=10, method="Moving Average"):
    if method == "Moving Average":
        return pd.Series(x).rolling(window=window, min_periods=1).mean().values
    elif method == "Exponential":
        return pd.Series(x).ewm(span=window, adjust=False).mean().values
    return x

st.set_page_config(page_title="Battery Life Real-Time Simulation & Prediction", page_icon="🔋")
st.title("🔋 Battery Life Real-Time Simulation & Prediction")

folder_path = "../data/processed"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
selected_file = st.selectbox("Select test file:", files)
csv_path = os.path.join(folder_path, selected_file)
df = load_csv(csv_path)

# Get battery type from metadata or filename
meta = Predictor.metadata[selected_file.replace('.csv','')]
type_code = BATTERY_TYPE_MAPPING[meta['type']]

# --- UI Controls ---
col1, col2, col3 = st.columns([1,1,2])
with col1:
    start = st.button("Start Simulation")
with col2:
    stop = st.button("Stop Simulation")
with col3:
    speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=1)

speed_map = {"Slow": 0.2, "Medium": 0.05, "Fast": 0.005}
sleep_time = speed_map[speed]

# --- Session state for simulation ---
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# --- Prepare lists for plotting ---
dod_vals, tod_vals, capacity_vals, step_indices, current_vals, voltage_vals = [], [], [], [], [], []

dod_plot = st.empty()
tod_plot = st.empty()
status_placeholder = st.empty()

# --- Simulation Loop ---
if st.session_state.running:
    for idx, row in df.iterrows():
        # Use the CSV's own columns for features
        features = pd.DataFrame({
            'Current': [row['Current']],
            'Voltage': [row['Voltage']],
            'Ah Out': [row['Ah Out']],
            'Power': [row['Power']],
            'Remaining Capacity': [row['Remaining Capacity']],
            'type': [type_code],
            'capacity': [meta['capacity']],
            'charged': [meta['charged']],
            'discharge_rate': [row['Current'] / (row['Voltage'] + 1e-6)],
            'discharge_ratio': [row['Ah Out'] / (meta['charged'] + 1e-6)]
        })

        pred = Predictor.model1.predict(features)
        if hasattr(pred, 'shape') and pred.shape[1] >= 2:
            predicted_TOD = pred[0][0]
        else:
            predicted_TOD = pred[0]

        dod = 100 * (meta['charged'] - row['Remaining Capacity']) / meta['charged']
        dod_vals.append(dod)
        tod_vals.append(predicted_TOD / 3600)
        capacity_vals.append(row['Remaining Capacity'])
        step_indices.append(idx / 60)
        current_vals.append(row['Current'])
        voltage_vals.append(row['Voltage'])

        # Apply smoothing to DoD and TOD (always on, moving average, window=10)
        smooth_dod_vals = apply_smoothing(dod_vals, window=10, method="Moving Average")
        smooth_tod_vals = apply_smoothing(tod_vals, window=10, method="Moving Average")

        # Plot DoD
        with dod_plot.container():
            plt.figure(figsize=(8, 3))
            plt.plot(step_indices, dod_vals, color='blue', alpha=0.3, label="Raw")
            plt.plot(step_indices, smooth_dod_vals, color='blue', linewidth=2, label="Smoothed")
            plt.xlabel("Time (hours)")
            plt.ylabel("Depth of Discharge (%)")
            plt.ylim(0, 100)
            plt.title("Depth of Discharge (DoD) Over Time")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt.gcf())
            plt.close()

        # Plot Time Remaining (TOD)
        with tod_plot.container():
            plt.figure(figsize=(8, 3))
            plt.plot(step_indices, tod_vals, color='red', alpha=0.3, label="Raw")
            plt.plot(step_indices, smooth_tod_vals, color='red', linewidth=2, label="Smoothed")
            plt.xlabel("Time (hours)")
            plt.ylabel("Predicted Time Remaining (hours)")
            plt.title("Predicted Time Remaining Over Time")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt.gcf())
            plt.close()

        rem_time_sec = predicted_TOD
        rem_time_str = format_time(rem_time_sec)
        status_placeholder.info(
            f"Step: {idx+1}/{len(df)} | Current: {row['Current']:.2f} A | Voltage: {row['Voltage']:.2f} V | "
            f"Remaining Capacity: {row['Remaining Capacity']:.2f} Ah | Predicted Time Remaining: {rem_time_str}"
        )

        time.sleep(sleep_time)
        if not st.session_state.running:
            break

    st.success("Simulation complete!")

    st.subheader("Final Battery Status")
    st.write(f"Final Remaining Capacity: {capacity_vals[-1]:.2f} Ah")
    st.write(f"Final Predicted Time Remaining: {format_time(tod_vals[-1]*3600)}")
    st.write(f"Final DoD: {dod_vals[-1]:.2f} %")

    st.subheader("Download Results")
    results_df = pd.DataFrame({
        "Step (hr)": step_indices,
        "Current (A)": current_vals,
        "Voltage (V)": voltage_vals,
        "Remaining Capacity (Ah)": capacity_vals,
        "Depth of Discharge (%)": dod_vals,
        "Predicted Time Remaining (s)": [tod*3600 for tod in tod_vals],
    })
    st.dataframe(results_df.tail(10))
    st.download_button(
        label="Download All Predictions as CSV",
        data=results_df.to_csv(index=False),
        file_name=f"{selected_file.replace('.csv','')}_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Click 'Start Simulation' to begin.")
