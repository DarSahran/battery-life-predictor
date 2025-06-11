import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import Predict

Predictor = Predict.prediction()

# Battery metadata (same as in Predict.py)
BATTERY_METADATA = Predictor.metadata
BATTERY_TYPE_MAPPING = {'tn1': 0, 'b1': 1, 'b2': 2, 'b3': 3, 'b5': 4}

def get_file_metadata(filename):
    key = filename.replace('.csv', '')
    return BATTERY_METADATA.get(key, None)

def get_files_by_battery_type(battery_type, folder_path):
    files = []
    for fname, meta in BATTERY_METADATA.items():
        if meta["type"].lower() == battery_type.lower():
            full_path = os.path.join(folder_path, fname + ".csv")
            if os.path.exists(full_path):
                files.append(fname + ".csv")
    return files

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} min {secs} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hr {minutes} min"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# Streamlit UI
st.set_page_config(page_title="Battery Life Real-Time Simulation & Prediction", page_icon="🔋")
st.title("🔋 Battery Life Real-Time Simulation & Prediction")

# Select battery type
battery_types = sorted(set(meta["type"].upper() for meta in BATTERY_METADATA.values()))
battery_type = st.selectbox("Select battery type:", battery_types)

# List and select CSV file
folder_path = "../data/processed"
files = get_files_by_battery_type(battery_type, folder_path)
if not files:
    st.error("No files found for this battery type.")
    st.stop()
selected_file = st.selectbox("Select test file:", files)

# Load and process data
csv_path = os.path.join(folder_path, selected_file)
meta = get_file_metadata(selected_file.replace('.csv', ''))
if not meta:
    st.error("No metadata found for this file.")
    st.stop()

df = load_csv(csv_path)
type_idx = BATTERY_TYPE_MAPPING[meta["type"]]
ini_charge = meta["charged"]

# Prepare lists for plotting
dod_vals = []
tod_vals = []
capacity_vals = []
step_indices = []
current_vals = []
voltage_vals = []

st.subheader("Real-Time Simulation")

dod_plot = st.empty()
tod_plot = st.empty()
status_placeholder = st.empty()

for idx, row in df.iterrows():
    current = row["Current"]
    voltage = row["Voltage"]

    # Get predictions for this step
    pred_df = Predictor.feature_derivator(voltage, current, meta["type"], ini_charge)

    # Extract predicted TOD and CDC from prediction output
    # pred_df is the output of predict_model2 (final output), assumed to be numpy array or similar
    # Adjust extraction depending on your model output format
    if hasattr(pred_df, 'shape') and pred_df.shape[1] >= 2:
        predicted_TOD = pred_df[0][0]
        predicted_CDC = pred_df[0][1]
    else:
        predicted_TOD = pred_df[0]
        predicted_CDC = 0

    # Calculate cumulative Ah out and DoD
    time_step_h = 1 / 60
    if idx == 0:
        cumulative_ah = abs(current) * time_step_h
    else:
        cumulative_ah += abs(current) * time_step_h
    remaining_capacity = ini_charge - cumulative_ah
    dod = (cumulative_ah / ini_charge) * 100 if ini_charge > 0 else 0

    # Append for plotting
    dod_vals.append(dod)
    tod_vals.append(predicted_TOD / 3600)  # Convert seconds to hours
    capacity_vals.append(remaining_capacity)
    step_indices.append(idx / 60)  # Convert step index to hours
    current_vals.append(current)
    voltage_vals.append(voltage)

    # Plot DoD
    with dod_plot.container():
        plt.figure(figsize=(8, 3))
        plt.plot(step_indices, dod_vals, color='blue')
        plt.xlabel("Time (hours)")
        plt.ylabel("Depth of Discharge (%)")
        plt.ylim(0, 100)
        plt.title("Depth of Discharge (DoD) Over Time")
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()

    # Plot Time Remaining (TOD)
    with tod_plot.container():
        plt.figure(figsize=(8, 3))
        plt.plot(step_indices, tod_vals, color='red')
        plt.xlabel("Time (hours)")
        plt.ylabel("Predicted Time Remaining (hours)")
        plt.title("Predicted Time Remaining Over Time")
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()

    # Show status
    rem_time_str = format_time(predicted_TOD)
    status_placeholder.info(
        f"Step: {idx+1}/{len(df)} | Current: {current:.2f} A | Voltage: {voltage:.2f} V | "
        f"Remaining Capacity: {remaining_capacity:.2f} Ah | Predicted Time Remaining: {rem_time_str}"
    )

    time.sleep(0.05)  # Simulate real-time update

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
