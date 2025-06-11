import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import Predict

Predictor = Predict.prediction()

# --- Model paths (relative) ---
MODEL1_PATH = "../Final_codes/battery_random_forest_model1.joblib"
MODEL2_PATH = "../Final_codes/battery_random_forest_model2.joblib"

# --- Battery metadata (from paste.txt) ---
BATTERY_METADATA = {
    "TEST_1_processed": {"capacity": 85, "charged": 85, "type": "b5"},
    "TEST_2_processed": {"capacity": 81.28, "charged": 81.28, "type": "b1"},
    "TEST_3_processed": {"capacity": 85, "charged": 85, "type": "b5"},
    "TEST_4_processed": {"capacity": 85, "charged": 85, "type": "b2"},
    "TEST_5_processed": {"capacity": 88.81, "charged": 88.81, "type": "b2"},
    "TEST_6_processed": {"capacity": 81.84, "charged": 81.84, "type": "b1"},
    "TEST_7_processed": {"capacity": 81.84, "charged": 36, "type": "b1"},
    "TEST_8_processed": {"capacity": 88.81, "charged": 27, "type": "b2"},
    "TEST_9_processed": {"capacity": 85, "charged": 80, "type": "tn1"},
    "TEST_10_processed": {"capacity": 85, "charged": 54, "type": "tn1"},
    "TEST_11_processed": {"capacity": 85, "charged": 85, "type": "b5"},
    "TEST_12_processed": {"capacity": 85, "charged": 67, "type": "b5"},
    "TEST_13_processed": {"capacity": 85, "charged": 85, "type": "b5"},
    "TEST_14_processed": {"capacity": 88.83, "charged": 52, "type": "b3"},
    "TEST_15_processed": {"capacity": 88.35, "charged": 70, "type": "b3"},
    "TEST_16_processed": {"capacity": 88.35, "charged": 61, "type": "b3"},
    "TEST_17_processed": {"capacity": 88.35, "charged": 88.35, "type": "b3"},
}
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

def predictions(df, meta, type, ini_charge):
    time_step_h = 1 / 60  # 1 min per row
    cumulative_ah = 0.0
    results = []
    for idx, row in df.iterrows():
        current = row["Current"]
        voltage = row["Voltage"]
        Predictor.feature_derivator(voltage, current, type, ini_charge)



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
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# --- Streamlit UI ---
st.set_page_config(page_title="Battery Life Real-Time Simulation & Prediction", page_icon="🔋")
st.title("🔋 Battery Life Real-Time Simulation & Prediction")

# 1. Select battery type
battery_types = sorted(set(meta["type"].upper() for meta in BATTERY_METADATA.values()))
battery_type = st.selectbox("Select battery type:", battery_types)

# 2. List and select CSV file (relative path)
folder_path = "../data/processed"
files = get_files_by_battery_type(battery_type, folder_path)
if not files:
    st.error("No files found for this battery type.")
    st.stop()
selected_file = st.selectbox("Select test file:", files)

# 3. Load and process data
csv_path = os.path.join(folder_path, selected_file)
meta = get_file_metadata(selected_file.replace('.csv', ''))
if not meta:
    st.error("No metadata found for this file.")
    st.stop()

# Debug: Show absolute path for troubleshooting
st.caption(f"Loading file: {os.path.abspath(csv_path)}")
if not os.path.exists(csv_path):
    st.error(f"File does not exist: {csv_path}")
    st.stop()

df = load_csv(csv_path)
predictions(df, meta, BATTERY_TYPE_MAPPING[meta["type"]], meta["charged"])



# 7. Real-time simulation and plotting
st.subheader("Real-Time Simulation")

dod_plot = st.empty()
tod_plot = st.empty()
status_placeholder = st.empty()

dod_vals = []
tod_vals = []
capacity_vals = []
step_indices = []
current_vals = []
voltage_vals = []

for i in range(len(features_df)):
    dod_vals.append(features_df.loc[i, "DoD"])
    tod_vals.append(features_df.loc[i, "Final Predicted TOD"] / 3600)  # Convert to hours
    capacity_vals.append(features_df.loc[i, "Remaining Capacity"])
    step_indices.append(features_df.loc[i, "Hour"])
    current_vals.append(features_df.loc[i, "Current"])
    voltage_vals.append(features_df.loc[i, "Voltage"])

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

    # Show current battery status
    rem_time_sec = features_df.loc[i, "Final Predicted TOD"]
    rem_time_str = format_time(rem_time_sec)
    status_placeholder.info(
        f"Step: {i+1}/{len(features_df)} | "
        f"Current: {current_vals[-1]:.2f} A | Voltage: {voltage_vals[-1]:.2f} V | "
        f"Remaining Capacity: {capacity_vals[-1]:.2f} Ah | "
        f"Predicted Time Remaining: {rem_time_str}"
    )

    time.sleep(0.05)  # Simulate real-time update

st.success("Simulation complete!")

st.subheader("Final Battery Status")
st.write(f"Final Remaining Capacity: {capacity_vals[-1]:.2f} Ah")
st.write(f"Final Predicted Time Remaining: {format_time(features_df['Final Predicted TOD'].iloc[-1])}")
st.write(f"Final DoD: {dod_vals[-1]:.2f} %")

st.subheader("Download Results")
st.dataframe(features_df.tail(10))
st.download_button(
    label="Download All Predictions as CSV",
    data=features_df.to_csv(index=False),
    file_name=f"{selected_file.replace('.csv','')}_predictions.csv",
    mime="text/csv"
)
