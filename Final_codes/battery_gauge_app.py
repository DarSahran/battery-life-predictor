import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import joblib
import os


# Set page configuration
st.set_page_config(layout="wide", page_title="Battery Life Predictor", page_icon="🔋")

# --- Constants and Model Loading ---

# Paths to pre-trained models
MODEL1_PATH = 'battery_random_forest_model1.joblib'
MODEL2_PATH = 'battery_random_forest_model2.joblib'

# Simulation parameters from BatteryDepletionPredictionModel
SIM_INTERVAL_MINUTES = 5
DEGRADATION_RATE_PER_INTERVAL = 0.0001
MAX_OPERATIONAL_VOLTAGE = 12.8
MIN_OPERATIONAL_VOLTAGE = 9.0
MAX_SIMULATED_CURRENT = 15.0

# Battery specifications
BATTERY_SPECS = {
    "TN1": {"nominal_capacity": 85.0, "charged_capacity": 80.0, "nominal_voltage": 12.6},
    "B1": {"nominal_capacity": 81.28, "charged_capacity": 81.28, "nominal_voltage": 12.6},
    "B2": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6},
    "B3": {"nominal_capacity": 88.35, "charged_capacity": 88.35, "nominal_voltage": 12.6},
    "B5": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6}
}
# Map battery codes to an index for the model
BATTERY_TYPE_MAPPING = {key: i for i, key in enumerate(BATTERY_SPECS.keys())}


@st.cache_resource
def load_models():
    """Load the pre-trained models."""
    if not os.path.exists(MODEL1_PATH) or not os.path.exists(MODEL2_PATH):
        st.error(f"Model files not found. Make sure '{MODEL1_PATH}' and '{MODEL2_PATH}' are in the same directory.")
        return None, None
    try:
        model1_obj = joblib.load(MODEL1_PATH)
        model2_obj = joblib.load(MODEL2_PATH)

        # Handle case where joblib.load returns a tuple (e.g., model and scaler)
        model1 = model1_obj[0] if isinstance(model1_obj, tuple) else model1_obj
        model2 = model2_obj[0] if isinstance(model2_obj, tuple) else model2_obj
        
        return model1, model2
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model1, model2 = load_models()


class BatterySimulator:
    def __init__(self, battery_type, initial_charge):
        self.battery_type = battery_type
        self.specs = BATTERY_SPECS[battery_type]
        self.nominal_capacity = self.specs["nominal_capacity"]
        
        self.current_capacity_ah = initial_charge
        self.time_elapsed_hours = 0
        self.step = 0
        
        # Initialize state variables based on the logic from the provided file
        self.battery_type_index = BATTERY_TYPE_MAPPING[battery_type]
        self.time_step_secs = SIM_INTERVAL_MINUTES * 60
        self.cumulative_ah_total = 0
        self.data_buffer = []

    def simulate_discharge_step(self):
        """Simulate one time step of battery discharge using the loaded models."""
        if self.current_capacity_ah <= 0 or model1 is None or model2 is None:
            return None

        # --- This logic is adapted from your provided simulation file ---
        
        # 1. Calculate current telemetry
        soc = self.current_capacity_ah / self.nominal_capacity if self.nominal_capacity > 0 else 0
        degradation_factor = 1 - (self.step * DEGRADATION_RATE_PER_INTERVAL)
        
        base_current = 8 + 7 * soc + np.random.normal(0, 0.5)
        current = max(0.5, min(MAX_SIMULATED_CURRENT, base_current))
        
        voltage = MIN_OPERATIONAL_VOLTAGE + (MAX_OPERATIONAL_VOLTAGE - MIN_OPERATIONAL_VOLTAGE) * soc * degradation_factor
        voltage += np.random.normal(0, 0.05)
        voltage = max(MIN_OPERATIONAL_VOLTAGE, min(MAX_OPERATIONAL_VOLTAGE, voltage))

        ah_out = (current * self.time_step_secs) / 3600
        self.cumulative_ah_total += ah_out
        power = voltage * current
        
        self.current_capacity_ah = max(0, self.current_capacity_ah - ah_out)
        percent_remaining = (self.current_capacity_ah / self.nominal_capacity) * 100 if self.nominal_capacity > 0 else 0
        
        # 2. Prepare data for Model 1 - with corrected column names and features
        discharge_rate = current / self.nominal_capacity if self.nominal_capacity > 0 else 0
        discharge_ratio = self.cumulative_ah_total / self.nominal_capacity if self.nominal_capacity > 0 else 0

        base_row = {
            'step_index': self.step,
            'type': self.battery_type_index,
            'Current': current,
            'Voltage': voltage,
            'Ah Out': ah_out,
            'Cumulative Actual Disch Ah': self.cumulative_ah_total,
            'Power': power,
            'Remaining Capacity': percent_remaining,
            'charged': self.current_capacity_ah,
            'capacity': self.nominal_capacity,
            'discharge_rate': discharge_rate,
            'discharge_ratio': discharge_ratio
        }
        
        X_input_df = pd.DataFrame([base_row])
        for i in range(1, 5):
            if len(self.data_buffer) >= i:
                past_data = self.data_buffer[-i]
                X_input_df[f'Current_t-{i}'] = past_data['Current']
                X_input_df[f'Voltage_t-{i}'] = past_data['Voltage']
            else:
                X_input_df[f'Current_t-{i}'] = current
                X_input_df[f'Voltage_t-{i}'] = voltage
        
        # 3. Predict with Model 1
        pred_model1 = model1.predict(X_input_df)
        predicted_voltage_next_step = pred_model1[0]
        
        # 4. Prepare data for Model 2
        X_input_df_for_model2 = X_input_df.copy()
        X_input_df_for_model2['Predicted_Voltage_next_step'] = predicted_voltage_next_step

        # 5. Predict with Model 2
        y_pred_model2 = model2.predict(X_input_df_for_model2)
        time_remaining_seconds = y_pred_model2[0]
        time_remaining_hours = max(0, time_remaining_seconds / 3600)
        
        # Update state for next step
        self.time_elapsed_hours += (self.time_step_secs / 3600)
        self.step += 1
        self.data_buffer.append(base_row)
        if len(self.data_buffer) > 5:
            self.data_buffer.pop(0)

        # 6. Return results
        return {
            'time_elapsed': self.time_elapsed_hours, 'current': current, 'voltage': voltage,
            'power': power, 'ah_remaining': self.current_capacity_ah,
            'percent_remaining': percent_remaining, 'soc': soc,
            'time_remaining': time_remaining_hours
        }

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'simulator' not in st.session_state:
    st.session_state.simulator = None

# --- UI Layout ---
st.title("🔋 EV Battery Life Predictor")
st.markdown("**Real-time battery discharge simulation with ML-powered time remaining predictions**")

st.sidebar.header("Battery Configuration")
selected_battery = st.sidebar.selectbox("Select Battery Type:", list(BATTERY_SPECS.keys()), key="sb_battery")
battery_info = BATTERY_SPECS[selected_battery]
initial_charge = st.sidebar.slider("Initial Charge (Ah):", min_value=5.0, max_value=battery_info["charged_capacity"], value=battery_info["charged_capacity"], step=0.5, key="sl_charge")
st.sidebar.markdown("---")
simulation_speed = st.sidebar.slider("Simulation Speed (updates/sec):", 1, 10, 5, key="sl_speed")

models_loaded = model1 is not None and model2 is not None

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Battery Telemetry")
    plot_placeholder = st.empty()

with col2:
    st.header("Battery Status")
    b_col1, b_col2, b_col3 = st.columns(3)
    if b_col1.button("▶️ Start", use_container_width=True, disabled=not models_loaded):
        st.session_state.is_running = True
        st.session_state.simulation_data = []
        st.session_state.simulator = BatterySimulator(selected_battery, initial_charge)
        st.rerun()
    if b_col2.button("⏸️ Stop", use_container_width=True):
        st.session_state.is_running = False
    if b_col3.button("🔄 Reset", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.simulator = None
        st.session_state.simulation_data = []
        st.rerun()

    st.markdown("---")
    capacity_placeholder = st.empty()
    current_placeholder = st.empty()
    voltage_placeholder = st.empty()
    time_rem_placeholder = st.empty()
    time_ela_placeholder = st.empty()

# --- Main Simulation Loop ---
if st.session_state.is_running and models_loaded:
    simulator = st.session_state.simulator
    while st.session_state.is_running:
        if simulator:
            data_point = simulator.simulate_discharge_step()
            if data_point:
                st.session_state.simulation_data.append(data_point)
                
                capacity_placeholder.metric("🔋 Remaining Capacity", f"{data_point['percent_remaining']:.1f}%")
                current_placeholder.metric("⚡ Current", f"{data_point['current']:.1f} A")
                voltage_placeholder.metric("🔌 Voltage", f"{data_point['voltage']:.1f} V")
                time_rem_placeholder.metric("⏱️ Time Remaining", f"{data_point['time_remaining']:.1f} hours")
                time_ela_placeholder.metric("🕐 Time Elapsed", f"{data_point['time_elapsed']:.1f} hours")
                
                df_plot = pd.DataFrame(st.session_state.simulation_data)
                chart_data = df_plot.rename(columns={
                    'time_elapsed': 'Time Elapsed (h)',
                    'time_remaining': 'Predicted Time Remaining (h)',
                    'percent_remaining': 'Battery Capacity (%)'
                }).set_index('Time Elapsed (h)')
                
                plot_placeholder.line_chart(chart_data[['Predicted Time Remaining (h)', 'Battery Capacity (%)']])
                
                time.sleep(1 / simulation_speed)
            else:
                st.session_state.is_running = False
                st.toast("🎉 Battery simulation completed!")
        else:
            st.session_state.is_running = False
    st.rerun()

# --- Display Final State ---
else:
    if not models_loaded:
        st.warning("Prediction models are not loaded. Cannot start simulation.")

    if st.session_state.simulation_data:
        last_data = st.session_state.simulation_data[-1]
        capacity_placeholder.metric("🔋 Remaining Capacity", f"{last_data['percent_remaining']:.1f}%")
        current_placeholder.metric("⚡ Current", f"{last_data['current']:.1f} A")
        voltage_placeholder.metric("🔌 Voltage", f"{last_data['voltage']:.1f} V")
        time_rem_placeholder.metric("⏱️ Time Remaining", f"{last_data.get('time_remaining', 0):.1f} hours")
        time_ela_placeholder.metric("🕐 Time Elapsed", f"{last_data['time_elapsed']:.1f} hours")
        
        df_plot = pd.DataFrame(st.session_state.simulation_data)
        chart_data = df_plot.rename(columns={
            'time_elapsed': 'Time Elapsed (h)',
            'time_remaining': 'Predicted Time Remaining (h)',
            'percent_remaining': 'Battery Capacity (%)'
        }).set_index('Time Elapsed (h)')
        plot_placeholder.line_chart(chart_data[['Predicted Time Remaining (h)', 'Battery Capacity (%)']])
    else:
        plot_placeholder.info("Configure battery and start the simulation to see live telemetry.")
        capacity_placeholder.metric("🔋 Remaining Capacity", "N/A")
        current_placeholder.metric("⚡ Current", "N/A")
        voltage_placeholder.metric("🔌 Voltage", "N/A")
        time_rem_placeholder.metric("⏱️ Time Remaining", "N/A")
        time_ela_placeholder.metric("🕐 Time Elapsed", "N/A")

if st.session_state.simulation_data:
    st.header("Simulation Data")
    df_display = pd.DataFrame(st.session_state.simulation_data)[['time_elapsed', 'percent_remaining', 'current', 'voltage', 'time_remaining']].round(2)
    df_display.columns = ['Time Elapsed (h)', 'Battery (%)', 'Current (A)', 'Voltage (V)', 'Time Remaining (h)']
    st.dataframe(df_display.tail(10))

st.markdown("---")
st.header("ℹ️ About This Application")
st.markdown("**Features:**\n- **Realistic Battery Simulation**: Models actual Li-ion battery discharge characteristics.\n- **Machine Learning Predictions**: Uses two pre-trained Random Forest models to predict voltage and remaining battery life.\n- **EV-Style Visualization**: Professional battery gauge similar to electric vehicle displays.\n- **Real-time Updates**: Live simulation with configurable update rates.\n- **Multiple Battery Types**: Support for different battery specifications.")