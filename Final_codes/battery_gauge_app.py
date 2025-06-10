import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 
# streamlit run battery_gauge_app.py
# 
# Set page configuration
st.set_page_config(layout="wide", page_title="Battery Life Predictor", page_icon="🔋")

# Battery specifications
BATTERY_SPECS = {
    "TN1": {"nominal_capacity": 85.0, "charged_capacity": 80.0, "nominal_voltage": 12.6},
    "B1": {"nominal_capacity": 81.28, "charged_capacity": 81.28, "nominal_voltage": 12.6},
    "B2": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6},
    "B3": {"nominal_capacity": 88.35, "charged_capacity": 88.35, "nominal_voltage": 12.6},
    "B5": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6}
}

class BatterySimulator:
    def __init__(self, battery_type, initial_charge):
        self.battery_type = battery_type
        self.specs = BATTERY_SPECS[battery_type]
        self.initial_charge = initial_charge
        self.current_capacity = initial_charge
        self.nominal_capacity = self.specs["nominal_capacity"]
        self.nominal_voltage = self.specs["nominal_voltage"]
        self.time_elapsed = 0  # in hours
        self.discharge_data = []
        
    def simulate_discharge_step(self, time_step_hours=0.1):
        """Simulate one time step of battery discharge"""
        if self.current_capacity <= 0:
            return None
            
        soc = self.current_capacity / self.nominal_capacity
        base_current = 8 + 7 * soc + np.random.normal(0, 0.5)
        current = max(0.5, min(15.0, base_current))
        
        voltage = 9.5 + 3.1 * soc + np.random.normal(0, 0.05)
        voltage = max(9.0, min(12.8, voltage))
        
        ah_discharged = current * time_step_hours
        self.current_capacity = max(0, self.current_capacity - ah_discharged)
        remaining_percent = (self.current_capacity / self.nominal_capacity) * 100
        power = voltage * current
        self.time_elapsed += time_step_hours
        
        data_point = {
            'time_elapsed': self.time_elapsed, 'current': current, 'voltage': voltage,
            'power': power, 'ah_remaining': self.current_capacity,
            'percent_remaining': remaining_percent, 'soc': soc
        }
        self.discharge_data.append(data_point)
        return data_point

class BatteryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self):
        training_data = []
        for battery_type in BATTERY_SPECS.keys():
            for initial_charge in np.linspace(10, BATTERY_SPECS[battery_type]["charged_capacity"], 20):
                sim = BatterySimulator(battery_type, initial_charge)
                while sim.current_capacity > 0.1:
                    data_point = sim.simulate_discharge_step(0.05)
                    if data_point is None: break
                    estimated_time_remaining = sim.current_capacity / (data_point['current'] + 0.1)
                    features = [
                        data_point['current'], data_point['voltage'], data_point['power'],
                        data_point['percent_remaining'], data_point['soc'],
                        sim.nominal_capacity, data_point['time_elapsed']
                    ]
                    training_data.append(features + [estimated_time_remaining])
        return np.array(training_data)
    
    @st.cache_resource
    def train_model(_self):
        training_data = _self.generate_training_data()
        X = training_data[:, :-1]
        y = training_data[:, -1]
        X_scaled = _self.scaler.fit_transform(X)
        _self.model.fit(X_scaled, y)
        _self.is_trained = True
            
    def predict_time_remaining(self, current, voltage, power, percent_remaining, soc, nominal_capacity, time_elapsed):
        if not self.is_trained: return 0
        features = np.array([[current, voltage, power, percent_remaining, soc, nominal_capacity, time_elapsed]])
        features_scaled = self.scaler.transform(features)
        return max(0, self.model.predict(features_scaled)[0])

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = BatteryPredictor()
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

if not st.session_state.predictor.is_trained:
    if st.sidebar.button("🚀 Initialize Prediction Model"):
        with st.spinner("Training prediction model... This may take a moment."):
            st.session_state.predictor.train_model()
        st.sidebar.success("Model trained successfully!")
        st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Battery Telemetry")
    plot_placeholder = st.empty()

with col2:
    st.header("Battery Status")
    b_col1, b_col2, b_col3 = st.columns(3)
    if b_col1.button("▶️ Start", use_container_width=True, disabled=not st.session_state.predictor.is_trained):
        st.session_state.is_running = True
        st.session_state.simulation_data = []
        st.session_state.simulator = BatterySimulator(selected_battery, initial_charge)
        st.rerun() # Rerun once to start the loop
    if b_col2.button("⏸️ Stop", use_container_width=True):
        st.session_state.is_running = False
        # No rerun needed here, the loop will terminate
    if b_col3.button("🔄 Reset", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.simulator = None
        st.session_state.simulation_data = []
        st.rerun() # Rerun once to clear state

    st.markdown("---")
    capacity_placeholder = st.empty()
    current_placeholder = st.empty()
    voltage_placeholder = st.empty()
    time_rem_placeholder = st.empty()
    time_ela_placeholder = st.empty()

# --- Main Simulation Loop (No more st.rerun() inside!) ---
if st.session_state.is_running:
    simulator = st.session_state.simulator
    while st.session_state.is_running:
        if simulator and simulator.current_capacity > 0:
            data_point = simulator.simulate_discharge_step(0.1)
            if data_point:
                time_remaining = st.session_state.predictor.predict_time_remaining(
                    data_point['current'], data_point['voltage'], data_point['power'],
                    data_point['percent_remaining'], data_point['soc'],
                    simulator.nominal_capacity, data_point['time_elapsed']
                )
                data_point['time_remaining'] = time_remaining
                st.session_state.simulation_data.append(data_point)
                
                # Update placeholders smoothly
                capacity_placeholder.metric("🔋 Remaining Capacity", f"{data_point['percent_remaining']:.1f}%")
                current_placeholder.metric("⚡ Current", f"{data_point['current']:.1f} A")
                voltage_placeholder.metric("🔌 Voltage", f"{data_point['voltage']:.1f} V")
                time_rem_placeholder.metric("⏱️ Time Remaining", f"{time_remaining:.1f} hours")
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
        else:
            st.session_state.is_running = False
            st.toast("🎉 Battery simulation completed!")
    st.rerun() # Rerun once after the loop finishes to update UI state
            
else: # Display the final or initial state when not running
    if st.session_state.simulation_data:
        last_data = st.session_state.simulation_data[-1]
        time_rem = last_data.get('time_remaining', 0)
        capacity_placeholder.metric("🔋 Remaining Capacity", f"{last_data['percent_remaining']:.1f}%")
        current_placeholder.metric("⚡ Current", f"{last_data['current']:.1f} A")
        voltage_placeholder.metric("🔌 Voltage", f"{last_data['voltage']:.1f} V")
        time_rem_placeholder.metric("⏱️ Time Remaining", f"{time_rem:.1f} hours")
        time_ela_placeholder.metric("🕐 Time Elapsed", f"{last_data['time_elapsed']:.1f} hours")
        
        df_plot = pd.DataFrame(st.session_state.simulation_data)
        chart_data = df_plot.rename(columns={
            'time_elapsed': 'Time Elapsed (h)',
            'time_remaining': 'Predicted Time Remaining (h)',
            'percent_remaining': 'Battery Capacity (%)'
        }).set_index('Time Elapsed (h)')
        plot_placeholder.line_chart(chart_data[['Predicted Time Remaining (h)', 'Battery Capacity (%)']])
    else:
        plot_placeholder.info("Start the simulation to see live telemetry.")
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
st.markdown("**Features:**\n- **Realistic Battery Simulation**: Models actual Li-ion battery discharge characteristics.\n- **Machine Learning Predictions**: Uses Random Forest to predict remaining battery life.\n- **EV-Style Visualization**: Professional battery gauge similar to electric vehicle displays.\n- **Real-time Updates**: Live simulation with configurable update rates.\n- **Multiple Battery Types**: Support for different battery specifications.")