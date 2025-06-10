import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            
        # Realistic discharge current based on capacity (higher current when more charged)
        soc = self.current_capacity / self.nominal_capacity  # State of Charge
        
        # Current varies with SOC and some randomness (typical EV discharge pattern)
        base_current = 8 + 7 * soc + np.random.normal(0, 0.5)
        current = max(0.5, min(15.0, base_current))  # Clamp between 0.5A and 15A
        
        # Voltage varies with SOC (realistic Li-ion curve)
        voltage = 9.5 + 3.1 * soc + np.random.normal(0, 0.05)
        voltage = max(9.0, min(12.8, voltage))
        
        # Calculate discharge
        ah_discharged = current * time_step_hours
        self.current_capacity = max(0, self.current_capacity - ah_discharged)
        
        # Calculate remaining percentage
        remaining_percent = (self.current_capacity / self.nominal_capacity) * 100
        
        # Power calculation
        power = voltage * current
        
        self.time_elapsed += time_step_hours
        
        # Store data point
        data_point = {
            'time_elapsed': self.time_elapsed,
            'current': current,
            'voltage': voltage,
            'power': power,
            'ah_remaining': self.current_capacity,
            'percent_remaining': remaining_percent,
            'soc': soc
        }
        
        self.discharge_data.append(data_point)
        return data_point

class BatteryPredictor:
    def __init__(self):
        # Create a simple but effective prediction model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self):
        """Generate synthetic training data for the model"""
        training_data = []
        
        for battery_type in BATTERY_SPECS.keys():
            for initial_charge in np.linspace(10, BATTERY_SPECS[battery_type]["charged_capacity"], 20):
                sim = BatterySimulator(battery_type, initial_charge)
                
                while sim.current_capacity > 0.1:  # Simulate until nearly depleted
                    data_point = sim.simulate_discharge_step(0.05)  # 3-minute steps
                    if data_point is None:
                        break
                        
                    # Calculate time remaining until depletion
                    estimated_time_remaining = sim.current_capacity / (data_point['current'] + 0.1)
                    
                    # Features for prediction
                    features = [
                        data_point['current'],
                        data_point['voltage'],
                        data_point['power'],
                        data_point['percent_remaining'],
                        data_point['soc'],
                        sim.nominal_capacity,
                        data_point['time_elapsed']
                    ]
                    
                    training_data.append(features + [estimated_time_remaining])
        
        return np.array(training_data)
    
    def train_model(self):
        """Train the prediction model"""
        with st.spinner("Training prediction model..."):
            training_data = self.generate_training_data()
            X = training_data[:, :-1]  # Features
            y = training_data[:, -1]   # Target (time remaining)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
    def predict_time_remaining(self, current, voltage, power, percent_remaining, soc, nominal_capacity, time_elapsed):
        """Predict remaining battery time in hours"""
        if not self.is_trained:
            return 0
            
        features = np.array([[current, voltage, power, percent_remaining, soc, nominal_capacity, time_elapsed]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return max(0, prediction)

def create_ev_style_plot(discharge_data, battery_type, initial_charge):
    """Create a professional EV-style battery discharge plot"""
    if not discharge_data:
        return None
        
    df = pd.DataFrame(discharge_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set style
    plt.style.use('default')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Main discharge curve
    line = ax.plot(df['time_remaining'], df['percent_remaining'], 
                   color='#2E86C1', linewidth=2.5, alpha=0.8)
    
    # Add data points
    scatter = ax.scatter(df['time_remaining'], df['percent_remaining'], 
                        c=df['percent_remaining'], cmap='RdYlGn', 
                        s=30, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Annotations for key points
    for i in range(0, len(df), max(1, len(df)//10)):  # Annotate every 10th point
        if i < len(df):
            ax.annotate(f'{df.iloc[i]["percent_remaining"]:.1f}%', 
                       (df.iloc[i]['time_remaining'], df.iloc[i]['percent_remaining']),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7, color='#2E86C1')
    
    # Final point annotation
    if len(df) > 0:
        final_time = df['time_remaining'].iloc[-1]
        final_percent = df['percent_remaining'].iloc[-1]
        ax.annotate(f'({final_time:.1f}h, {final_percent:.1f}%)', 
                   (final_time, final_percent),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Customize the plot
    ax.set_xlabel('Predicted Time Remaining (Hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Battery Capacity (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'EV-Style Battery Display - {battery_type} (Charged: {initial_charge:.0f}Ah)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set limits and grid
    ax.set_xlim(0, max(8, df['time_remaining'].max() * 1.1) if len(df) > 0 else 8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Invert x-axis to show countdown
    ax.invert_xaxis()
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Battery Level (%)', rotation=270, labelpad=20)
    
    # Style improvements
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    
    return fig

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = BatteryPredictor()
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Main App
st.title("🔋 EV Battery Life Predictor")
st.markdown("**Real-time battery discharge simulation with ML-powered time remaining predictions**")

# Sidebar controls
st.sidebar.header("Battery Configuration")
selected_battery = st.sidebar.selectbox("Select Battery Type:", list(BATTERY_SPECS.keys()))
battery_info = BATTERY_SPECS[selected_battery]

initial_charge = st.sidebar.slider(
    "Initial Charge (Ah):", 
    min_value=5.0, 
    max_value=battery_info["charged_capacity"], 
    value=battery_info["charged_capacity"],
    step=0.5
)

st.sidebar.markdown("---")
simulation_speed = st.sidebar.slider("Simulation Speed (updates/sec):", 1, 10, 5)

# Train model if not trained
if not st.session_state.predictor.is_trained:
    if st.sidebar.button("🚀 Initialize Prediction Model"):
        st.session_state.predictor.train_model()
        st.sidebar.success("Model trained successfully!")

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Battery Status")
    
    if st.button("▶️ Start Simulation", disabled=not st.session_state.predictor.is_trained):
        st.session_state.is_running = True
        st.session_state.simulation_data = []
        st.rerun()
    
    if st.button("⏸️ Stop Simulation"):
        st.session_state.is_running = False
        st.rerun()
    
    if st.button("🔄 Reset"):
        st.session_state.simulation_data = []
        st.session_state.is_running = False
        st.rerun()

# Live simulation
if st.session_state.is_running and st.session_state.predictor.is_trained:
    simulator = BatterySimulator(selected_battery, initial_charge)
    
    # Restore previous data if exists
    if st.session_state.simulation_data:
        last_data = st.session_state.simulation_data[-1]
        simulator.current_capacity = last_data['ah_remaining']
        simulator.time_elapsed = last_data['time_elapsed']
    
    # Status metrics
    with col2:
        if simulator.current_capacity > 0:
            data_point = simulator.simulate_discharge_step(0.1)  # 6-minute steps
            
            if data_point:
                # Predict time remaining
                time_remaining = st.session_state.predictor.predict_time_remaining(
                    data_point['current'], data_point['voltage'], data_point['power'],
                    data_point['percent_remaining'], data_point['soc'],
                    simulator.nominal_capacity, data_point['time_elapsed']
                )
                
                data_point['time_remaining'] = time_remaining
                st.session_state.simulation_data.append(data_point)
                
                # Display metrics
                st.metric("🔋 Remaining Capacity", f"{data_point['percent_remaining']:.1f}%")
                st.metric("⚡ Current", f"{data_point['current']:.1f} A")
                st.metric("🔌 Voltage", f"{data_point['voltage']:.1f} V")
                st.metric("⏱️ Time Remaining", f"{time_remaining:.1f} hours")
                st.metric("🕐 Time Elapsed", f"{data_point['time_elapsed']:.1f} hours")
                
                # Auto-refresh
                time.sleep(1/simulation_speed)
                st.rerun()
        else:
            st.session_state.is_running = False
            st.success("🎉 Battery simulation completed!")

# Plot results
with col1:
    st.header("Battery Discharge Curve")
    
    if st.session_state.simulation_data:
        fig = create_ev_style_plot(st.session_state.simulation_data, selected_battery, initial_charge)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Start simulation to see the discharge curve")

# Data table
if st.session_state.simulation_data:
    st.header("Simulation Data")
    df_display = pd.DataFrame(st.session_state.simulation_data)
    df_display = df_display[['time_elapsed', 'percent_remaining', 'current', 'voltage', 'time_remaining']].round(2)
    df_display.columns = ['Time Elapsed (h)', 'Battery (%)', 'Current (A)', 'Voltage (V)', 'Time Remaining (h)']
    st.dataframe(df_display.tail(10))

# Information panel
st.markdown("---")
st.header("ℹ️ About This Application")
st.markdown("""
**Features:**
- **Realistic Battery Simulation**: Models actual Li-ion battery discharge characteristics[4][5]
- **Machine Learning Predictions**: Uses Random Forest to predict remaining battery life[6]
- **EV-Style Visualization**: Professional battery gauge similar to electric vehicle displays[5]
- **Real-time Updates**: Live simulation with configurable update rates
- **Multiple Battery Types**: Support for different battery specifications

**How it works:**
1. Select your battery type and initial charge level
2. Initialize the ML prediction model (one-time setup)
3. Start the simulation to see real-time discharge prediction
4. The system predicts remaining time based on current discharge patterns

**Technical Details:**
- Uses Random Forest regression for time remaining predictions
- Incorporates realistic voltage curves and current patterns
- Features state-of-charge dependent discharge characteristics
- Provides professional EV-style battery visualization
""")
