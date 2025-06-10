import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Battery Life Monitor", page_icon="🔋")

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL1_PATH = os.path.join(BASE_DIR, "..", "models", "battery_random_forest_model1.joblib")
MODEL2_PATH = os.path.join(BASE_DIR, "..", "models", "battery_random_forest_model2.joblib")

TIME_STEPS = 10
SIM_INTERVAL_MINUTES = 5.0
SIM_INTERVAL_HOURS = SIM_INTERVAL_MINUTES / 60.0
DEGRADATION_RATE_PER_INTERVAL = 0.000005
MAX_OPERATIONAL_VOLTAGE = 12.6
MIN_OPERATIONAL_VOLTAGE = 9.4
MAX_SIMULATED_CURRENT = 15.0

# --- Battery Metadata ---
BATTERY_METADATA = {
    "b1":  {"capacity": 81.28, "charged": 81.28, "description": "B1 - 81.28 Ah"},
    "b2":  {"capacity": 85.00, "charged": 85.00, "description": "B2 - 85.00 Ah"},
    "b3":  {"capacity": 88.35, "charged": 88.35, "description": "B3 - 88.35 Ah"},
    "b5":  {"capacity": 85.00, "charged": 85.00, "description": "B5 - 85.00 Ah"},
    "tn1": {"capacity": 85.00, "charged": 80.00, "description": "TN1 - 85.00 Ah (Charged to 80 Ah)"},
}

# --- Initialize Session State for Real-time Plotting ---
def initialize_session_state():
    """Initialize session state variables for real-time plotting"""
    if 'rt_time_remaining' not in st.session_state:
        st.session_state.rt_time_remaining = []
    if 'rt_battery_percent' not in st.session_state:
        st.session_state.rt_battery_percent = []
    if 'rt_voltage' not in st.session_state:
        st.session_state.rt_voltage = []
    if 'rt_current' not in st.session_state:
        st.session_state.rt_current = []
    if 'rt_power' not in st.session_state:
        st.session_state.rt_power = []
    if 'rt_timestamps' not in st.session_state:
        st.session_state.rt_timestamps = []
    if 'rt_simulation_active' not in st.session_state:
        st.session_state.rt_simulation_active = False

def clear_realtime_data():
    """Clear all real-time data for a fresh start"""
    st.session_state.rt_time_remaining = []
    st.session_state.rt_battery_percent = []
    st.session_state.rt_voltage = []
    st.session_state.rt_current = []
    st.session_state.rt_power = []
    st.session_state.rt_timestamps = []

def create_realtime_ev_plot():
    """Create real-time EV-style battery plot using Plotly"""
    if not st.session_state.rt_time_remaining:
        # Create empty plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Battery Level',
                                line=dict(color='#2E86C1', width=3),
                                marker=dict(size=8, color='#2E86C1')))
        
        fig.update_layout(
            title="Real-Time EV Battery Display",
            xaxis_title="Predicted Time Remaining (Hours)",
            yaxis_title="Battery Capacity (%)",
            xaxis=dict(range=[8, 0], autorange=False),  # Inverted for countdown
            yaxis=dict(range=[0, 105], autorange=False),
            template="plotly_white",
            height=500,
            font=dict(size=12),
            title_font=dict(size=16, color='#2c3e50'),
            showlegend=True
        )
        return fig
    
    # Create plot with data
    time_hours = np.array(st.session_state.rt_time_remaining) / 3600.0
    
    fig = go.Figure()
    
    # Main discharge curve
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=st.session_state.rt_battery_percent,
        mode='lines+markers',
        name='Battery Level',
        line=dict(color='#2E86C1', width=3),
        marker=dict(
            size=8,
            color=st.session_state.rt_battery_percent,
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            colorbar=dict(title="Battery %")
        ),
        text=[f'{p:.1f}%' for p in st.session_state.rt_battery_percent],
        textposition="top center",
        textfont=dict(size=10)
    ))
    
    # Add critical zones
    fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                  annotation_text="Low Battery Warning")
    fig.add_hline(y=5, line_dash="dash", line_color="red", 
                  annotation_text="Critical Battery Level")
    
    # Update layout
    x_max = max(time_hours[0] + 0.5, 8) if len(time_hours) > 0 else 8
    x_min = max(0, time_hours[-1] - 0.5) if len(time_hours) > 0 else 0
    
    fig.update_layout(
        title=f"Real-Time EV Battery Display - Live Updates",
        xaxis_title="Predicted Time Remaining (Hours)",
        yaxis_title="Battery Capacity (%)",
        xaxis=dict(range=[x_max, x_min], autorange=False),  # Inverted
        yaxis=dict(range=[0, 105], autorange=False),
        template="plotly_white",
        height=500,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        showlegend=True,
        grid=dict(rows=1, columns=1),
        annotations=[
            dict(
                x=time_hours[-1] if len(time_hours) > 0 else 0,
                y=st.session_state.rt_battery_percent[-1] if st.session_state.rt_battery_percent else 0,
                text=f"Latest: {time_hours[-1]:.1f}h, {st.session_state.rt_battery_percent[-1]:.1f}%" if len(time_hours) > 0 else "",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="white",
                bordercolor="red",
                borderwidth=2
            )
        ] if len(time_hours) > 0 else []
    )
    
    return fig

def create_realtime_metrics_plot():
    """Create real-time metrics plots (Voltage, Current, Power)"""
    if not st.session_state.rt_timestamps:
        return None
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Voltage (V)', 'Current (A)', 'Power (W)'),
        vertical_spacing=0.08
    )
    
    timestamps = list(range(len(st.session_state.rt_timestamps)))
    
    # Voltage plot
    fig.add_trace(go.Scatter(
        x=timestamps, y=st.session_state.rt_voltage,
        mode='lines+markers', name='Voltage',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ), row=1, col=1)
    
    # Current plot
    fig.add_trace(go.Scatter(
        x=timestamps, y=st.session_state.rt_current,
        mode='lines+markers', name='Current',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ), row=2, col=1)
    
    # Power plot
    fig.add_trace(go.Scatter(
        x=timestamps, y=st.session_state.rt_power,
        mode='lines+markers', name='Power',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ), row=3, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Real-Time Battery Metrics",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

# --- Original Utility Functions (keeping your existing code) ---
def simulate_battery_values_revised(
    current_simulation_step,
    cumulative_ah_discharged_start_of_step,
    capacity,
    charged_ah_feature
):
    """Your existing simulation function"""
    degradation_multiplier = 1.0 - (DEGRADATION_RATE_PER_INTERVAL * current_simulation_step)
    current_effective_capacity = capacity * max(0.20, degradation_multiplier)

    intervals_per_24h = (24 * 60) / SIM_INTERVAL_MINUTES
    time_of_day_cycle_position = (current_simulation_step % intervals_per_24h) / intervals_per_24h

    base_current_value = 8 + 7 * np.sin(2 * np.pi * time_of_day_cycle_position)
    simulated_current = min(MAX_SIMULATED_CURRENT, base_current_value + np.random.normal(0, 0.3))
    simulated_current = max(0.1, simulated_current)

    ah_discharged_this_interval = simulated_current * SIM_INTERVAL_HOURS
    total_cumulative_ah_discharged = cumulative_ah_discharged_start_of_step + ah_discharged_this_interval

    if current_effective_capacity > 0:
        current_rc = 1.0 - (total_cumulative_ah_discharged / current_effective_capacity)
    else:
        current_rc = 0.0
    current_rc = max(0.0, min(1.0, current_rc))

    simulated_voltage = MIN_OPERATIONAL_VOLTAGE + (MAX_OPERATIONAL_VOLTAGE - MIN_OPERATIONAL_VOLTAGE) * current_rc
    simulated_voltage += np.random.normal(0, 0.03)
    simulated_voltage = max(MIN_OPERATIONAL_VOLTAGE - 0.2, simulated_voltage)
    simulated_voltage = min(MAX_OPERATIONAL_VOLTAGE + 0.2, simulated_voltage)

    simulated_power = simulated_voltage * simulated_current
    remaining_capacity_percent = current_rc * 100.0

    return (
        simulated_current,
        simulated_voltage,
        ah_discharged_this_interval,
        total_cumulative_ah_discharged,
        simulated_power,
        remaining_capacity_percent,
        charged_ah_feature
    )

@st.cache_resource
def load_ml_models(model1_path, model2_path):
    """Your existing model loading function"""
    try:
        if not os.path.exists(model1_path):
            st.error(f"Model 1 file not found: {model1_path}")
            raise FileNotFoundError(f"Model 1 file not found: {model1_path}")
        if not os.path.exists(model2_path):
            st.error(f"Model 2 file not found: {model2_path}")
            raise FileNotFoundError(f"Model 2 file not found: {model2_path}")

        model1_pipeline, model1_raw_feature_cols = joblib.load(model1_path)
        model2_pipeline, model2_raw_feature_cols = joblib.load(model2_path)

        if not hasattr(model1_pipeline, 'predict') or not hasattr(model2_pipeline, 'predict'):
            st.error("Loaded objects are not valid machine learning models.")
            raise ValueError("Invalid model object loaded.")

        return model1_pipeline, model1_raw_feature_cols, model2_pipeline, model2_raw_feature_cols
    except Exception as e:
        st.exception(f"Error loading models: {e}")
        st.stop()

# --- Enhanced Simulation Function with Real-time Plotting ---
def run_simulation_with_realtime_plotting(
    model1_pipeline, model1_raw_feature_cols,
    model2_pipeline, model2_raw_feature_cols,
    selected_battery_type_code, battery_capacity, charged_ah
):
    """Enhanced simulation with real-time plotting"""
    
    clear_realtime_data()  # Clear previous data
    st.session_state.rt_simulation_active = True
    
    st.info(f"🚀 Starting real-time simulation for {BATTERY_METADATA[selected_battery_type_code]['description']}")

    TIME_STEP = 0
    cumulative_ah_discharged_total = battery_capacity - charged_ah
    feature_data_buffer = []

    # Create layout for real-time display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Real-Time EV Battery Display")
        rt_main_plot = st.empty()
        
        st.markdown("### 📈 Real-Time Battery Metrics")
        rt_metrics_plot = st.empty()
    
    with col2:
        st.markdown("### 🔋 Live Status")
        rt_status = st.empty()
        
        st.markdown("### ⚡ Current Readings")
        rt_current_metric = st.empty()
        rt_voltage_metric = st.empty()
        rt_power_metric = st.empty()
        rt_remaining_metric = st.empty()
        
        st.markdown("### 🎛️ Simulation Controls")
        if st.button("⏹️ Stop Simulation"):
            st.session_state.rt_simulation_active = False
            st.rerun()

    start_time = datetime.now()

    # Main simulation loop with real-time plotting
    while st.session_state.rt_simulation_active:
        TIME_STEP += 1

        # Your existing simulation logic
        current, voltage, ah_out_interval, updated_cumulative_ah, power, remaining_perc, _ = \
            simulate_battery_values_revised(
                current_simulation_step=(TIME_STEP - 1),
                cumulative_ah_discharged_start_of_step=cumulative_ah_discharged_total,
                capacity=battery_capacity,
                charged_ah_feature=charged_ah
            )
        cumulative_ah_discharged_total = updated_cumulative_ah

        # Check termination conditions
        if voltage < MIN_OPERATIONAL_VOLTAGE - 0.5:
            st.session_state.rt_simulation_active = False
            st.success(f"🚨 Simulation ended: Voltage critical ({voltage:.2f}V)")
            break
        if remaining_perc < 0.1 and TIME_STEP > TIME_STEPS:
            st.session_state.rt_simulation_active = False
            st.success(f"🔋 Simulation ended: Battery depleted ({remaining_perc:.2f}%)")
            break

        # Feature engineering (your existing code)
        discharge_rate = current / (voltage + 1e-6)
        discharge_ratio = ah_out_interval / (charged_ah + 1e-6)
        step_index = TIME_STEP

        current_step_features = {
            'Current': float(current),
            'Voltage': float(voltage),
            'Ah Out': float(ah_out_interval),
            'Cumulative Actual Disch Ah': float(cumulative_ah_discharged_total),
            'Power': float(power),
            'Remaining Capacity': float(remaining_perc),
            'capacity': float(battery_capacity),
            'charged': float(charged_ah),
            'discharge_rate': float(discharge_rate),
            'step_index': float(step_index),
            'discharge_ratio': float(discharge_ratio),
            'type': selected_battery_type_code
        }

        # Data validation
        for key, value in current_step_features.items():
            if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                current_step_features[key] = 0.0

        feature_data_buffer.append(current_step_features)

        # Make predictions
        time_remaining_pred_seconds = 0
        if len(feature_data_buffer) >= TIME_STEPS:
            try:
                df_for_model1_input = pd.DataFrame(feature_data_buffer).reindex(columns=model1_raw_feature_cols)
                pred_model1_ratio = model1_pipeline.predict(df_for_model1_input)

                df_for_model2_input = pd.DataFrame(feature_data_buffer).reindex(columns=model2_raw_feature_cols)
                if 'prediction' not in df_for_model2_input.columns:
                    df_for_model2_input['prediction'] = 0.0
                df_for_model2_input['prediction'] = pred_model1_ratio[-1]
                df_for_model2_input = df_for_model2_input.reindex(columns=model2_raw_feature_cols)

                time_remaining_pred_seconds = model2_pipeline.predict(df_for_model2_input)[-1]
                feature_data_buffer.pop(0)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                time_remaining_pred_seconds = 0

        # Update real-time data
        st.session_state.rt_time_remaining.append(max(0, time_remaining_pred_seconds))
        st.session_state.rt_battery_percent.append(remaining_perc)
        st.session_state.rt_voltage.append(voltage)
        st.session_state.rt_current.append(current)
        st.session_state.rt_power.append(power)
        st.session_state.rt_timestamps.append(datetime.now().strftime('%H:%M:%S'))

        # Keep only last 50 points for performance
        if len(st.session_state.rt_time_remaining) > 50:
            st.session_state.rt_time_remaining.pop(0)
            st.session_state.rt_battery_percent.pop(0)
            st.session_state.rt_voltage.pop(0)
            st.session_state.rt_current.pop(0)
            st.session_state.rt_power.pop(0)
            st.session_state.rt_timestamps.pop(0)

        # Update real-time plots
        with col1:
            rt_main_plot.plotly_chart(create_realtime_ev_plot(), use_container_width=True)
            metrics_fig = create_realtime_metrics_plot()
            if metrics_fig:
                rt_metrics_plot.plotly_chart(metrics_fig, use_container_width=True)

        # Update metrics
        with col2:
            elapsed_time = datetime.now() - start_time
            rt_status.info(f"🔄 **Step:** {TIME_STEP}\n\n⏱️ **Elapsed:** {str(elapsed_time).split('.')[0]}")
            
            rt_current_metric.metric("⚡ Current", f"{current:.2f} A", delta=f"{current-8:.2f}")
            rt_voltage_metric.metric("🔌 Voltage", f"{voltage:.2f} V", delta=f"{voltage-12:.2f}")
            rt_power_metric.metric("💡 Power", f"{power:.1f} W", delta=f"{power-100:.1f}")
            
            h_rem = int(time_remaining_pred_seconds // 3600)
            m_rem = int((time_remaining_pred_seconds % 3600) // 60)
            rt_remaining_metric.metric("⏳ Time Left", f"{h_rem:02d}h {m_rem:02d}m")

        # Control simulation speed
        time.sleep(1.0)  # 1 second between updates
        
        # Force rerun to update the display
        if st.session_state.rt_simulation_active:
            st.rerun()

    st.session_state.rt_simulation_active = False
    st.balloons()
    st.success("🎉 Real-time simulation completed!")

# --- Main Application ---
def main():
    initialize_session_state()
    
    st.title("🔋 Advanced Battery Life Monitor with Real-Time Plotting")
    st.markdown("**Real-time battery discharge simulation with live EV-style visualization**")

    st.divider()

    # Model Loading
    st.header("1. 🤖 Load Machine Learning Models")
    try:
        model1_pipeline, model1_raw_feature_cols, model2_pipeline, model2_raw_feature_cols = \
            load_ml_models(MODEL1_PATH, MODEL2_PATH)
        st.success("✅ Random Forest Models loaded successfully!")
    except:
        st.error("❌ Could not load models. Using simulation mode only.")
        model1_pipeline = model1_raw_feature_cols = model2_pipeline = model2_raw_feature_cols = None

    st.divider()

    # Battery Selection
    st.header("2. 🔋 Select Battery Parameters")
    
    battery_type_options = [data['description'] for data in BATTERY_METADATA.values()]
    selected_battery_desc = st.selectbox(
        "Select Battery Type:",
        options=battery_type_options,
        index=0,
        help="Choose the battery type for simulation"
    )

    selected_battery_type_code = [code for code, data in BATTERY_METADATA.items() 
                                  if data['description'] == selected_battery_desc][0]
    selected_battery_data = BATTERY_METADATA[selected_battery_type_code]

    battery_capacity = selected_battery_data['capacity']
    max_charge_ah = selected_battery_data['charged']

    charged_ah = st.number_input(
        f"Initial Charge (Ah, max: {max_charge_ah:.2f}):",
        min_value=0.0,
        max_value=float(max_charge_ah),
        value=float(max_charge_ah),
        step=0.1,
        format="%.2f"
    )

    st.info(f"**Selected:** {selected_battery_desc} (Capacity: {battery_capacity:.2f} Ah)")

    st.divider()

    # Real-time Simulation Controls
    st.header("3. 🚀 Real-Time Simulation & Plotting")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("▶️ Start Real-Time Simulation", disabled=st.session_state.rt_simulation_active):
            if model1_pipeline and model2_pipeline:
                run_simulation_with_realtime_plotting(
                    model1_pipeline, model1_raw_feature_cols,
                    model2_pipeline, model2_raw_feature_cols,
                    selected_battery_type_code, battery_capacity, charged_ah
                )
            else:
                st.error("Models not loaded. Cannot run simulation.")
    
    with col2:
        if st.button("🔄 Reset Data"):
            clear_realtime_data()
            st.session_state.rt_simulation_active = False
            st.success("Data cleared!")
            st.rerun()
    
    with col3:
        if st.session_state.rt_simulation_active:
            st.success("🟢 Simulation Running")
        else:
            st.info("⚫ Simulation Stopped")

    # Display current real-time data if available
    if st.session_state.rt_time_remaining:
        st.divider()
        st.header("4. 📊 Current Real-Time Display")
        
        # Show latest plot
        st.plotly_chart(create_realtime_ev_plot(), use_container_width=True)
        
        # Show metrics plot
        metrics_fig = create_realtime_metrics_plot()
        if metrics_fig:
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Show data summary
        st.subheader("📈 Data Summary")
        if len(st.session_state.rt_battery_percent) > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Battery", f"{st.session_state.rt_battery_percent[-1]:.1f}%")
            col2.metric("Latest Voltage", f"{st.session_state.rt_voltage[-1]:.2f}V")
            col3.metric("Latest Current", f"{st.session_state.rt_current[-1]:.2f}A")
            col4.metric("Latest Power", f"{st.session_state.rt_power[-1]:.1f}W")

    st.divider()
    
    # About section
    st.header("ℹ️ About This Application")
    st.markdown("""
    **🔋 Advanced Battery Life Monitor Features:**
    
    - **Real-Time EV Display**: Live updating battery visualization similar to electric vehicle dashboards
    - **Multiple Metrics Tracking**: Voltage, current, power monitoring in real-time
    - **Machine Learning Predictions**: Uses Random Forest models for accurate time-remaining predictions
    - **Interactive Plotly Charts**: Professional, interactive visualizations with zoom and hover features
    - **Live Status Updates**: Real-time KPIs and metrics that update every second
    - **Simulation Controls**: Start, stop, and reset functionality for complete control
    
    **🚀 Real-Time Features:**
    - Live battery discharge curve updates
    - Real-time voltage, current, and power plotting
    - Dynamic time-remaining predictions
    - Professional EV-style visualization
    - Automatic data buffering (last 50 points for performance)
    """)

if __name__ == "__main__":
    main()
