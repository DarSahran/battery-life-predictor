import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

MODEL1_PATH = "../models/battery_random_forest_model.joblib"
MODEL2_PATH = "../models/battery_random_forest_model2.joblib"
TIME_STEPS = 10

# --- Simulation Parameters ---
SIM_INTERVAL_MINUTES = 5.0
SIM_INTERVAL_HOURS = SIM_INTERVAL_MINUTES / 60.0
DEGRADATION_RATE_PER_INTERVAL = 0.000005
MAX_OPERATIONAL_VOLTAGE = 12.6
MIN_OPERATIONAL_VOLTAGE = 9.4
MAX_SIMULATED_CURRENT = 15.0

try:
    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found. Please check paths: \n{MODEL1_PATH}\n{MODEL2_PATH}")
    exit()
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

battery_type_capacities = {
    "B1":81.28,
    "B2":85.0,
    "B3":88.35,
    "TN1":85.0,
    "B5":85.0
}

def simulate_battery_values_revised(
    current_simulation_step,
    cumulative_ah_discharged_start_of_step,
    capacity,
    charged_ah_feature
):
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

# --- Main Simulation Loop ---
battery_type_index = int(input("Enter battery type index (0-B1,1-B2,2-B3,3-TN1,4-B5): "))

TIME_STEP = 0
cumulative_ah_total = 0.0

data_buffer = []
prediction_timestamps = []
prediction_values = []
remaining_capacity_percent_list = []

battery_keys_list = list(battery_type_capacities.keys())
if battery_type_index >= len(battery_keys_list):
    print(f"Warning: battery_type_index {battery_type_index} out of bounds. Defaulting to 0.")
    battery_type_index = 0
current_battery_code = battery_keys_list[battery_type_index]
capacity = float(battery_type_capacities[current_battery_code])

charged_ah = int(input("Enter charged Ah value (0-100): "))
if charged_ah < 0 or charged_ah > 100:
    print("Warning: charged Ah value out of bounds. Defaulting to 0.")
    charged_ah = 0
if charged_ah > 100:
    charged_ah = 100

print(f"Starting simulation for: {current_battery_code} (Nominal Capacity: {capacity} Ah)")
simulation_start_time = datetime.now()

for i in range(600):
    TIME_STEP += 1

    current, voltage, ah_out, updated_cumulative_ah, power, remaining_perc, charged_ah_val = \
        simulate_battery_values_revised(
            current_simulation_step=(TIME_STEP - 1),
            cumulative_ah_discharged_start_of_step=cumulative_ah_total,
            capacity=capacity,
            charged_ah_feature=charged_ah
        )

    cumulative_ah_total = updated_cumulative_ah

    if voltage < MIN_OPERATIONAL_VOLTAGE:
        print(f"INFO: Voltage ({voltage:.2f}V) fell below cutoff ({MIN_OPERATIONAL_VOLTAGE:.2f}V) at step {TIME_STEP}. Stopping.")
        break
    if remaining_perc < 0.1:
        print(f"INFO: Remaining capacity ({remaining_perc:.2f}%) is critical at step {TIME_STEP}. Stopping.")
        break

    base_row = {
        'Current': float(current),
        'Voltage': float(voltage),
        'Ah Out': float(ah_out),
        'Cumulative Actual Disch Ah': float(cumulative_ah_total),
        'Power': float(power),
        'Remaining Capacity': float(remaining_perc),
        'type': current_battery_code,
        'capacity': float(capacity),
        'charged': float(charged_ah_val),
    }

    data_buffer.append(base_row)

    if len(data_buffer) >= TIME_STEPS:
        X_input_df = pd.DataFrame(data_buffer)

        try:
            pred_model1 = model1.predict(X_input_df)
            X_input_df_for_model2 = X_input_df.copy()

            if isinstance(pred_model1, np.ndarray) and len(pred_model1) > 0:
                X_input_df_for_model2['prediction'] = pred_model1[-1]
            else:
                X_input_df_for_model2['prediction'] = pred_model1

            y_pred_model2 = model2.predict(X_input_df_for_model2)
            time_remaining_seconds = y_pred_model2[-1]

            # Store predictions and corresponding battery capacity
            prediction_timestamps.append(TIME_STEP)
            prediction_values.append(time_remaining_seconds)
            remaining_capacity_percent_list.append(remaining_perc)

            data_buffer.pop(0)

        except Exception as e:
            print(f"ERROR during model prediction at step {TIME_STEP}: {e}")
            if data_buffer:
                data_buffer.pop(0)
            pass

    # Periodic status update and plotting
    if TIME_STEP % 50 == 0:
        print(f"Progress: Step {TIME_STEP}, V={voltage:.2f}, SoC={remaining_perc:.1f}%")

        # --- EV-Style Plot: Time Remaining vs Battery Capacity (Descending Y-axis) ---
        if prediction_values:
            plt.figure(figsize=(12, 8))
            time_remaining_hours = np.array(prediction_values) / 3600.0
            
            # Plot: X-axis = Time Remaining, Y-axis = Battery Capacity %
            plt.plot(time_remaining_hours, remaining_capacity_percent_list, marker='o', linestyle='-', linewidth=2, markersize=6)
            
            # Set Y-axis: 0-100% with step of 5, DESCENDING order (100 at bottom, 0 at top)
            plt.ylim(0, 100)
            plt.yticks(np.arange(0, 101, 5))
            plt.gca().invert_yaxis()  # Invert Y-axis for descending order[5][9]
            
            # Add data labels
            for t_rem, cap_pct in zip(time_remaining_hours, remaining_capacity_percent_list):
                plt.text(t_rem, cap_pct, f"{cap_pct:.1f}%", fontsize=8, ha='center', va='bottom')
            
            plt.xlabel("Predicted Time Remaining (Hours)")
            plt.ylabel("Battery Capacity (%)")
            plt.title(f"EV-Style Battery Display - {current_battery_code} (Charged: {charged_ah}Ah)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("No predictions were generated to plot. Simulation may have ended early or buffer not filled.")

print(f"\nSimulation for {current_battery_code} finished at step {TIME_STEP}.")
