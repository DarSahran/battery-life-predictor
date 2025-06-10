import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Complete Battery Metadata ---
BATTERY_METADATA = {
    "TEST_1_processed":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_2_processed":  {"capacity": 81.28, "charged": 81.28, "type": "b1"},
    "TEST_3_processed":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_4_processed":  {"capacity": 85,    "charged": 85,    "type": "b2"},
    "TEST_5_processed":  {"capacity": 88.81, "charged": 88.81, "type": "b2"},
    "TEST_6_processed":  {"capacity": 81.84, "charged": 81.84, "type": "b1"},
    "TEST_7_processed":  {"capacity": 81.84, "charged": 36,    "type": "b1"},
    "TEST_8_processed":  {"capacity": 88.81, "charged": 27,    "type": "b2"},
    "TEST_9_processed":  {"capacity": 85,    "charged": 80,    "type": "tn1"},
    "TEST_10_processed": {"capacity": 85,    "charged": 54,    "type": "tn1"},
    "TEST_11_processed": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_12_processed": {"capacity": 85,    "charged": 67,    "type": "b5"},
    "TEST_13_processed": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "TEST_14_processed": {"capacity": 88.83, "charged": 52,    "type": "b3"},
    "TEST_15_processed": {"capacity": 88.35, "charged": 70,    "type": "b3"},
    "TEST_16_processed": {"capacity": 88.35, "charged": 61,    "type": "b3"},
    "TEST_17_processed": {"capacity": 88.35, "charged": 88.35, "type": "b3"},
}

MODEL1_PATH = "../models/battery_random_forest_model1.joblib"
MODEL2_PATH = "../models/battery_random_forest_model2.joblib"
TIME_STEPS = 10

SIM_INTERVAL_MINUTES = 5.0
SIM_INTERVAL_HOURS = SIM_INTERVAL_MINUTES / 60.0
DEGRADATION_RATE_PER_INTERVAL = 0.000005
MAX_OPERATIONAL_VOLTAGE = 12.6
MIN_OPERATIONAL_VOLTAGE = 9.4
MAX_SIMULATED_CURRENT = 15.0

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

def main():
    # Load models
    try:
        model1 = joblib.load(MODEL1_PATH)
        model2 = joblib.load(MODEL2_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Get unique battery types and their max capacity from metadata
    battery_type_to_max_capacity = {}
    for entry in BATTERY_METADATA.values():
        btype = entry['type']
        cap = entry['capacity']
        if btype not in battery_type_to_max_capacity or cap > battery_type_to_max_capacity[btype]:
            battery_type_to_max_capacity[btype] = cap

    battery_types = sorted(battery_type_to_max_capacity.keys())
    print("Available battery types:")
    for i, t in enumerate(battery_types):
        print(f"{i+1}. {t.upper()} (Max capacity: {battery_type_to_max_capacity[t]}Ah)")
    type_choice = int(input("\nSelect battery type (number): ")) - 1
    if type_choice < 0 or type_choice >= len(battery_types):
        print("Invalid selection. Using first type.")
        type_choice = 0
    selected_type = battery_types[type_choice]
    capacity = battery_type_to_max_capacity[selected_type]

    # User enters charge level
    charged_ah = float(input(f"Enter charged Ah (0 to {capacity}): "))
    if charged_ah > capacity:
        print(f"Charged value too high, using max {capacity}Ah.")
        charged_ah = capacity
    if charged_ah < 0:
        print("Charged value too low, using 0.")
        charged_ah = 0

    print(f"\nSimulating for type: {selected_type.upper()}, capacity: {capacity}Ah, charged: {charged_ah}Ah")

    # Simulation variables
    TIME_STEP = 0
    cumulative_ah_total = capacity - charged_ah  # Initial discharged Ah
    data_buffer = []
    prediction_values = []
    remaining_capacity_percent_list = []

    for _ in range(600):
        TIME_STEP += 1

        current, voltage, ah_out, updated_cumulative_ah, power, remaining_perc, _ = \
            simulate_battery_values_revised(
                current_simulation_step=(TIME_STEP - 1),
                cumulative_ah_discharged_start_of_step=cumulative_ah_total,
                capacity=capacity,
                charged_ah_feature=charged_ah
            )

        cumulative_ah_total = updated_cumulative_ah

        if voltage < MIN_OPERATIONAL_VOLTAGE or remaining_perc < 0.1:
            print(f"Simulation ended: Voltage={voltage:.2f}V, SOC={remaining_perc:.2f}% at step {TIME_STEP}")
            break

        data_buffer.append({
            'Current': current,
            'Voltage': voltage,
            'Ah Out': ah_out,
            'Cumulative Actual Disch Ah': cumulative_ah_total,
            'Power': power,
            'Remaining Capacity': remaining_perc,
            'type': selected_type,
            'capacity': capacity,
            'charged': charged_ah
        })

        if len(data_buffer) >= TIME_STEPS:
            X_input_df = pd.DataFrame(data_buffer[-TIME_STEPS:])
            try:
                pred_model1 = model1.predict(X_input_df)
                X_input_df_for_model2 = X_input_df.copy()
                if isinstance(pred_model1, np.ndarray):
                    X_input_df_for_model2['prediction'] = pred_model1[-1]
                else:
                    X_input_df_for_model2['prediction'] = pred_model1
                y_pred_model2 = model2.predict(X_input_df_for_model2)
                time_remaining_seconds = y_pred_model2[-1]

                prediction_values.append(time_remaining_seconds)
                remaining_capacity_percent_list.append(remaining_perc)
                data_buffer.pop(0)
            except Exception as e:
                print(f"Prediction error at step {TIME_STEP}: {e}")
                if data_buffer:
                    data_buffer.pop(0)

        # Plot every 50 steps
        if TIME_STEP % 50 == 0 and prediction_values:
            plt.figure(figsize=(12, 7))
            time_remaining_hours = np.array(prediction_values) / 3600.0
            plt.plot(time_remaining_hours, remaining_capacity_percent_list, 
                    marker='o', markersize=6, linestyle='-', linewidth=1.5, color='#3498db')
            plt.gca().invert_xaxis()
            plt.xlim(8, 0)
            plt.ylim(0, 100)
            plt.yticks(np.arange(0, 101, 10))
            plt.grid(True, alpha=0.3)
            plt.title(f"Battery Discharge: {selected_type.upper()} | {capacity}Ah | Charged: {charged_ah}Ah")
            plt.xlabel("Time Remaining (hours)")
            plt.ylabel("Remaining Capacity (%)")
            for t_rem, cap in zip(time_remaining_hours, remaining_capacity_percent_list):
                plt.text(t_rem, cap, f"{cap:.1f}%", ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.show()

    print(f"\nSimulation completed for {selected_type.upper()} ({capacity}Ah, {charged_ah}Ah charged) after {TIME_STEP} steps")

if __name__ == "__main__":
    main()
