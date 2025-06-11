import joblib
import os
import pandas as pd

class prediction():
    def __init__(self):
        self.model1_path = "../Final_codes/battery_random_forest_model1.joblib"
        self.model2_path = "../Final_codes/battery_random_forest_model2.joblib"
        self.model1 = joblib.load(self.model1_path)
        self.model2 = joblib.load(self.model2_path)

    def predict_model1(self, data):
        Y_pred = self.model1.predict(data)
        return Y_pred

    def predict_model2(self, Y_pred, data):
        pred_TOD = Y_pred[0][0]
        pred_CDC = Y_pred[0][1]
        data['predicted TOD'] = pred_TOD
        data['predicted CDC'] = pred_CDC

        final_output = self.model2.predict(data)
        return final_output[0][0],final_output[0][1]

    def feature_derivator(self):
        # --- Battery specifications and mapping ---
        BATTERY_SPECS = {
            "TN1": {"nominal_capacity": 85.0, "charged_capacity": 80.0, "nominal_voltage": 12.6},
            "B1": {"nominal_capacity": 81.28, "charged_capacity": 81.28, "nominal_voltage": 12.6},
            "B2": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6},
            "B3": {"nominal_capacity": 88.35, "charged_capacity": 88.35, "nominal_voltage": 12.6},
            "B5": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6}
        }
        BATTERY_TYPE_MAPPING = {key: i for i, key in enumerate(BATTERY_SPECS.keys())}

        # --- Battery test metadata ---
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

        def get_matching_tests(battery_type):
            """Get all test file names for a given battery type."""
            return [test for test, data in BATTERY_METADATA.items() if data["type"].lower() == battery_type.lower()]

        def select_test(matching_tests):
            """Prompt user to select a test file from the list."""
            print("Available test files:")
            for i, test in enumerate(matching_tests, 1):
                print(f"{i}: {test}")
            while True:
                try:
                    choice = int(input("Select a test file (enter number): ")) - 1
                    if 0 <= choice < len(matching_tests):
                        return matching_tests[choice]
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")

        # --- STEP 1: Select battery type ---
        print("Select battery type:")
        for i, key in enumerate(BATTERY_SPECS.keys(), 1):
            print(f"{i}: {key}")
        option = int(input("Enter option number here: ")) - 1
        selected_battery = list(BATTERY_SPECS.keys())[option]
        print(f"Selected battery: {selected_battery}")

        # --- STEP 2: Get matching test files ---
        matching_tests = get_matching_tests(selected_battery)
        if not matching_tests:
            raise ValueError(f"No test files found for battery type: {selected_battery}")

        # --- STEP 3: Select test file ---
        selected_test = select_test(matching_tests)
        print(f"Selected test file: {selected_test}")

        # --- STEP 4: Load CSV ---
        csv_path = f"../data/processed/{selected_test}.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise IOError(f"Could not load file {csv_path}: {e}")

        # --- STEP 5: Process each row (1 min per row) ---
        specs = BATTERY_SPECS[selected_battery]
        battery_type_idx = BATTERY_TYPE_MAPPING[selected_battery]
        time_step_h = 1 / 60  # 1 minute per row
        cumulative_ah = 0.0
        results = []

        for idx, row in df.iterrows():
            current = row["Current"]
            voltage = row["Voltage"]
            ah_out = current * time_step_h
            cumulative_ah += ah_out
            remaining_capacity = specs["charged_capacity"] - cumulative_ah
            power = voltage * current
            time_to_depletion = (remaining_capacity / current) * 3600 if current > 0 else float('inf')
            discharge_rate = current / specs["nominal_capacity"]
            discharge_ratio = ah_out / specs["charged_capacity"] if specs["charged_capacity"] > 0 else 0

            step_data = {
                "battery_type": selected_battery,
                "current": current,
                "voltage": voltage,
                "Ah Out": ah_out,
                "Power": power,
                "Remaining Capacity": remaining_capacity,
                "capacity": specs["nominal_capacity"],
                "charged": specs["charged_capacity"],
                "discharge_rate": discharge_rate,
                "discharge_ratio": discharge_ratio,
                "step_index": idx
            }
            results.append(step_data)

        # --- STEP 6: Convert to DataFrame and print results ---
        results_df = pd.DataFrame(results)
        print("\nSelected battery:", selected_battery)
        print("Selected test file:", selected_test)
        print("\nFirst row of calculated results:")
        print(results_df.head(1))

        # --- STEP 7: Latest step for model input ---
        latest_input = results_df.iloc[-1].to_dict()
        print("\nLatest step input for model:")
        for key, value in latest_input.items():
            print(f"{key}: {value}")