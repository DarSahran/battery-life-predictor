import joblib
import os
import pandas as pd

class Prediction:
    def __init__(self):
        self.model1_path = "../Final_codes/battery_random_forest_model1.joblib"
        self.model2_path = "../Final_codes/battery_random_forest_model2.joblib"
        self.model1 = joblib.load(self.model1_path)
        self.model2 = joblib.load(self.model2_path)

        # Battery specifications and mapping
        self.BATTERY_SPECS = {
            "TN1": {"nominal_capacity": 85.0, "charged_capacity": 80.0, "nominal_voltage": 12.6},
            "B1": {"nominal_capacity": 81.28, "charged_capacity": 81.28, "nominal_voltage": 12.6},
            "B2": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6},
            "B3": {"nominal_capacity": 88.35, "charged_capacity": 88.35, "nominal_voltage": 12.6},
            "B5": {"nominal_capacity": 85.0, "charged_capacity": 85.0, "nominal_voltage": 12.6}
        }
        self.BATTERY_TYPE_MAPPING = {key: i for i, key in enumerate(self.BATTERY_SPECS.keys())}

        # Battery test metadata
        self.BATTERY_METADATA = {
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

    def predict_model1(self, data):
        """Predict using model1."""
        Y_pred = self.model1.predict(data)
        return Y_pred

    def predict_model2(self, Y_pred, data):
        """Predict using model2, given predictions from model1 and input data."""
        # Ensure Y_pred is 1D or 2D as expected by your model2
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        # Add predictions as new columns (without modifying input)
        data_with_preds = data.copy()
        data_with_preds['predicted_TOD'] = Y_pred[:, 0]
        # If model2 expects two features, you may need to add a dummy column or reshape
        # For simplicity, this example assumes model2 expects (n_samples, 2), but you may need to adapt.
        if Y_pred.shape[1] == 1:
            # Pad with zeros or another feature if needed
            Y_pred_2d = pd.DataFrame({'predicted_TOD': Y_pred[:, 0], 'dummy': 0})
        else:
            Y_pred_2d = pd.DataFrame(Y_pred, columns=['predicted_TOD', 'predicted_CDC'])
        # Merge with main features (assuming model2 expects full DataFrame with predictions)
        data_with_preds = pd.concat([data_with_preds, Y_pred_2d], axis=1)
        # Predict with model2
        final_output = self.model2.predict(data_with_preds)
        # If model2 returns two outputs, return both
        return final_output[0, 0], final_output[0, 1] if final_output.shape[1] == 2 else (final_output[0, 0], None)

    def select_battery_and_test(self):
        """Select battery type and test file using CLI."""
        print("Select battery type:")
        for i, key in enumerate(self.BATTERY_SPECS.keys(), 1):
            print(f"{i}: {key}")
        option = int(input("Enter option number here: ")) - 1
        selected_battery = list(self.BATTERY_SPECS.keys())[option]
        print(f"Selected battery: {selected_battery}")

        matching_tests = self.get_matching_tests(selected_battery)
        if not matching_tests:
            raise ValueError(f"No test files found for battery type: {selected_battery}")

        selected_test = self.select_test(matching_tests)
        print(f"Selected test file: {selected_test}")
        return selected_battery, selected_test

    def get_matching_tests(self, battery_type):
        """Get all test file names for a given battery type."""
        return [test for test, data in self.BATTERY_METADATA.items() if data["type"].lower() == battery_type.lower()]

    def select_test(self, matching_tests):
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

    def load_test_data(self, selected_test):
        """Load test data from CSV."""
        csv_path = f"../data/processed/{selected_test}.csv"
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            raise IOError(f"Could not load file {csv_path}: {e}")

    def derive_features(self, df, selected_battery):
        """Derive features for each row based on battery type."""
        specs = self.BATTERY_SPECS[selected_battery]
        battery_type_idx = self.BATTERY_TYPE_MAPPING[selected_battery]
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
                "type": battery_type_idx,  # Label-encoded battery type
                "Current": current,
                "Voltage": voltage,
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

        results_df = pd.DataFrame(results)
        return results_df

    def predict_pipeline(self, data=None, selected_battery=None, selected_test=None):
        """
        Run the full prediction pipeline.
        If data is None, selects battery and test, loads data, and derives features.
        If data is provided, uses it directly (must have required columns).
        """
        if data is None:
            if selected_battery is None or selected_test is None:
                selected_battery, selected_test = self.select_battery_and_test()
            df = self.load_test_data(selected_test)
            data = self.derive_features(df, selected_battery)
        
        # Ensure 'type' is label-encoded and all columns are in the correct case and order
        # (Already handled in derive_features)
        # If you need to rename, do it here:
        # data = data.rename(columns={'battery_type': 'type', 'current': 'Current', 'voltage': 'Voltage'})
        # But in this version, derive_features already uses the correct names

        # Predict with model1
        Y_pred = self.predict_model1(data)
        # Predict with model2
        predicted_TOD, predicted_CDC = self.predict_model2(Y_pred, data)
        return predicted_TOD, predicted_CDC, data

if __name__ == "__main__":
    predictor = Prediction()
    predicted_TOD, predicted_CDC, data = predictor.predict_pipeline()
    print("\nPredicted TOD:", predicted_TOD)
    print("Predicted CDC:", predicted_CDC)
    print("\nProcessed data (first row):")
    print(data.head(1))
