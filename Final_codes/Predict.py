import joblib
import pandas as pd


class prediction():
    def __init__(self):
        self.model1_path = "../Final_codes/battery_random_forest_model1.joblib"
        self.model2_path = "../Final_codes/battery_random_forest_model2.joblib"
        self.model1 = joblib.load(self.model1_path)
        self.model2 = joblib.load(self.model2_path)

        self.metadata = {
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

    def predict_model1(self, data):
        Y_pred = self.model1.predict(data)
        self.predict_model2(Y_pred, data)

    def predict_model2(self, Y_pred, data):
        # Battery type encoding mapping
        type_encoding = {
            'b1': 0, 'b2': 1, 'b3': 2, 'b5': 3, 'tn1': 4
        }

        pred_TOD = Y_pred[0]
        pred_CDC = Y_pred[1]
        data['predicted TOD'] = pred_TOD
        data['predicted CDC'] = pred_CDC

        # Add encoded battery type
        if 'type' in data.columns:
            data['type'] = data['type'].map(type_encoding)

        final_output = self.model2.predict(data)
        return final_output

    def feature_derivator(self, voltage, current, battery_type, ini_charge):
        # Initialize variables
        time_step = 1  # assuming 1 second intervals
        ah_out = []
        power = []
        remaining_capacity = []
        discharge_rate = []
        discharge_ratio = []

        # Get battery capacity from metadata based on type
        battery_capacity = 0
        for test in self.metadata.values():
            if test['type'] == battery_type:
                battery_capacity = test['capacity']
                break

        # Calculate cumulative values
        current_ah = 0

        # Calculate Ah out (cumulative)
        current_ah += (abs(current) * time_step) / 3600
        ah_out = current_ah

        # Calculate power
        power = voltage * current

        # Calculate remaining capacity
        remaining_capacity = battery_capacity - current_ah

        # Calculate discharge rate
        discharge_rate = abs(current) / battery_capacity

        # Calculate discharge ratio
        discharge_ratio = current_ah / battery_capacity

        # Create DataFrame
        df = pd.DataFrame({
            'Voltage': float(voltage),
            'Current': float(current),
            'Ah_out': float(ah_out),
            'Power': float(power),
            'Remaining_Capacity': float(remaining_capacity),
            'type': battery_type,
            'capacity': float(battery_capacity),
            'charged': float(ini_charge),
            'Discharge_Rate': float(discharge_rate),
            'Discharge_Ratio': float(discharge_ratio)
        })

        return self.predict_model1(df)

