import joblib
import os

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
        pred_TOD = Y_pred[0]
        pred_CDC = Y_pred[1]
        data['predicted TOD'] = pred_TOD
        data['predicted CDC'] = pred_CDC

        final_output = self.model2.predict(data)
