import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter threading issue
import matplotlib.pyplot as plt
import threading


class preprocessing():
    def __init__(self):
        Trainer = training()
        print("Initiating Preprocessing... 1\n")
        self.preprocessing1()
        print("Training Random Forest Model 1... \n")
        Trainer.RandomForest1()
        print("Initiating Preprocessing... 2\n")
        self.preprocessing2()
        print("Training Random Forest Model 2... \n")
        Trainer.RandomForest2()

    def preprocessing1(self):
        data_dir = "../data/processed/"
        listdir = os.listdir(data_dir)

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

        DATASET = []

        for i in listdir:
            for j in BATTERY_METADATA.keys():
                if i == j + '.csv':
                    dataset = pd.read_csv(os.path.join(data_dir, i))
                    dataset['type'] = BATTERY_METADATA[j]['type']
                    dataset['capacity'] = BATTERY_METADATA[j]['capacity']
                    dataset['charged'] = BATTERY_METADATA[j]['charged']
                    DATASET.append(dataset)

        final_df = pd.concat(DATASET, ignore_index=True)
        final_df.to_csv("../new_code/DATASET.csv", index=False)

    def preprocessing2(self):
        data = pd.read_csv("../new_code/DATASET.csv")
        model = joblib.load("../models/battery_random_forest_model1.joblib")

        # Thread-safe predictions
        predictions = [None] * len(data)
        lock = threading.Lock()

        def predictor(start_idx, end_idx):
            for index in range(start_idx, end_idx):
                x = data.drop(columns=['Time to Depletion']).iloc[index:index+1]
                pred = model.predict(x)[0]
                with lock:
                    predictions[index] = pred

        threads = []
        chunk_size = 1000
        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            t = threading.Thread(target=predictor, args=(start, end))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        data['prediction'] = predictions
        data.to_csv("../new_code/DATASET2.csv", index=False)


class training():
    def RandomForest1(self):
        TARGET_VARIABLE = 'Time to Depletion'
        data = pd.read_csv("../new_code/DATASET.csv")

        Y = data[TARGET_VARIABLE]
        X = data.drop(columns=[TARGET_VARIABLE])

        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print("Numerical Features:", numerical_features)
        print("Categorical Features:", categorical_features)

        for column in data.columns:
            if data[column].isna().any():
                print(f"NaNs in {column}:")
                print(data[data[column].isna()].index)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        rf_model = RandomForestRegressor(
            random_state=42,
            n_estimators=150,
            max_depth=14,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            criterion='absolute_error',
            n_jobs=-1
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])

        print("Training Random Forest Model 1...")
        model = pipeline.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.2f}")

        joblib.dump(model, "../models/battery_random_forest_model1.joblib")

    def RandomForest2(self):
        TARGET_VARIABLE = 'Time to Depletion'
        data = pd.read_csv("../new_code/DATASET2.csv")

        Y = data[TARGET_VARIABLE]
        X = data.drop(columns=[TARGET_VARIABLE])

        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print("Numerical Features:", numerical_features)
        print("Categorical Features:", categorical_features)

        for column in data.columns:
            if data[column].isna().any():
                print(f"NaNs in {column}:")
                print(data[data[column].isna()].index)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        rf_model = RandomForestRegressor(
            random_state=42,
            n_estimators=100,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            criterion='absolute_error',
            n_jobs=-1
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])

        print("Training Random Forest Model 2...")
        model = pipeline.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.2f}")

        joblib.dump(model, "../models/battery_random_forest_model2.joblib")


if __name__ == "__main__":
    preprocessing()
