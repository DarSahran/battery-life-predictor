import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter threading issue
import threading
from sklearn.inspection import permutation_importance

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
        final_df.to_csv("../Final_codes/DATASET.csv", index=False)

    @staticmethod
    def add_engineered_features(data):
        data = data.copy()
        data['discharge_rate'] = data['Current'] / (data['Voltage'] + 1e-6)
        data['discharge_ratio'] = data['Ah Out'] / (data['charged'] + 1e-6)
        data['step_index'] = data.groupby('type').cumcount()
        return data

    def preprocessing2(self):
        data = pd.read_csv("DATASET.csv")
        data = self.add_engineered_features(data)
        model, feature_cols = joblib.load("../Final_codes/battery_random_forest_model1.joblib")

        predictions = [None] * len(data)
        lock = threading.Lock()

        def predictor(start_idx, end_idx):
            for index in range(start_idx, end_idx):
                x = data.iloc[index:index+1][feature_cols]
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
        data.to_csv("../Final_codes/DATASET2.csv", index=False)

class training():
    @staticmethod
    def add_engineered_features(data):
        data = data.copy()
        data['discharge_rate'] = data['Current'] / (data['Voltage'] + 1e-6)
        data['discharge_ratio'] = data['Ah Out'] / (data['charged'] + 1e-6)
        data['step_index'] = data.groupby('type').cumcount()
        return data

    def RandomForest1(self):
        TARGET_VARIABLE = 'Time to Depletion'
        data = pd.read_csv("DATASET.csv")
        data = self.add_engineered_features(data)

        numerical_features = [
            'Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah',
            'Power', 'Remaining Capacity', 'capacity', 'charged',
            'discharge_rate', 'step_index', 'discharge_ratio'
        ]
        categorical_features = ['type']
        features_for_model = [f for f in numerical_features if f in data.columns] + categorical_features

        data['Depletion Ratio'] = data['Time to Depletion'] / (data['charged'] + 1e-6)
        TARGET_VARIABLE = 'Depletion Ratio'

        Y = data[TARGET_VARIABLE]
        X = data[features_for_model]

        print("Numerical Features:", [f for f in numerical_features if f in X.columns])
        print("Categorical Features:", categorical_features)

        for column in data.columns:
            if data[column].isna().any():
                print(f"NaNs in {column}:")
                print(data[data[column].isna()].index)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), [f for f in numerical_features if f in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        rf_model = RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
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
        pipeline.fit(X_train, Y_train)

        Y_pred = pipeline.predict(X_test)
        print(f"Mean Absolute Error (Depletion Ratio): {mean_absolute_error(Y_test, Y_pred):.4f}")

        Y_test_orig = Y_test * (X_test['charged'].values + 1e-6)
        Y_pred_orig = Y_pred * (X_test['charged'].values + 1e-6)
        print(f"Mean Absolute Error (Time to Depletion, original units): {mean_absolute_error(Y_test_orig, Y_pred_orig):.2f}")

        scores = cross_val_score(
            pipeline, X, Y,
            scoring='neg_mean_absolute_error',
            cv=5, n_jobs=-1
        )
        print(f"Cross-validated MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

        # --- FIX: get feature names from preprocessor ---
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        result = permutation_importance(
            pipeline, X_test, Y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        if len(feature_names) != len(result.importances_mean):
            print(f"Warning: Length mismatch. Expected {len(feature_names)} features, got {len(result.importances_mean)} importance values.")
            # Print for debugging
            print("Feature names:", feature_names)
            print("Importances:", result.importances_mean)
        else:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
            print("Top 10 Features by Importance:")
            print(importance_df.head(10))

        # Save both model and feature columns for safe prediction
        joblib.dump((pipeline, X.columns.tolist()), "battery_random_forest_model1.joblib")

    def RandomForest2(self):
        TARGET_VARIABLE = 'Time to Depletion'
        data = pd.read_csv("DATASET2.csv")
        data = self.add_engineered_features(data)

        numerical_features = [
            'Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah',
            'Power', 'Remaining Capacity', 'capacity', 'charged',
            'discharge_rate', 'step_index', 'discharge_ratio'
        ]
        categorical_features = ['type']
        features_for_model = [f for f in numerical_features if f in data.columns] + categorical_features

        data['Depletion Ratio'] = data['Time to Depletion'] / (data['charged'] + 1e-6)
        TARGET_VARIABLE = 'Depletion Ratio'

        Y = data[TARGET_VARIABLE]
        X = data[features_for_model]

        print("Numerical Features:", [f for f in numerical_features if f in X.columns])
        print("Categorical Features:", categorical_features)

        for column in data.columns:
            if data[column].isna().any():
                print(f"NaNs in {column}:")
                print(data[data[column].isna()].index)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), [f for f in numerical_features if f in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        rf_model = RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
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
        pipeline.fit(X_train, Y_train)

        Y_pred = pipeline.predict(X_test)
        print(f"Mean Absolute Error (Depletion Ratio): {mean_absolute_error(Y_test, Y_pred):.4f}")

        Y_test_orig = Y_test * (X_test['charged'].values + 1e-6)
        Y_pred_orig = Y_pred * (X_test['charged'].values + 1e-6)
        print(f"Mean Absolute Error (Time to Depletion, original units): {mean_absolute_error(Y_test_orig, Y_pred_orig):.2f}")

        scores = cross_val_score(
            pipeline, X, Y,
            scoring='neg_mean_absolute_error',
            cv=5, n_jobs=-1
        )
        print(f"Cross-validated MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        result = permutation_importance(
            pipeline, X_test, Y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        if len(feature_names) != len(result.importances_mean):
            print(f"Warning: Length mismatch. Expected {len(feature_names)} features, got {len(result.importances_mean)} importance values.")
            print("Feature names:", feature_names)
            print("Importances:", result.importances_mean)
        else:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
            print("Top 10 Features by Importance:")
            print(importance_df.head(10))

        joblib.dump((pipeline, X.columns.tolist()), "../Final_codes/battery_random_forest_model2.joblib")

if __name__ == "__main__":
    preprocessing()
