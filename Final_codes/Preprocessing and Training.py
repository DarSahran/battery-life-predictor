import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt
import threading

class preprocessing():
    def __init__(self):
        print("Starting preprocessing workflow...")
        self.dataset1 = self._preprocessing1()

        if self.dataset1.empty:
            print("Halting workflow as initial dataset is empty.")
            return

        trainer = training()
        
        self.model1_path = "../models/battery_random_forest_model1.joblib"
        print(f"\nTraining RandomForest1 and saving to {self.model1_path}...")
        self.model1 = trainer.RandomForest1(self.dataset1.copy(), self.model1_path) # Use .copy() if dataset1 is reused elsewhere unmodified

        # Path for the model to be used in _preprocessing2 for initial predictions
        # This was "../models/battery_random_forest_model.joblib" in the original code.
        # If you intend to use the model just trained (model1), pass self.model1 directly
        # or load it using self.model1_path.
        # For this refactoring, I'm keeping it as a distinct path as in the original.
        initial_prediction_model_path = "../models/battery_random_forest_model.joblib"
        print(f"\nRunning preprocessing step 2 using model from {initial_prediction_model_path}...")
        if self.model1 is not None: # Proceed only if first model training was successful
            # If initial_prediction_model_path refers to the model trained by RandomForest1,
            # you should use self.model1 or self.model1_path here.
            # Example: self.dataset2 = self._preprocessing2(self.dataset1.copy(), model_to_use=self.model1)
            # Or: self.dataset2 = self._preprocessing2(self.dataset1.copy(), model_path_to_load=self.model1_path)
            self.dataset2 = self._preprocessing2(self.dataset1.copy(), model_path_to_load=initial_prediction_model_path)
        else:
            print("Skipping preprocessing step 2 and RandomForest2 due to failure in RandomForest1.")
            self.dataset2 = None
            self.model2 = None

        if self.dataset2 is not None and not self.dataset2.empty:
            self.model2_path = "../models/battery_random_forest_model2.joblib"
            print(f"\nTraining RandomForest2 and saving to {self.model2_path}...")
            self.model2 = trainer.RandomForest2(self.dataset2.copy(), self.model2_path)
        else:
            print("Skipping RandomForest2 as dataset2 is not available.")
            self.model2 = None

        # Optionally save the final DataFrames
        if not self.dataset1.empty:
            self.save_dataframe(self.dataset1, "../new_code/DATASET_optimized.csv")
        if self.dataset2 is not None and not self.dataset2.empty:
            self.save_dataframe(self.dataset2, "../new_code/DATASET2_optimized.csv")
        
        print("\nPreprocessing and training workflow finished.")

    def save_dataframe(self, df, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Saved DataFrame to {path}")
        except Exception as e:
            print(f"Error saving DataFrame to {path}: {e}")

    def _preprocessing1(self):
        print("Running preprocessing step 1 (data loading and merging)...")
        data_dir = "../data/processed/"
        try:
            if not os.path.exists(data_dir):
                print(f"Error: Data directory not found: {data_dir}")
                return pd.DataFrame()
            listdir = os.listdir(data_dir)
        except Exception as e:
            print(f"Error accessing data directory {data_dir}: {e}")
            return pd.DataFrame()

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
        
        DATASET = pd.DataFrame()
        
        for filename in listdir:
            processed_name = filename.replace('.csv', '')
            if processed_name in BATTERY_METADATA:
                try:
                    dataset_path = os.path.join(data_dir, filename)
                    dataset = pd.read_csv(dataset_path)
                    dataset['type'] = BATTERY_METADATA[processed_name]['type']
                    dataset['capacity'] = BATTERY_METADATA[processed_name]['capacity']
                    dataset['charged'] = BATTERY_METADATA[processed_name]['charged']
                    DATASET = pd.concat([DATASET, dataset], ignore_index=True)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue
        
        if DATASET.empty:
            print("Warning: No data loaded in _preprocessing1. Resulting DataFrame is empty.")
        else:
            print(f"Preprocessing 1 finished. Shape of DATASET: {DATASET.shape}")
        return DATASET
            
    def _preprocessing2(self, data_df, model_path_to_load=None, model_to_use=None):
        if data_df.empty:
            print("Skipping _preprocessing2 as input data is empty.")
            return pd.DataFrame()

        model = None
        if model_to_use:
            model = model_to_use
            print(f"Using provided model object for predictions.")
        elif model_path_to_load:
            try:
                model = joblib.load(model_path_to_load)
                print(f"Loaded model from {model_path_to_load} for predictions.")
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_path_to_load}. Predictions will not be run.")
                data_df['prediction'] = np.nan
                return data_df
            except Exception as e:
                print(f"Error loading model from {model_path_to_load}: {e}. Predictions will not be run.")
                data_df['prediction'] = np.nan
                return data_df
        else:
            print("Error: No model or model path provided for _preprocessing2. Predictions will not be run.")
            data_df['prediction'] = np.nan
            return data_df

        # Plotting (optional, consider moving to a separate utility function)
        try:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(data_df['Voltage'], data_df['Current'], alpha=0.5)
            plt.xlabel('Voltage'); plt.ylabel('Current'); plt.title('Voltage vs Current')
            plt.subplot(2, 2, 2)
            plt.scatter(data_df['Power'], data_df['Remaining Capacity'], alpha=0.5)
            plt.xlabel('Power'); plt.ylabel('Remaining Capacity'); plt.title('Power vs Remaining Capacity')
            plt.subplot(2, 2, 3)
            plt.scatter(data_df['Time to Depletion'], data_df['Cumulative Actual Disch Ah'], alpha=0.5)
            plt.xlabel('Time to Depletion'); plt.ylabel('Cumulative Actual Discharge'); plt.title('Depletion Time vs Cumulative Discharge')
            plt.subplot(2, 2, 4)
            plt.scatter(data_df['Ah Out'], data_df['Remaining Capacity'], alpha=0.5)
            plt.xlabel('Ah Out'); plt.ylabel('Remaining Capacity'); plt.title('Ah Out vs Remaining Capacity')
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Plotting skipped: A required column is missing: {e}")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")


        predictions_map = {} 

        def predictor_thread_func(pred_map, model_obj, data_chunk, chunk_offset):
            try:
                # The model pipeline should handle feature selection and preprocessing
                chunk_predictions = model_obj.predict(data_chunk)
                for i, prediction_val in enumerate(chunk_predictions):
                    pred_map[data_chunk.index[i]] = prediction_val # Use original DataFrame index
            except Exception as e:
                print(f"Error during batch prediction in thread: {e}. Rows {chunk_offset} to {chunk_offset + len(data_chunk)-1} may not be predicted.")
                # Fallback or mark as NaN for this chunk
                for i in range(len(data_chunk)):
                     pred_map[data_chunk.index[i]] = np.nan


        num_rows = len(data_df)
        if num_rows == 0:
            print("No data to predict on in _preprocessing2.")
            data_df['prediction'] = np.nan
            return data_df

        num_threads = 6 
        chunk_size = (num_rows + num_threads - 1) // num_threads
        threads = []

        print(f"Starting predictions with {num_threads} threads...")
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_rows)
            if start_idx >= end_idx:
                continue
            
            data_chunk = data_df.iloc[start_idx:end_idx]
            thread = threading.Thread(target=predictor_thread_func, args=(predictions_map, model, data_chunk, start_idx))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        prediction_series = pd.Series(predictions_map, name='prediction').reindex(data_df.index)
        data_df['prediction'] = prediction_series
        
        # Fill any NaNs that might result from threading errors or if some chunks failed
        if data_df['prediction'].isnull().any():
            print(f"Warning: {data_df['prediction'].isnull().sum()} predictions are NaN. Check for errors in predictor threads.")
            # data_df['prediction'].fillna(defaultValue, inplace=True) # Optional: fill with a default

        print(f"Preprocessing 2 finished. Shape of data with predictions: {data_df.shape}")
        return data_df

class training():
    def _train_model(self, X, Y, model_save_path, param_grid, model_name="Model"):
        if X.empty or Y.empty:
            print(f"Skipping training for {model_name} due to empty features or target.")
            return None

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\n--- {model_name} Training ---")
        print(f"Numerical Features: {numerical_features}")
        print(f"Categorical Features: {categorical_features}")

        # Basic NaN check (consider more sophisticated handling)
        if X.isnull().sum().sum() > 0:
            print(f"Warning: NaNs found in features for {model_name}. Imputers in pipeline will handle them.")
        if Y.isnull().any():
            print(f"Error: NaNs found in target variable for {model_name}. Please clean the target variable.")
            return None

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()) 
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Handles NaNs in categoricals
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough') # 'passthrough' columns not specified, or 'drop'
        
        rf_model = RandomForestRegressor(
            random_state=42,
            bootstrap=True,
            criterion='absolute_error', 
            n_jobs=-1,
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=2, # Using CV=2 for speed; consider 3-5 for better results
            verbose=1,
            n_jobs=-1, # Use all available cores
            return_train_score=True,
            refit=True
        )
        
        print(f"Initiating Grid Search for {model_name}...")
        try:
            grid_search.fit(X_train, Y_train)
            print(f"Grid Search Finished for {model_name}.")
            
            best_model = grid_search.best_estimator_
            Y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(Y_test, Y_pred)
            print(f"{model_name} - Mean Absolute Error on Test Set: {mae:.4f}")
            
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump(best_model, model_save_path)
            print(f"Saved {model_name} to {model_save_path}")
            
            print(f"{model_name} - Best Parameters: {grid_search.best_params_}")
            return best_model

        except Exception as e:
            print(f"Error during GridSearchCV for {model_name}: {e}")
            print("Check for issues like all-NaN columns after splits, or incompatible data types for transformers.")
            # print("X_train sample for debugging:")
            # print(X_train.head().to_string())
            # print("\nX_train dtypes:\n", X_train.dtypes)
            # print("\nX_train NaN counts:\n", X_train.isnull().sum())
            return None

    def RandomForest1(self, data_df, model_save_path):
        TARGET_VARIABLE = 'Time to Depletion'
        if data_df.empty or TARGET_VARIABLE not in data_df.columns:
            print(f"Error: Target variable '{TARGET_VARIABLE}' not in DataFrame or DataFrame is empty for RandomForest1.")
            return None
        
        Y = data_df[TARGET_VARIABLE]
        X = data_df.drop(TARGET_VARIABLE, axis=1)

        # Remove 'prediction' column if it exists from a prior step and isn't a feature for this model
        if 'prediction' in X.columns:
            X = X.drop('prediction', axis=1)
            print("Dropped 'prediction' column before training RandomForest1.")

        param_grid_rf1 = {
            'regressor__n_estimators': [100, 150], # Reduced for example speed
            'regressor__max_depth': [10, 14],
            'regressor__min_samples_split': [2, 4],
            'regressor__min_samples_leaf': [1, 3]
        }
        return self._train_model(X, Y, model_save_path, param_grid_rf1, "RandomForest1")

    def RandomForest2(self, data_df, model_save_path):
        TARGET_VARIABLE = 'Time to Depletion'
        if data_df.empty or TARGET_VARIABLE not in data_df.columns:
            print(f"Error: Target variable '{TARGET_VARIABLE}' not in DataFrame or DataFrame is empty for RandomForest2.")
            return None

        Y = data_df[TARGET_VARIABLE]
        X = data_df.drop(TARGET_VARIABLE, axis=1)

        # For RandomForest2, the 'prediction' column (output of _preprocessing2) is a feature.
        # Ensure 'prediction' column is numeric and handle NaNs if any.
        if 'prediction' in X.columns:
            X['prediction'] = pd.to_numeric(X['prediction'], errors='coerce')
            if X['prediction'].isnull().any():
                print(f"Warning: NaNs found in 'prediction' feature for RandomForest2. Will be imputed by pipeline.")
        else:
            print("Warning: 'prediction' column not found in data for RandomForest2. This model might expect it as a feature.")

        param_grid_rf2 = { # Can be same or different from RF1
            'regressor__n_estimators': [100, 180], 
            'regressor__max_depth': [12, 16],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 4]
        }
        return self._train_model(X, Y, model_save_path, param_grid_rf2, "RandomForest2")


if __name__ == '__main__':
    # This is an example of how you might run the preprocessing.
    # Ensure your data is in '../data/processed/' and the
    # '../models/' and '../new_code/' directories exist or can be created.
    
    # Create dummy directories and data for a self-contained example
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../new_code", exist_ok=True)

    # Create dummy CSV files based on BATTERY_METADATA keys
    dummy_csv_content = "Voltage,Current,Power,Remaining Capacity,Time to Depletion,Cumulative Actual Disch Ah,Ah Out\n" + \
                        "1,1,1,1,1,1,1\n" + \
                        "2,2,4,0.5,0.5,2,2"
    
    for i in range(1, 18):
        with open(f"../data/processed/TEST_{i}_processed.csv", "w") as f:
            f.write(dummy_csv_content)

    # Create a dummy pre-existing model for _preprocessing2 to load, if it doesn't use model1
    # This dummy model needs to be a valid scikit-learn pipeline model
    try:
        from sklearn.linear_model import LinearRegression
        dummy_pipeline = Pipeline([('regressor', LinearRegression())])
        # It needs to be trained on something, even dummy data, to have attributes like `predict`
        dummy_X = pd.DataFrame({'Voltage': [1,2], 'Current': [1,2], 'Power': [1,4], 
                                'Remaining Capacity': [1, .5], 'Cumulative Actual Disch Ah': [1,2], 
                                'Ah Out': [1,2], 'type': ['b5','b1'], 'capacity': [85,81], 'charged': [85,81]})
        dummy_Y = pd.Series([1,0.5])

        # A minimal preprocessor for the dummy model
        dummy_numerical_features = ['Voltage', 'Current', 'Power', 'Remaining Capacity', 'Cumulative Actual Disch Ah', 'Ah Out', 'capacity', 'charged']
        dummy_categorical_features = ['type']
        dummy_num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        dummy_cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        dummy_col_transformer = ColumnTransformer(transformers=[
            ('num', dummy_num_transformer, dummy_numerical_features),
            ('cat', dummy_cat_transformer, dummy_categorical_features)
        ], remainder='passthrough')
        
        full_dummy_pipeline_for_joblib = Pipeline(steps=[
            ('preprocessor', dummy_col_transformer),
            ('regressor', LinearRegression())
        ])
        full_dummy_pipeline_for_joblib.fit(dummy_X, dummy_Y)
        joblib.dump(full_dummy_pipeline_for_joblib, "../models/battery_random_forest_model.joblib")
        print("Created a dummy 'battery_random_forest_model.joblib' for testing.")
    except Exception as e:
        print(f"Could not create dummy model for testing: {e}")

    # Run the main preprocessing and training
    processor = preprocessing()