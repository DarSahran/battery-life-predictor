import pandas as pd
import numpy as np
import glob
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import re

# ------------------ CONFIG ------------------ #
DATA_DIR = "../data/processed/"
MODEL_SAVE_PATH = "../models/battery_lstm_model.keras"
SCALER_SAVE_PATH_X = "../models/input_scaler.pkl"
SCALER_SAVE_PATH_Y = "../models/target_scaler.pkl"
TIME_STEPS = 30
EPOCHS = 25
BATCH_SIZE = 32

# ------------------ METADATA ------------------ #
BATTERY_METADATA = {
    "test1":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "test2":  {"capacity": 81.28, "charged": 81.28, "type": "b1"},
    "test3":  {"capacity": 85,    "charged": 85,    "type": "b5"},
    "test4":  {"capacity": 85,    "charged": 85,    "type": "b2"},
    "test5":  {"capacity": 88.81, "charged": 88.81, "type": "b2"},
    "test6":  {"capacity": 81.84, "charged": 81.84, "type": "b1"},
    "test7":  {"capacity": 81.84, "charged": 36,    "type": "b1"},
    "test8":  {"capacity": 88.81, "charged": 27,    "type": "b2"},
    "test9":  {"capacity": 85,    "charged": 80,    "type": "tn1"},
    "test10": {"capacity": 85,    "charged": 54,    "type": "tn1"},
    "test11": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "test12": {"capacity": 85,    "charged": 67,    "type": "b5"},
    "test13": {"capacity": 85,    "charged": 85,    "type": "b5"},
    "test14": {"capacity": 88.83, "charged": 52,    "type": "b3"},
    "test15": {"capacity": 88.35, "charged": 70,    "type": "b3"},
    "test16": {"capacity": 88.35, "charged": 61,    "type": "b3"},
    "test17": {"capacity": 88.35, "charged": 88.35, "type": "b3"},
}

# ------------------ LOAD DATA ------------------ #
def load_data(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_rows = []
    types = []

    for file in csv_files:
        basename = os.path.basename(file).lower()
        match = re.search(r"test_(\d+)", basename)
        if not match:
            print(f"[⚠️] Skipping file (invalid name): {basename}")
            continue
        sheet_name = f"test{int(match.group(1))}"

        meta = BATTERY_METADATA.get(sheet_name)
        if not meta:
            print(f"[⚠️] Metadata missing for: {sheet_name}")
            continue

        df = pd.read_csv(file)
        df = df[df['Voltage'] >= 9.3]
        if df.shape[0] <= TIME_STEPS:
            print(f"[⚠️] Skipped {sheet_name}: only {df.shape[0]} rows")
            continue

        df = df.reset_index(drop=True)
        df['Discharge %'] = 100 * (1 - df.index / df.shape[0])
        df['Capacity'] = meta['capacity']
        df['Charged_Upto'] = meta['charged']
        df['Battery_Type'] = meta['type']

        all_rows.append(df)
        types.append(meta['type'])

    if not all_rows:
        raise ValueError("🚨 No valid test sheets found! Check file names, voltage filters, or row counts.")

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all, types

# ------------------ PREPROCESS ------------------ #
def preprocess(df, battery_types, features, target, time_steps=30):
    df = df[df[target] > 0].copy()

    for col in features:
        df[col] = df[col].ewm(alpha=0.3).mean()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    battery_type_encoded = ohe.fit_transform(df[['Battery_Type']])
    feature_names = ohe.get_feature_names_out(['Battery_Type'])

    df_encoded = pd.concat([
    df.reset_index(drop=True),
    pd.DataFrame(battery_type_encoded, columns=feature_names)
    ], axis=1)

    extended_features = features + ['Capacity', 'Charged_Upto'] + list(feature_names)


    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df_encoded[extended_features])

    y = df[target].values
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - time_steps):
        X_seq.append(X_scaled[i:i + time_steps])
        y_seq.append(y_scaled[i + time_steps])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    is_finite = np.isfinite(X_seq).all(axis=(1, 2)) & np.isfinite(y_seq).flatten()
    return X_seq[is_finite], y_seq[is_finite], scaler_x, scaler_y

# ------------------ MODEL ------------------ #
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ------------------ PLOT ------------------ #
def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Training Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ MAIN ------------------ #
def main():
    print("[INFO] Loading data...")
    df, types = load_data(DATA_DIR)

    features = ['Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah', 'Power', 'Remaining Capacity']
    target = 'Discharge %'

    print("[INFO] Preprocessing...")
    X, y, scaler_x, scaler_y = preprocess(df, types, features, target, TIME_STEPS)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    model = build_lstm_model((X.shape[1], X.shape[2]))
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    print("[INFO] Training model...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[lr_schedule]
    )

    print("[INFO] Saving model and scalers...")
    os.makedirs("../models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler_x, SCALER_SAVE_PATH_X)
    joblib.dump(scaler_y, SCALER_SAVE_PATH_Y)

    print(f"[✅] Model saved to {MODEL_SAVE_PATH}")
    print(f"[✅] Input scaler saved to {SCALER_SAVE_PATH_X}")
    print(f"[✅] Target scaler saved to {SCALER_SAVE_PATH_Y}")

    plot_training(history)

if __name__ == "__main__":
    main()