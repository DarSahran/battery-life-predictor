import os
import pandas as pd
import matplotlib.pyplot as plt
import Predict

# IMPORTANT: Set the backend before importing pyplot or creating any plots
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
BATTERY_TYPE_MAPPING = {'tn1': 0, 'b1': 1, 'b2': 2, 'b3': 3, 'b5': 4}
SMOOTH_WINDOW = 10
OUTPUT_DIR = "./predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Predictor = Predict.prediction()

def apply_smoothing(x, window=SMOOTH_WINDOW):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values

def run_simulation(csv_path, output_dir=OUTPUT_DIR):
    try:
        df = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        key = filename.replace('.csv', '')
        meta = Predictor.metadata.get(key)
        if not meta:
            print(f"No metadata found for {filename}")
            return
        type_code = BATTERY_TYPE_MAPPING.get(meta['type'].lower())
        if type_code is None:
            print(f"Unknown battery type for {filename}")
            return

        print(f"\nProcessing: {filename} (Type: {meta['type']}, Charged: {meta['charged']}Ah)")

        dod_vals, tod_vals, capacity_ah, step_indices = [], [], [], []

        for idx, row in df.iterrows():
            features = pd.DataFrame({
                'Current': [row['Current']],
                'Voltage': [row['Voltage']],
                'Ah Out': [row['Ah Out']],
                'Power': [row['Power']],
                'Remaining Capacity': [row['Remaining Capacity']],
                'type': [type_code],
                'capacity': [meta['capacity']],
                'charged': [meta['charged']],
                'discharge_rate': [row['Current'] / (row['Voltage'] + 1e-6)],
                'discharge_ratio': [row['Ah Out'] / (meta['charged'] + 1e-6)]
            })

            pred = Predictor.model1.predict(features)
            if hasattr(pred, 'shape') and pred.shape[1] >= 2:
                predicted_TOD = pred[0][0]
            else:
                predicted_TOD = pred[0]

            tod_hr = predicted_TOD / 3600
            tod_vals.append(tod_hr)
            capacity_ah.append(row['Remaining Capacity'])
            dod = 100 * (meta['charged'] - row['Remaining Capacity']) / meta['charged']
            dod_vals.append(dod)
            step_indices.append(idx / 60)

        # Smoothing
        smooth_dod_vals = apply_smoothing(dod_vals)
        smooth_tod_vals = apply_smoothing(tod_vals)
        smooth_capacity_ah = apply_smoothing(capacity_ah)

        # Save results
        results_df = pd.DataFrame({
            "Step (hr)": step_indices,
            "Remaining Capacity (Ah)": capacity_ah,
            "Depth of Discharge (%)": dod_vals,
            "Predicted Time Remaining (hr)": tod_vals,
        })

        output_path = os.path.join(output_dir, f"{key}_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

        # Plot 1: DoD (%) vs Time (hours)
        plt.figure(figsize=(8, 4))
        plt.plot(step_indices, dod_vals, color='blue', alpha=0.3, label="Raw")
        plt.plot(step_indices, smooth_dod_vals, color='blue', linewidth=2, label="Smoothed")
        plt.xlabel("Time (hours)")
        plt.ylabel("Depth of Discharge (%)")
        plt.ylim(0, 100)
        plt.title(f"DoD vs Time - {meta['type'].upper()} (Charged: {meta['charged']}Ah)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_dod.png"))
        plt.close()

        # Plot 2: Remaining Capacity (Ah) vs Predicted Time Remaining (hours)
        plt.figure(figsize=(8, 4))
        plt.plot(smooth_tod_vals, smooth_capacity_ah, marker='o', color='green')
        for i, (x, y) in enumerate(zip(smooth_tod_vals, smooth_capacity_ah)):
            if i % 10 == 0 or i == len(smooth_tod_vals)-1:
                plt.annotate(f"{y:.1f}Ah", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        plt.xlabel("Predicted Time Remaining (Hours)")
        plt.ylabel("Remaining Capacity (Ah)")
        plt.title(f"Remaining Capacity vs Time Remaining - {meta['type'].upper()} (Charged: {meta['charged']}Ah)")
        plt.ylim(0, meta['charged'])
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_ah.png"))
        plt.close()

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def main():
    folder_path = "../data/processed"
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process")
    for csv_path in csv_files:
        run_simulation(csv_path)

if __name__ == "__main__":
    main()
