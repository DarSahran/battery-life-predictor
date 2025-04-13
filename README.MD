# 🔋 Battery Life Predictor

A real-time, deep learning-based simulation and prediction tool for estimating battery discharge life. It uses LSTM neural networks trained on per-minute discharge data and integrates a Streamlit-based web interface for live simulations.

---

## 📂 Project Structure

```
BATTEY_GAUGE/
├── data/
│   ├── excel/                      # Raw test data in Excel format
│   │   ├── battery_info.xlsx
│   │   └── Temp_Rev_Tubular26_Tests.xlsx
│   ├── Merged/
│   │   └── battery_training_data.csv
│   └── processed/                 # Cleaned CSVs per battery test (TEST_1_processed.csv ... TEST_17_processed.csv)
├── models/                        # Trained model and scalers
│   ├── battery_lstm_model.keras
│   ├── input_scaler.pkl
│   └── target_scaler.pkl
├── scripts/                       # Model training and prediction scripts
│   ├── train_lstm_model.py
│   ├── predict_battery_time.py
│   └── process_battery_sheets.py
├── webpage/                       # Streamlit app
│   └── battery_life_gui.py
├── requirements.txt
└── .gitignore
```

---

## 🚀 Features

- **LSTM-Based Deep Learning Model** trained on realistic battery usage data.
- **Per-test Metadata Integration** (Capacity, Battery Type, Charge Level).
- **Exponential Smoothing** applied to Current, Voltage, and derived signals.
- **Real-time Simulation** using Streamlit with visual prediction updates.
- **Battery Type Selector** (b1, b2, b3, b5, tn1).
- **Live Plotting** of Remaining Hours in GUI.

---

## ⚙️ Setup Instructions

1. **Clone the Repo**
   ```bash
   git clone https://github.com/DarSahran/battery-life-predictor.git
   cd battery-life-predictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python scripts/train_lstm_model.py
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run webpage/battery_life_gui.py
   ```

---

## 📊 Model Details

- **Input Features**: `Current`, `Voltage`, `Ah Out`, `Cumulative Discharge`, `Power`, `Remaining Capacity`, `Battery Type`, `Charged Upto`, `Capacity`
- **Target**: `Discharge %`
- **Time Series Window**: 30 minutes
- **Model Type**: LSTM (128 + 64 units) + Dense layers
- **Output**: Estimated Remaining Time (discharge % ➔ time in HH:MM:SS)

---

## 💡 Battery Types & Metadata

| Battery Type | Capacity (Ah) | Charged Upto (Ah) |
|--------------|----------------|-------------------|
| B1           | 81.28          | 81.28             |
| B2           | 85 - 88.81     | 85 - 88.81        |
| B3           | 88.35          | 61 - 88.35        |
| B5           | 85             | 67 - 85           |
| TN1          | 85             | 54 - 80           |

---

## 💮 License

This project is licensed under the MIT License.

---

## 👤 Author

**Dar Sahran**  
GitHub: [@DarSahran](https://github.com/DarSahran)
