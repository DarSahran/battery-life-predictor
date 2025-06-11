# 🔋 Battery Life Prediction System

Predict remaining battery life (Time to Depletion, TOD) and Depth of Discharge (DoD) for various battery types using machine learning. This project provides real-time simulation, batch processing, and EV-style visualizations for battery discharge cycles.

---

## 🚀 Features

- **Real-time prediction** of battery remaining time and DoD using Random Forest models
- **Interactive Streamlit UI** for simulation, visualization, and speed control
- **Batch processing** of all battery test files with automatic result saving
- **EV-style plots**: DoD vs Time, Capacity vs Predicted Time Remaining
- **Automatic feature engineering** and metadata integration
- **Ready-to-use for new battery types and test files

---


**Notes:**

- All code is in the project root.
- Input data is in `data/processed/`.
- Model files are in `models/`.
- All outputs (predictions and plots) are saved in `predictions/`.
- The `.venv/` folder contains your Python environment and can be ignored for code/data purposes.


## ⚙️ Setup

1. **Clone this repository**

```bash
git clone https://github.com/DarSahran/battery-life-predictor.git
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Ensure processed battery CSVs are in `data/processed/`**
Each file should be named like `TEST_X_processed.csv`.
4. **Ensure trained models are in `models/`**
    - `battery_random_forest_model1.joblib`
    - `battery_random_forest_model2.joblib`

---

## 🖥️ Usage

### **A. Real-Time Simulation (Streamlit UI)**

```bash
streamlit run battery_gauge_app.py
```

- Select a test file and simulation speed.
- Click **Start Simulation** to see live DoD and capacity plots.
- Download results after simulation.


### **B. Batch Processing**

```bash
python batch_simulation.py
```

- Runs simulation for all test files in `data/processed/`.
- Saves results and plots to `predictions/`.

---

## 📊 Output Explanation

For each battery test, the following files are generated in `predictions/`:

- `TEST_X_processed_dod.png`
*Depth of Discharge (%) vs Time (hours) plot.*
- `TEST_X_processed_ah.png`
*Remaining Capacity (Ah) vs Predicted Time Remaining (hours) plot.*
- `TEST_X_processed_predictions.csv`
*Detailed step-by-step predictions (time, DoD, remaining capacity, predicted TOD, etc).*

---

## 🧑‍💻 File Descriptions

| File | Purpose |
| :-- | :-- |
| `Predict.py` | Core prediction class (feature engineering, models) |
| `battery_gauge_app.py` | Streamlit app for interactive simulation |
| `batch_simulation.py` | Batch simulation and output saving |
| `process_battery_sheets.py` | Raw data cleaning (optional) |
| `Preprocessing-and-Training-PT2.py` | Feature engineering, dataset creation |
| `Random-Forest1.ipynb` | Model 1 training notebook |
| `Random-Forest2.ipynb` | Model 2 training notebook |
| `data/processed/` | Input processed battery CSVs |
| `models/` | Trained model files |
| `predictions/` | Output CSVs and plots |


---

## 🛠️ Troubleshooting

- **ValueError: could not convert string to float: 'b1'**
→ Ensure battery type is encoded as an integer before prediction.
- **RuntimeError: main thread is not in main loop**
→ When running batch scripts, make sure `matplotlib` backend is set to `'Agg'`.
- **Unrealistic predictions (e.g., 119hr for a 5hr battery)**
→ Check that features are calculated cumulatively and only use data available up to the current step.

---

## 📈 Example Results

For each battery, you will get:

- **DoD vs Time (hours):**
- **Remaining Capacity (Ah) vs Predicted Time Remaining (hours):**

---
