import pandas as pd
import os

# Manual metadata: capacity and charged upto (Ah)
battery_info = {
    "test1":   {"capacity": 85, "charged_upto": "full"},
    "test2":   {"capacity": 81.28, "charged_upto": "full"},
    "test3":   {"capacity": 85, "charged_upto": "full"},
    "test4":   {"capacity": 85, "charged_upto": "full"},
    "test5":   {"capacity": 88.81, "charged_upto": "full"},
    "test6":   {"capacity": 81.84, "charged_upto": "full"},
    "test7":   {"capacity": 81.84, "charged_upto": 36},
    "test8":   {"capacity": 88.81, "charged_upto": 27},
    "test9":   {"capacity": 85, "charged_upto": 80},
    "test10":  {"capacity": 85, "charged_upto": 54},
    "test11":  {"capacity": 85, "charged_upto": "full"},
    "test12":  {"capacity": 85, "charged_upto": 67},
    "test13":  {"capacity": 85, "charged_upto": "full"},
    "test14":  {"capacity": 88.83, "charged_upto": 52},
    "test15":  {"capacity": 88.35, "charged_upto": 70},
    "test16":  {"capacity": 88.35, "charged_upto": 61},
    "test17":  {"capacity": 88.35, "charged_upto": "full"}
}

# Paths
FILE_PATH = r"E:\Battey_Gauge\data\Temp_Rev_Tubular26_Tests.xlsx"
OUTPUT_DIR = r"E:\Battey_Gauge\processed_sheets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Excel
excel_file = pd.ExcelFile(FILE_PATH)
selected_sheets = [f"TEST {i}" for i in range(1, 18) if f"TEST {i}" in excel_file.sheet_names]

# Robust column finder
def find_column(columns, keyword, alternatives=None):
    keyword = keyword.lower()
    alternatives = alternatives or []
    for col in columns:
        col_clean = col.lower().strip()
        if keyword in col_clean or any(alt in col_clean for alt in alternatives):
            return col
    raise ValueError(f"Missing column with keyword '{keyword}' or alternatives {alternatives}")

# Process each sheet
for sheet in selected_sheets:
    try:
        sheet_key = sheet.lower().replace(" ", "")
        info = battery_info[sheet_key]
        capacity = info["capacity"]
        charged_ah = capacity if info["charged_upto"] == "full" else float(info["charged_upto"])

        # Read sheet (header in second row)
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, skiprows=1)
        df.columns = df.columns.str.strip()

        # Match columns
        current_col = find_column(df.columns, 'current')
        voltage_col = find_column(df.columns, 'volt')
        ah_out_col = find_column(df.columns, 'ah out')
        cum_discharge_col = find_column(df.columns, 'cumulative', ['cummulative', 'cum'])

        # Select and rename
        selected_df = df[[current_col, voltage_col, ah_out_col, cum_discharge_col]].copy()
        selected_df.columns = ['Current', 'Voltage', 'Ah Out', 'Cumulative Actual Disch Ah']

        # Convert to numeric (fixes TypeError)
        for col in selected_df.columns:
            selected_df[col] = pd.to_numeric(selected_df[col], errors='coerce')

        # Derived features
        selected_df['Power'] = selected_df['Current'] * selected_df['Voltage']
        selected_df['Remaining Capacity'] = charged_ah - selected_df['Cumulative Actual Disch Ah']
        selected_df['Time to Depletion'] = (selected_df['Remaining Capacity'] * 3600) / selected_df['Current']
        selected_df = selected_df.dropna()

        # Export
        output_file = os.path.join(OUTPUT_DIR, f"{sheet.replace(' ', '_')}_processed.csv")
        selected_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file} | Battery: {capacity}Ah | Charged up to: {charged_ah}Ah")

    except Exception as e:
        print(f"⚠️ Skipped {sheet}: {type(e).__name__} - {e}")
