import pandas as pd
import numpy as np

def load_profile(path="data/Profile.xlsx"):
    """
    Read time, current, and voltage from Profile.xlsx
    Args:
        path (str): Defaults to "data/Profile.xlsx".
    """
    df = pd.read_excel(path) # Read the Excel file
    print("Columns found in Excel file")
    print(df.columns)

    # Extract columns by Position
    time_col = df["Time"]
    time_s = pd.to_timedelta(time_col).dt.total_seconds().to_numpy(dtype = float)
    current_a = df["Current(A)"].to_numpy(dtype = float)
    voltage_v = df["Voltage(V)"].to_numpy(dtype = float)

    #Estimate dt
    dt = np.mean(np.diff(time_s))
    print(f"\nEstimated dt: {dt: .4f} s")
    return time_s, voltage_v, current_a, df

if __name__ == "__main__":
    time_s, voltage_v, current_a, df = load_profile()
    print("\nFirst 5 rows:")
    print(df.head())
    