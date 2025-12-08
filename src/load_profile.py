import pandas as pd
import numpy as np
from pathlib import Path

def load_profile(path="data/Profile.xlsx"):
    """
    Standard profile loader: expects columns
    'Time', 'Current(A)', 'Voltage(V)'.
    """
    df = pd.read_excel(path)
    print("\n[INFO] Loading profile →", path)
    print("Columns:", list(df.columns))

    time_col  = df["Time"]
    time_s    = pd.to_timedelta(time_col).dt.total_seconds().to_numpy(dtype=float)
    current_a = df["Current(A)"].to_numpy(dtype=float)
    voltage_v = df["Voltage(V)"].to_numpy(dtype=float)

    dt = np.mean(np.diff(time_s))
    print(f"Estimated dt: {dt:.4f} s")

    return time_s, voltage_v, current_a, df


def load_profile_ocvsoc(path="data/ocv_test_SP20.xlsx"):
    """
    Load low-current OCV/SOC test data from ocv_test_SP20.xlsx.

    Columns in the Excel file:
        'Duration (sec)'  -> time in seconds
        'mV'              -> voltage in millivolts
        'mA'              -> current in milliamps
    """
    path = Path(path)
    print(f"\n[INFO] Loading OCVSOC dataset → {path}")

    df = pd.read_excel(path)
    print("[INFO] Columns:", list(df.columns))

    time_s    = df["Duration (sec)"].to_numpy(dtype=float)
    voltage_v = df["mV"].to_numpy(dtype=float) / 1000.0      # mV → V
    current_raw = df["mA"].to_numpy(dtype=float) / 1000.0    # mA → A

    # ---- SIGN NORMALIZATION BLOCK ----
    # We want discharge < 0, charge > 0.
    # Assume most non-zero current in this file is discharge.
    nonzero = current_raw[np.abs(current_raw) > 1e-6]
    mean_current = np.mean(nonzero) if nonzero.size > 0 else 0.0

    if mean_current > 0:
        # File uses positive for discharge → flip the whole series.
        current_a = -current_raw
        print("[INFO] Flipped current sign so discharge is NEGATIVE.")
    else:
        # File already uses negative for discharge.
        current_a = current_raw
        print("[INFO] Kept current sign (discharge already negative).")
    # ----------------------------------

    if len(time_s) > 1:
        dt = np.mean(np.diff(time_s))
        print(f"[INFO] Estimated dt: {dt:.4f} s")

    return time_s, voltage_v, current_a, df

if __name__ == "__main__":
    time_s, voltage_v, current_a, df = load_profile()
    print("\nFirst 5 rows:")
    print(df.head())
    