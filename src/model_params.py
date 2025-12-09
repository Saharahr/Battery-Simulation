"""
Utilities for saving and loading battery model parameters.
We store ECM + OCV parameters as a small dictionary in a .npy file.
"""
import numpy as np

def save_combined_params(
    path,
    R0, R1, C1, R2, C2,
    ocv_coeffs,
    Q_Ah,
    soc0,
):
    """
    Save combined ECM + OCV parameters to a .npy file.
    """
    pack = {
        "R0": float(R0),
        "R1": float(R1),
        "C1": float(C1),
        "R2": float(R2),
        "C2": float(C2),
        "ocv_coeffs": np.array(ocv_coeffs, dtype=float),
        "Q_Ah": float(Q_Ah),
        "soc0": float(soc0),
    }
    np.save(path, pack)
    print(f"[INFO] Saved combined parameters → {path}")


def load_combined_params(path="data/combined_params.npy"):
    """
    Load combined ECM + OCV parameters from a .npy file.

    Returns
    -------
    pack : dict
        {
          "R0", "R1", "C1", "R2", "C2",
          "ocv_coeffs", "Q_Ah", "soc0"
        }
    """
    pack = np.load(path, allow_pickle=True).item()
    print(f"[INFO] Loaded combined parameters from {path}")
    return pack