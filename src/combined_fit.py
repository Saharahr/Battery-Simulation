"""
Unified identification of:
 - 2nd-order ECM parameters (R0, R1, C1, R2, C2)
 - OCV-SOC polynomial coefficients (poly0...poly8)
 - Effective capacity Q_Ah
 - Initial SOC soc0
using a single dataset and nonlinear least-squares (scipy.optimize.least_squares).

By default, this uses the low-current OCV test dataset:
 data/ocv_test_SP20.xlsx (Assumes discharge/rest profile, SOC from 1.0 down)

Note: This approach fits the OCV curve dynamically during the optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from load_profile import load_profile_ocvsoc   # or load_profile
from soc_update import compute_soc
from model_params import save_combined_params

POLY_ORDER = 8
N_POLY = POLY_ORDER + 1
# ECM simulation that takes OCV(t) as explicit input
def simulate_ecm_2nd_order_ocv(time_s, current_a, ocv_v,
                               R0, R1, C1, R2, C2):
    """
    2nd-order Thevenin ECM simulation with explicit OCV(t) input.
    States:
        v1: voltage across RC branch 1
        v2: voltage across RC branch 2
    Equations (continuous time):
        dv1/dt = -v1 / (R1*C1) + I / C1
        dv2/dt = -v2 / (R2*C2) + I / C2
        Vterm  = OCV + I*R0 + v1 + v2
    Discretization: forward Euler on a non-uniform time grid.
    """
    time_s = np.asarray(time_s, dtype=float)
    I = np.asarray(current_a, dtype=float)
    ocv_v = np.asarray(ocv_v, dtype=float)
    n = len(time_s)
    v1 = np.zeros(n)
    v2 = np.zeros(n)
    v_term = np.zeros(n)
    v_term[0] = ocv_v[0] + I[0] * R0  # initial terminal voltage

    for k in range(1, n):
        dt = time_s[k] - time_s[k - 1]
        if dt <= 0:
            dt = 1e-6  # avoid zero dt

        # state update (Euler)
        v1[k] = v1[k - 1] + dt * (-v1[k - 1] / (R1 * C1) + I[k - 1] / C1)
        v2[k] = v2[k - 1] + dt * (-v2[k - 1] / (R2 * C2) + I[k - 1] / C2)

        # terminal voltage
        v_term[k] = ocv_v[k] + I[k] * R0 + v1[k] + v2[k]

    return v_term, v1, v2
def combined_residuals(params, time_s, current_a, v_meas,
                       eta_coulomb=0.9974):
    """
    params = [R0, R1, C1, R2, C2,
              poly0, poly1, ..., poly8,   # highest power first
              Q_Ah, soc0]

    Returns residual vector: v_sim - v_meas
    """
    params = np.asarray(params, dtype=float)
    # Unpack
    R0, R1, C1, R2, C2 = params[0:5]
    poly_coeffs = params[5:5 + N_POLY]
    Q_Ah = params[5 + N_POLY]
    soc0 = params[6 + N_POLY]  # <-- FIXED
    # 1) SOC trajectory from Coulomb counting
    soc = compute_soc(time_s, current_a,
                      Q_Ah=Q_Ah,
                      eta=eta_coulomb,
                      soc0=soc0)
    soc = np.clip(soc, 0.0, 1.0)
    # 2) OCV(SOC) using polynomial
    ocv_v = np.polyval(poly_coeffs, soc)
    # 3) ECM simulation
    v_sim, v1, v2 = simulate_ecm_2nd_order_ocv(
        time_s, current_a, ocv_v,
        R0, R1, C1, R2, C2
    )
    # 4) Residuals (skip first point if you want)
    return v_sim - v_meas

# Main identification script
def main():
    # Load dataset
    # Here we use the OCVSOC dataset. To use Profile.xlsx instead,
    # replace the call with load_profile().
    data_path = Path("data/ocv_test_SP20.xlsx")
    time_s, v_meas, current_a, df = load_profile_ocvsoc(str(data_path))
    print("\n[INFO] Using dataset:", data_path)
    print("[INFO] Number of samples:", len(time_s))

    # Initial guess for parameters
    # a) ECM parameters (start from your previous identification or
    #    generic values)
    R0_0 = 0.03
    R1_0 = 0.02
    C1_0 = 2000.0
    R2_0 = 0.01
    C2_0 = 2000.0

    # b) OCV polynomial initial guess:
    #    simple linear-ish start using least squares on SOC inferred
    #    from an approximate Q_Ah / soc0
    Q_init = 1.55
    soc0_init = 1.0
    soc_init = compute_soc(time_s, current_a,
                           Q_Ah=Q_init, eta=0.9974, soc0=soc0_init)
    soc_init = np.clip(soc_init, 0.0, 1.0)
    # Fit polynomial on (soc_init, v_meas) as a rough starting point
    poly_init = np.polyfit(soc_init, v_meas, POLY_ORDER)
    # c) Q_Ah and soc0
    Q_Ah_0 = Q_init
    soc0_0 = soc0_init
    # Combine all into one parameter vector
    p0 = np.concatenate([
        np.array([R0_0, R1_0, C1_0, R2_0, C2_0]),
        poly_init,
        np.array([Q_Ah_0, soc0_0])
    ])
    print("\n[INFO] Initial parameter guess:")
    print(" R0, R1, C1, R2, C2 =", p0[0:5])
    print(" poly_init =", poly_init)
    print(" Q_Ah_0, soc0_0 =", Q_Ah_0, soc0_0)
    # Bounds (very important for stability)
    # ECM bounds
    R0_bounds = (1e-4, 0.2)
    R1_bounds = (1e-4, 0.5)
    C1_bounds = (10.0, 1e5)
    R2_bounds = (1e-4, 0.5)
    C2_bounds = (10.0, 1e5)
    # Polynomial coeffs: wide bounds (we trust initial fit)
    poly_bounds_low = [-1e4] * N_POLY
    poly_bounds_high = [1e4] * N_POLY
    # Q_Ah bounds: ±20% around nominal
    Q_bounds = (1.2, 1.9)
    # soc0 bounds: [0.8, 1.05]
    soc0_bounds = (0.8, 1.05)
    lower_bounds = np.array([
        R0_bounds[0], R1_bounds[0], C1_bounds[0], R2_bounds[0], C2_bounds[0],
        *poly_bounds_low,
        Q_bounds[0], soc0_bounds[0]
    ])

    upper_bounds = np.array([
        R0_bounds[1], R1_bounds[1], C1_bounds[1], R2_bounds[1], C2_bounds[1],
        *poly_bounds_high,
        Q_bounds[1], soc0_bounds[1]
    ])
# Run nonlinear least-squares
    print("\n[INFO] Starting combined least-squares fit...")
    result = least_squares(
        combined_residuals,
        p0,
        bounds=(lower_bounds, upper_bounds),
        args=(time_s, current_a, v_meas),
        verbose=2
    )

    # Extract optimal parameters
    p_opt = result.x
    R0_opt, R1_opt, C1_opt, R2_opt, C2_opt = p_opt[0:5]
    poly_opt = p_opt[5:5 + N_POLY]
    Q_Ah_opt = p_opt[5 + N_POLY]
    soc0_opt = p_opt[6 + N_POLY]

    print("\n===== Combined Identification Result =====")
    print("R0, R1, C1, R2, C2 =", [R0_opt, R1_opt, C1_opt, R2_opt, C2_opt])
    print("Polynomial coefficients (highest power first):")
    print(poly_opt)
    print(f"Q_Ah_opt = {Q_Ah_opt:.4f} Ah")
    print(f"soc0_opt = {soc0_opt:.4f}")

    # 5) SAVE combined parameters to data/combined_params.npy
    # Ensure data directory exists
    Path("data").mkdir(parents=True, exist_ok=True)
    save_combined_params(
        "data/combined_params.npy",
        R0_opt, R1_opt, C1_opt, R2_opt, C2_opt,
        poly_opt,
        Q_Ah_opt,
        soc0_opt,
    )
    # 6) Simulate with optimal parameters and plot
    soc_opt = compute_soc(time_s, current_a, Q_Ah=Q_Ah_opt,
                          eta=0.9974, soc0=soc0_opt)
    soc_opt = np.clip(soc_opt, 0.0, 1.0)
    ocv_opt = np.polyval(poly_opt, soc_opt)

    v_sim_opt, v1_opt, v2_opt = simulate_ecm_2nd_order_ocv(
        time_s, current_a, ocv_opt,
        R0_opt, R1_opt, C1_opt, R2_opt, C2_opt
    )
    mae = np.mean(np.abs(v_sim_opt - v_meas))
    print(f"\n[INFO] Final Voltage MAE (sim vs meas): {mae:.6f} V")

    plt.figure(figsize=(10, 4))
    plt.plot(time_s, v_meas, label="Measured V", linewidth=2)
    plt.plot(time_s, v_sim_opt, label="ECM + OCV (combined fit)", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Combined ECM + OCV Identification Result")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/combined_fit_ocvsoc.png", dpi=300, bbox_inches="tight")
    plt.show()
 
if __name__ == "__main__":
    main()