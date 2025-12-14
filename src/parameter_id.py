import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from load_profile import load_profile
from soc_update import compute_soc
from ecm_2ndorder import simulate_ecm_2nd_order

def residuals(params, time_s, current_a, soc, v_meas):
    """
    Use scipy.optimize.least_squares to find R0, R1, C1, R2, C2 that minimize voltage error.
    Residuals for least-squares:
        r(k) = V_sim(k; params) - V_meas(k)
    """
    # simulate ECM with given parameters
    R0, R1, C1, R2, C2 = params
    v_sim, v1, v2 = simulate_ecm_2nd_order(time_s, current_a, soc,
                                           R0, R1, C1, R2, C2)
    return(v_sim - v_meas) # residual = simulated - measured

def main():
    time_s, v_meas, current_a, df = load_profile()
    soc = compute_soc(time_s, current_a, Q_Ah = 1.5546, eta = 0.9974, soc0 = 0.9037)
    # remove the very first sample to avoid the v_term[0]=0 transient
    time_fit = time_s[1:]
    v_meas_fit = v_meas[1:]
    current_fit = current_a[1:]
    soc_fit = soc[1:]
    # Initial guesses (same as we used before)
    p0 = np.array([
        0.01,   # R0
        0.01,   # R1
        1500.0, # C1
        0.02,   # R2
        4000.0  # C2
    ])
     # Bounds to keep parameters positive and in a reasonable range
    lower_bounds = np.array([1e-4, 1e-4, 1.0,   1e-4, 1.0])
    upper_bounds = np.array([0.5,  1.0,  1e5,  1.0,  1e5])
    # Run nonlinear least-squares optimization\
    result = least_squares(
        residuals,
        p0,
        bounds = (lower_bounds, upper_bounds),
        args = (time_fit, current_fit, soc_fit, v_meas_fit),
        verbose = 2
    )

    p_opt = result.x
    R0_opt, R1_opt, C1_opt, R2_opt, C2_opt = p_opt
    print("\n===== Identified ECM Parameters =====")
    np.save("data/identified_params.npy", p_opt)
    print("[INFO] Saved parameters to data/identified_params.npy")
    print(f"R0 = {R0_opt:.6f} ohm")
    print(f"R1 = {R1_opt:.6f} ohm")
    print(f"C1 = {C1_opt:.2f} F")
    print(f"R2 = {R2_opt:.6f} ohm")
    print(f"C2 = {C2_opt:.2f} F")

    # Simulate again with optimal parameters over full profile
    v_sim_full, v1_full, v2_full = simulate_ecm_2nd_order(
        time_s, current_a, soc, R0_opt, R1_opt, C1_opt, R2_opt, C2_opt
    )

    #Compute MAE between measured and sumulated voltage
    mae = np.mean(np.abs(v_sim_full[1:] - v_meas[1:]))
    print(f"\nVoltage MAE (sim vs meas): {mae:.6f} V")
    plot_parameter_id(time_s[1:], v_meas[1:], v_sim_full[1:])


    #plot measured vs simulated voltage
def plot_parameter_id(time_s, v_meas, v_sim,
                      save=True, show=True, auto_open=False,
                      custom_name=None):
    fig = plt.figure(figsize=(10,4))
    plt.plot(time_s, v_meas, label="Measured V", linewidth=2)
    plt.plot(time_s, v_sim, label="ECM V (identified)", linewidth=2, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("2nd-Order ECM - Parameter Identification Result")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        if custom_name is None:
            save_path = "images/parameter_id.png"
        else:
            save_path = custom_name

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved → {save_path}")

if __name__ == "__main__":
    main()