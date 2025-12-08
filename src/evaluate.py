# evaluate.py
import numpy as np
from pathlib import Path
from load_profile import load_profile, load_profile_ocvsoc
from soc_update import compute_soc
from ecm_2ndorder import simulate_ecm_2nd_order
from parameter_id import plot_parameter_id
from ocv_fit import plot_OCV_curve
# ECM EVALUATION
def evaluate_dataset(params, dataset_path, *, is_ocvsoc=False, Q_Ah=1.5546, eta=0.9974, soc0=0.95, drop_first=True,
):
    """
    Evaluate ECM parameters on any dataset.
    """
    params = np.asarray(params, dtype=float)
    R0, R1, C1, R2, C2 = params
    dataset_path = Path(dataset_path)
    print(" ECM Evaluation")
    print(" Dataset:", dataset_path)
    print(" Format :", "OCVSOC" if is_ocvsoc else "Standard")
    # Load dataset
    if is_ocvsoc:
        time_s, v_meas, current_a, df = load_profile_ocvsoc(str(dataset_path))
    else:
        time_s, v_meas, current_a, df = load_profile(str(dataset_path))

    # Compute SOC
    soc = compute_soc(time_s, current_a, Q_Ah=Q_Ah, eta=eta, soc0=soc0)

    # Simulate ECM
    v_sim, v1, v2 = simulate_ecm_2nd_order(
        time_s, current_a, soc,
        R0, R1, C1, R2, C2
    )

    # Remove first point if desired
    if drop_first:
        t = time_s[1:]
        vm = v_meas[1:]
        vs = v_sim[1:]
    else:
        t = time_s
        vm = v_meas
        vs = v_sim

    mae = float(np.mean(np.abs(vs - vm)))
    print(f"[INFO] Voltage MAE: {mae:.6f} V")

    out_name = f"images/ecm_eval_{'ocvsoc' if is_ocvsoc else 'profile'}.png"
    plot_parameter_id(t, vm, vs, save=True, show=True, custom_name=out_name)

    return mae


def load_identified_params(path="data/identified_params.npy"):
    """Load ECM parameters saved by parameter_id.py."""
    p = np.load(path)
    print(f"[INFO] Loaded parameters from {path}: {p}")
    return p

# OCV FIT EVALUATION
def evaluate_ocv_fit(dis_soc, dis_ocv, poly_order=8):
    """
    Evaluate the OCV-SOC polynomial fit using your existing function.
    This produces:
     - Plot of measured vs fitted OCV
     - Polynomial coefficients
     - MAE
    """
    print(" OCV-SOC Fit Evaluation")
    print(" Polynomial order:", poly_order)

    coeffs, mae = plot_OCV_curve(
        dis_soc,
        dis_ocv,
        poly_order=poly_order,
        show_plot=True
    )

    print(f"[INFO] OCV Fit MAE: {mae:.6f} V")
    return coeffs, mae
# MAIN
def main():

    # Load ECM parameters
    params = load_identified_params("data/identified_params.npy")

    # -------------------------
    # 1) Evaluate ECM
    # -------------------------
    evaluate_dataset(
        params,
        "data/Profile.xlsx",
        is_ocvsoc=False,
        soc0=0.9037,
    )

    evaluate_dataset(
        params,
        "data/ocv_test_SP20.xlsx",
        is_ocvsoc=True,
        soc0=1.0,
    )
    # -------------------------
    # 2) Evaluate OCV fit
    # -------------------------
    from ocv_fit import dis_soc, dis_ocv
    evaluate_ocv_fit(dis_soc, dis_ocv, poly_order=8)


if __name__ == "__main__":
    main()
