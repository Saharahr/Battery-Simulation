"""
Evaluate a 2nd-order ECM (plus SOC model) on one or more datasets.
- Profile.xlsx is ALWAYS evaluated with parameters from
  data/identified_params.npy  (parameter_id.py result).
- ocv_test_SP20.xlsx is evaluated with parameters from
  data/combined_params.npy    (combined_fit.py result).
Usage
-----
Default (evaluate both datasets):
    python src/evaluate.py
Single dataset (standard format):
    python src/evaluate.py --dataset data/Profile.xlsx
Single dataset (OCVSOC / OCV test style):
    python src/evaluate.py --dataset data/Incremental OCV test_SP20-1.xlsx --ocvsoc
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from load_profile import load_profile, load_profile_ocvsoc
from soc_update import compute_soc
from ecm_2ndorder import simulate_ecm_2nd_order
from ocv_fit import plot_OCV_curve, dis_soc, dis_ocv

def load_identified_params(path="data/identified_params.npy"):
    """Load ECM parameters identified on Profile.xlsx (parameter_id.py)."""
    p = np.load(path)
    print(f"[INFO] Loaded ECM parameters from {path}: {p}")
    return p

def plot_ecm_result(time_s, v_meas, v_sim, title, save_path):
    """Plot measured vs simulated voltage and save to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(time_s, v_meas, label="Measured V", linewidth=2)
    plt.plot(time_s, v_sim, label="ECM V", linewidth=2, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Saved plot → {save_path}")

# Core evaluation function
def evaluate_dataset(
    params_ecm,
    dataset_path,
    *,
    is_ocvsoc=False,
    Q_Ah=1.5546,
    eta=0.9974,
    soc0=0.95,
    drop_first=True,
    label="",
    save_name=None,
):
    """
    Evaluate ECM on a given dataset.
    """
    dataset_path = Path(dataset_path)
    params_ecm = np.asarray(params_ecm, dtype=float)
    R0, R1, C1, R2, C2 = params_ecm

    print("\n=======================================")
    print(" ECM Evaluation")
    print(" Dataset:", dataset_path)
    print(" Format :", "OCVSOC" if is_ocvsoc else "Standard")
    print("=======================================\n")

    # 1) Load profile
    if is_ocvsoc:
        time_s, v_meas, current_a, df = load_profile_ocvsoc(str(dataset_path))
    else:
        time_s, v_meas, current_a, df = load_profile(str(dataset_path))

    # 2) SOC trajectory
    soc = compute_soc(time_s, current_a, Q_Ah=Q_Ah, eta=eta, soc0=soc0)

    # 3) ECM simulation
    v_sim, v1, v2 = simulate_ecm_2nd_order(
        time_s, current_a, soc, R0, R1, C1, R2, C2
    )

    # 4) Optionally drop first sample
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

    # 5) Plot and save
    if save_name is None:
        basename = dataset_path.stem.replace(" ", "_").lower()
        save_name = f"images/ecm_eval_{basename}.png"

    title = "2nd-Order ECM Evaluation"
    if label:
        title += f" - {label}"

    plot_ecm_result(t, vm, vs, title, save_name)

    return mae

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ECM model on one or more datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a single Excel dataset to evaluate. "
             "If omitted, both data/Profile.xlsx and data/Incremental OCV test_SP20-1.xlsx are run.",
    )
    parser.add_argument(
        "--ocvsoc",
        action="store_true",
        help="Interpret the --dataset file as OCVSOC format "
             "(Duration(sec), mV, mA, ...).",
    )
    parser.add_argument(
        "--no-drop-first",
        action="store_true",
        help="Do not drop the first sample when computing MAE/plotting.",
    )
    return parser.parse_args()

# MAIN
def main():
    args = parse_args()
    # Load ECM parameters identified from Profile experiment
    ecm_profile = load_identified_params("data/identified_params.npy")
    # Use the SAME ECM params and SAME capacity for all datasets
    ecm_ocv = ecm_profile          # same R0, R1, C1, R2, C2
    Q_Ah_profile = 1.5546          # capacity used for Profile
    Q_Ah_ocv = Q_Ah_profile        # same capacity for OCV test
    soc0_profile = 0.9037          # initial SOC for Profile dataset
    soc0_ocv = 1.0                 # initial SOC for OCV dataset (fully charged)

    if args.dataset:
        # -------- Single Dataset Mode --------
        dataset_path = args.dataset

        if args.ocvsoc:
            # OCVSOC format: use SAME ECM params as Profile
            params = ecm_ocv
            q_ah = Q_Ah_ocv
            soc_0 = soc0_ocv
        else:
            # Standard format: use SAME ECM params as usual
            params = ecm_profile
            q_ah = Q_Ah_profile
            soc_0 = soc0_profile

        evaluate_dataset(
            params,
            dataset_path,
            is_ocvsoc=args.ocvsoc,
            Q_Ah=q_ah,
            soc0=soc_0,
            label=Path(dataset_path).name,
        )

    else:
        # -------- Default: Run Both Main Datasets --------

        # 1. Evaluate Profile.xlsx (Standard format)
        evaluate_dataset(
            ecm_profile,
            "data/Profile.xlsx",
            is_ocvsoc=False,
            Q_Ah=Q_Ah_profile,
            soc0=soc0_profile,
            label="Profile.xlsx",
            save_name="images/ecm_eval_profile.png",
        )

        # 2. Evaluate ocv_test_SP20.xlsx (OCVSOC format)
        #    Uses EXACTLY the same R0, R1, C1, R2, C2 and Q_Ah as above,
        #    only SOC0 differs (1.0 for the OCV test).
        evaluate_dataset(
            ecm_ocv,
            "data/Incremental OCV test_SP20-1.xlsx",
            is_ocvsoc=True,
            Q_Ah=Q_Ah_ocv,
            soc0=soc0_ocv,
            label="Incremental OCV test_SP20-1.xlsx",
            save_name="images/ecm_eval_ocvsoc.png",
        )

if __name__ == "__main__":
    main()