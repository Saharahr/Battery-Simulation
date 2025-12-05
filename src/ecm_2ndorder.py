import numpy as np
from load_profile import load_profile
from soc_update import compute_soc
from ocv_model import ocv_from_soc

def simulate_ecm_2nd_order(time_s, current_a, soc,
                            R0, R1, C1, R2, C2):
    """Simulate a 2nd-order ECM (R0 + two RC branches).

    Args:
        time_s (float): From load Profile
        current_a (float): From load Profile
        soc (float): State of Charge from soc_Update
        R0 (float): Ohmiv resistance
        R1, C1 (float): first RC pair
        R2, C2 (float): Second RC pair        
        States:
        v1, v2 : RC branch voltages (V)
        Terminal voltage:
        V = OCV(SOC) - I_dis*R0 - v1 - v2
        OCV(SOC) from polynomial
    """
    time_s = np.asarray(time_s, dtype = float)
    current_a = np.asarray(current_a, dtype = float)
    soc = np.asarray(soc, dtype = float)

    n = len(time_s)
    v1 = np.zeros(n, dtype = float)
    v2 = np.zeros(n, dtype = float)
    v_term = np.zeros(n, dtype = float)

    for k in range(1, n): #Loop over time
        dt = time_s[k] - time_s[k-1]
        I_dis = -current_a[k] #positive discharge Current
        # Exact discrete-time updates for RC branches
        # dv/dt = -v/(R*C) + I/C 
        a1 = np.exp(-dt / (R1 * C1))
        a2 = np.exp(-dt / (R2 * C2))
        v1[k] = a1 * v1[k-1] + R1 * (1 - a1) * I_dis
        v2[k] = a2 * v2[k-1] + R2 * (1 - a2) * I_dis

        ocv_k = ocv_from_soc(soc[k]) #ocv from SOC
        v_term[k] = ocv_k - I_dis * R0 - v1[k] - v2[k]
    return v_term, v1, v2

if __name__ == "__main__":
    time_s, v_meas, current_a, df = load_profile()
    soc = compute_soc(time_s, current_a, Q_Ah = 1.5546, eta = 0.9974, soc0 = 0.9037)
    # Rough initial guesses – will be replaced by identified values later
    R0 = 0.05
    R1 = 0.01
    C1 = 2000.0
    R2 = 0.02
    C2 = 1500.0
    v_sim, v1, v2 = simulate_ecm_2nd_order(time_s, current_a, soc, R0, R1, C1, R2, C2)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_s, v_meas, label="Measured V")
    plt.plot(time_s, v_sim, label="ECM V (initial guess)", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.title("2nd-Order ECM - Initial Parameter Guess")
    plt.show()