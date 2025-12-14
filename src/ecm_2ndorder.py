import numpy as np
from load_profile import load_profile
from soc_update import compute_soc
from ocv_model import ocv_from_soc

def simulate_ecm_2nd_order(time_s, current_a, soc, R0, R1, C1, R2, C2, v0_meas=None, discharge_sign=-1):
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    soc = np.asarray(soc, dtype=float)

    n = len(time_s)
    v1 = np.zeros(n, dtype=float)
    v2 = np.zeros(n, dtype=float)
    v_term = np.zeros(n, dtype=float)

    # Sign convention: I_dis = discharge_sign * current
    # if discharge_sign=-1 -> I_dis = -current (discharge current is negative in data)
    # if discharge_sign=+1 -> I_dis = +current (discharge current is positive in data)

    I_dis0 = discharge_sign * current_a[0]
    ocv0 = ocv_from_soc(soc[0])

    # Initialize polarization to match first measured voltage (if provided)
    if v0_meas is not None:
        # V = OCV - I*R0 - v1 - v2  => v1+v2 = OCV - I*R0 - V_meas
        pol0 = ocv0 - I_dis0 * R0 - float(v0_meas)
        v1[0] = pol0
        v2[0] = 0.0

    v_term[0] = ocv0 - I_dis0 * R0 - v1[0] - v2[0]

    for k in range(1, n):
        dt = time_s[k] - time_s[k - 1]
        if dt <= 0:
            dt = 0.0

        I_dis = discharge_sign * current_a[k]

        a1 = np.exp(-dt / (R1 * C1))
        a2 = np.exp(-dt / (R2 * C2))
        v1[k] = a1 * v1[k - 1] + R1 * (1.0 - a1) * I_dis
        v2[k] = a2 * v2[k - 1] + R2 * (1.0 - a2) * I_dis

        ocv_k = ocv_from_soc(soc[k])
        v_term[k] = ocv_k - I_dis * R0 - v1[k] - v2[k]

    return v_term, v1, v2