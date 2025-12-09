import numpy as np
from load_profile import load_profile
def compute_soc(time_s, current_a, Q_Ah = 1.5546, eta = 0.9974, soc0 = 0.9037):
    """
    ComputeSoC(k) from current and time
    Args:
        time_s (float)
        current_a (float)
        Q_Ah (float): -capacity-.Defaults to 1.5546.
        eta (float): _coulombic efficiency_. Defaults to 0.9974.
        soc0 (float): _initial cell SoC Defaults to 0.9037.
    Returns:
        soc : np.ndarray
            SOC trajectory (same length as time_s), clipped to [0, 1].
    """
    time_s = np.asarray(time_s, dtype = float)
    current_a = np.asarray(current_a, dtype = float)
    n = len(time_s)
    soc = np.zeros(n, dtype = float)
    soc[0] = soc0
    for k in range(1, n): #loop over time steps
        dt = time_s[k] - time_s[k-1]
        I = -current_a[k]
        d_soc = eta * I * dt / (3600.0 * Q_Ah)
        soc[k] = soc[k-1] - d_soc
        soc[k] = np.clip(soc[k], 0.0, 1.0) #enforce [0, 1]
    return soc

if __name__ == "__main__":
    time_s, voltage_v, current_a, df = load_profile()
    soc = compute_soc(time_s, current_a, Q_Ah = 1.5546, eta = 0.9974, soc0 = 0.9037)
    print("First 20 current values:")
    print(current_a[:20])
    print("SOC min:", soc.min())
    print("SOC max:", soc.max())
    print("\nFirst 15 SOC values:")
    print(soc[:15])
    print(f"\nFinal SOC: {soc[-1]:.5f}")