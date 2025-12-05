import numpy as np
# 8th-order OCV-SOC polynominal from ocv_fit.py
OCV_POLY_COEFFS = np.array([
    -6.09700815e+02,
     2.66789725e+03,
    -4.84696165e+03,
     4.74204500e+03,
    -2.71200299e+03,
     9.21018442e+02,
    -1.79710157e+02,
     1.89412066e+01,
     2.67148071e+00,
])
def ocv_from_soc(soc):
    """
    Evaluate OCV(SOC) using the fitted polynomial.
    Args:
        soc (float)
    Return:
     ocv : float or np.ndarray
        Open-circuit voltage (V).
    """
    soc = np.clip(soc, 0.0, 1.0)
    return np.polyval(OCV_POLY_COEFFS, soc)