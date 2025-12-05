import numpy as np
import matplotlib.pyplot as plt
#Exprimental OCV-SOC data (discharge)
dis_soc = np.array([
    1.0000000000000000,
    0.9037630258587420,
    0.8074618937347230,
    0.7111607616107040,
    0.6148596294866850,
    0.5185584973626660,
    0.4222573652386470,
    0.3259562331146280,
    0.2296551009906090,
    0.1333539688665900,
    0.1141065740383380,
    0.0948591792100860,
    0.0756117843818350,
    0.0563643895535830,
    0.0371169947253310
])
dis_ocv = np.array([
    4.1980,
    4.0795,
    4.0226,
    3.9527,
    3.8666,
    3.7811,
    3.6935,
    3.6372,
    3.5651,
    3.4895,
    3.4800,
    3.4621,
    3.4005,
    3.3059,
    3.1677
])
def plot_OCV_curve(dis_soc, dis_ocv, poly_order = 8, show_plot = True):
    """Take discrete experimental OCV-SOC samples and fit an 8th-order polynomial.

    Args:
        dis_soc (array): experimental data arrays
        dis_ocv (array): experimental data arrays
        poly_order (int): _description_. Defaults to 8.
        show_plot (bool): _description_. Defaults to True.
    
    Returns:
    coeffs: np.ndarray
    mae: float
    """ 
    dis_soc = np.asarray(dis_soc)
    dis_ocv = np.asarray(dis_ocv)

    coeffs = np.polyfit(dis_soc, dis_ocv, poly_order) #Polynominal fit

    #Evaluate polynominal at original SOC points for Mean Absolute Error.
    ocv_fit_at_data = np.polyval(coeffs, dis_soc)
    mae = np.mean(np.abs(ocv_fit_at_data - dis_ocv))

    #For smooth curve, evaluate on a dense SOC grid
    soc_grid = np.linspace(0.0, 1.0, 200)
    ocv_grid = np.polyval(coeffs, soc_grid)

    #Plot
    if show_plot:
        plt.figure()
        plt.scatter(dis_soc, dis_ocv, label = "Measuerd data", marker = "o")
        plt.plot(soc_grid, ocv_grid, label = f"{poly_order}th-order fit", linewidth = 2)
        plt.xlabel("SOC")
        plt.ylabel("OCV (V)")
        plt.title("OCV-SOC Curve Fitting")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Polynominal coefficients (highest power first) :\n{coeffs}")
    print(f"Mean Absoulute Error: {mae: .6f} V")

    return coeffs, mae

if __name__ == "__main__":
    plot_OCV_curve(dis_soc, dis_ocv)
