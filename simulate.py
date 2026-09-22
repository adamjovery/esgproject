#Importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import yfinance as yf

#Stopping the print from showing up with "np.float64"
np.set_printoptions(legacy='1.25')

#Setting years we wish to follow within the historical data collection
firstyear = 2016
lastyear = 2026

#Setting simulation parameters for esg
horizon = 10
steps_yearly = 12
steps_total = steps_yearly * horizon

#Intialising keys
model_go = False
liability_go = False
record = False
record2 = False

#Requires input from user as to when to begin the model, and whether to save the fan chart produced
while model_go != True:
    print("\nPlease enter 1 if you wish to begin running the model")
    if input() == "1": 
        model_go = True
    else:
        print("\nModel not begun")
print("\nPlease enter 1 if you wish to save the fan chart to a file labelled: 'esg_fan_chart.png', \notherwise it will not be saved")
if input() == "1":
    print("\nFan chart will be saved")
    save_path_direction="esg_fan_chart.png"
else:
    print("\nFan chart will not be saved")
    save_path_direction = None

#Historical data sources used:
# rates:     10-Year Treasury yield (DGS10)
# inflation: CPI YoY % change (CPIAUCSL)
# equity:    SPY ETF close (proxying for S&P 500)

#Importing the functions from the modelling, correlation, calibration, and fan chart files.
from correlation import correlated_shocks
from models import vasicek, ornstein, gbm_log
from calibration import calibrate_mean_reverting, calibrate_gbm, calibrate_correlation_matrix
from fan_chart import plot_esg_fan_charts
from liabilities import value_liability, discount_factors, escalation_factors, liability_metrics, deterministic_pv

#Function to build the parameters from the historical data through OLS for the mean reverting models, and through expected returns for the GBM, as well as calculating the correlation matrix
def hist_params(hist_rates, hist_inflation, hist_equity, dt_hist):
    #Using calibration functions to calculate the parameters for each model and the correlation matrix
    rate_fit = calibrate_mean_reverting(hist_rates, dt=dt_hist)
    inflation_fit = calibrate_mean_reverting(hist_inflation, dt=dt_hist)
    equity_fit = calibrate_gbm(hist_equity, dt=dt_hist)
    corr_matrix = calibrate_correlation_matrix(hist_rates, hist_inflation, hist_equity)

    #Creating a dictionary of the parameters within each model
    rate_params = dict(r0=hist_rates[-1], a=rate_fit["a"], b=rate_fit["b"], sigma=rate_fit["sigma"])
    inflation_params = dict(i0=hist_inflation[-1], a=inflation_fit["a"], b=inflation_fit["b"], sigma=inflation_fit["sigma"])
    equity_params = dict(e0=hist_equity[-1], mu=equity_fit["mu"], sigma=equity_fit["sigma"])

    #Returns the caculated correlation matrix and parameter dictionaries, as well as the targets fitted from the historical data
    return {"corr_matrix": corr_matrix, "rate_params": rate_params, "inflation_params": inflation_params, "equity_params": equity_params, "diagnostics": {"rate_fit": rate_fit, "inflation_fit": inflation_fit, "equity_fit": equity_fit},}
 
#Function to run the correlated 3 factor esg
def run_esg(T, steps, paths, corr_matrix, rate_params, inflation_params, equity_params):
    #Build the correlated shocks from the correlation matrix and shocks generated from a normal distribution
    z = correlated_shocks(corr_matrix, steps, paths)

    #Runs each model with the historical parameters and generated correlated shocks
    rates = vasicek(**rate_params, T=T, steps=steps, paths=paths, z=z[0])
    inflation = ornstein(**inflation_params, T=T, steps=steps, paths=paths, z=z[1])
    equity = gbm_log(**equity_params, T=T, steps=steps, paths=paths, z=z[2])
    
    #Returns the results of running the esg 
    return {"rates": rates, "inflation": inflation, "equity": equity}

#Main body of code
if model_go is True:
    dt_monthly = horizon / steps_total

    #Intialising dates
    start = datetime.datetime(firstyear, 1, 1)
    end = datetime.datetime(lastyear, 1, 1)

    #Raw series data collection
    rates_raw = web.DataReader('DGS10', 'fred', start, end) / 100 
    cpi_raw = web.DataReader('CPIAUCSL', 'fred', start, end)     
    equity_raw = yf.download('SPY', start=start, end=end)['Close']   
    equity_raw.columns = equity_raw.columns.get_level_values(0)

    #Resample each to monthly, converting CPI to YoY inflation
    rates_m = rates_raw.resample('MS').last()
    inflation_m = cpi_raw.pct_change(12, fill_method=None).resample('MS').last() 
    equity_m = equity_raw.resample('MS').last()
 
    #Aligning on a shared date index and dropping missing data
    combined = pd.concat([rates_m, inflation_m, equity_m], axis=1, keys=['rate', 'inflation', 'equity']).dropna()

    #Separating and flattening to get historical series for each variable
    hist_rates = combined['rate'].values.flatten()
    hist_inflation = combined['inflation'].values.flatten()
    hist_equity = combined['equity'].values.flatten()

    #Calling parameter finding function from historical data and then printing the results
    esg_inputs = hist_params(hist_rates, hist_inflation, hist_equity, dt_hist=dt_monthly)
    print("\nCalibrated parameters for each model:")
    print("Rates:    ", esg_inputs["rate_params"])
    print("Inflation:", esg_inputs["inflation_params"])
    print("Equity:   ", esg_inputs["equity_params"])
    print("\nCorrelation matrix [rates, inflation, equity]:")
    print(np.round(esg_inputs["corr_matrix"], 3))
 
    #Simulating the esg forwards across 10000 paths monthly for ten years using the found parameters
    results = run_esg(T=horizon, steps=steps_total, paths=10000, corr_matrix=esg_inputs["corr_matrix"], rate_params=esg_inputs["rate_params"], inflation_params=esg_inputs["inflation_params"], equity_params=esg_inputs["equity_params"])
    rates, inflation, equity = results["rates"], results["inflation"], results["equity"]
    
    #Plotting the fan charts for the esg results
    fan_chart = plot_esg_fan_charts(results, T=horizon, save_path=save_path_direction)
    
    #Calibrating targets from the historical data
    b_rate = esg_inputs["rate_params"]["b"]
    b_inflation = esg_inputs["inflation_params"]["b"]
    e0 = esg_inputs["equity_params"]["e0"]
    mu = esg_inputs["equity_params"]["mu"]

    #Printing esg results summary and comparison with targets derived historical results
    print("\nSummary in year 10 across 10,000 paths (calibrations in brackets):")
    print(f"Mean interest rate:  {rates[-1].mean():.4f}  (calibrated target b={b_rate:.4f})")
    print(f"Mean inflation rate:   {inflation[-1].mean():.4f}  (calibrated target b={b_inflation:.4f})")
    print(f"Mean equity level:{equity[-1].mean():.2f}  (calibrated expected value of equity={e0*np.exp(mu*10):.2f})")
 
    #Validating whether the correlations in the esg results are similar to the correlations in the historical data
    #Finding correlation matrix by calculating changes/log-returns in the esg results series
    d_rates = np.diff(rates, axis=0)
    d_inflation = np.diff(inflation, axis=0)
    d_log_equity = np.diff(np.log(equity), axis=0)

    #Calculating the pearson correlation coefficient for the interest rates versus the inflation rates and the interest rates versus the equity price levels
    corr_rate_inflation = np.corrcoef(d_rates.flatten(), d_inflation.flatten())[0, 1]
    corr_rate_equity = np.corrcoef(d_rates.flatten(), d_log_equity.flatten())[0, 1]

    #Retrieving the targeted correlations in the historical data from the historical parameters
    target_rate_inflation = esg_inputs["corr_matrix"][0, 1]
    target_rate_equity = esg_inputs["corr_matrix"][0, 2]

    #Printing the realised correlations versus the targeted correlations
    print("\nRealised correlations (calibrations in brackets)")
    print(f"rates vs inflation: {corr_rate_inflation:.3f}  (calibrated target {target_rate_inflation:.3f})")
    print(f"rates vs equity:    {corr_rate_equity:.3f}  (calibrated target {target_rate_equity:.3f})")

    #Displays the fan chart and prevents it from being displayed after the file concludes 
    display(fan_chart["overall_chart"])
    plt.close(fan_chart["overall_chart"])

#Requires input from user as to when to begin the liability calculation section, and parameters such as the annual payment value of the liability, the desired confidence interval, and whether the liability is inflation linked
#Repeats until a 1 is entered
while liability_go != True:
    print("\nPlease enter 1 if you wish to perform liability calculations based upon the esg results")
    if input() == "1":
        liability_go = True
    else:
        print("\nModel not begun")

#Repeats until a valid liability amount is entered
while record != True:
    print("\nPlease enter the current annual payment value of your liability in integer or decimal form")
    liability_value = input()
    error = 0

    #Tests for the form of the string and denies if it cannot be a float
    try:
        float(liability_value)
    except ValueError:
        error = 1
    if error == 0:
        liability_value = float(liability_value)
        record = True
    else:
        print("\nError please try again")

#Repeats until a valid confidence interval is entered
while record2 != True:
    print("\nPlease enter your desired confidence interval for the VaR and TVaR in decimal form, \nwith this being less than or equal to 1 and greater than or equal to 0")
    error2 = 0
    confidence = input()

    #Tests for the form of the string and denies if it cannot be a float
    try:
        float(confidence)
    except ValueError:
        error2 = 1
    if error2 == 0 and 1 >= float(confidence) >= 0:
        confidence = float(confidence)
        record2 = True
    else:
        print("\nError please try again")

#Asks about the inflation linked status of the liability
print("\nPlease enter 1 if you wish the liability to be inflation linked")
if input() == "1":
    print("\nLiability recorded as inflation linked")
    link_status = True
else:
    print("\nLiability recorded as not inflation linked")
    link_status = False

#Asks whether user wants to save the liability PV histogram
print("\nPlease enter 1 if you wish to save the liability PV histogram to a file labelled: 'liabilities_pv_histogram.png', \notherwise it will not be saved")
if input() == "1":
    print("\nHistogram will be saved")
    save_histogram = 1
else:
    print("\nHistogram will not be saved")
    save_histogram = 0

#Main body of the liability section of the model
if liability_go == True:
    #Calls the function which calculates the present values of the liability under each path
    path_pv = value_liability(rates, inflation, base_annual_payment=liability_value, T=horizon, steps_per_year=12, inflation_linked=link_status)

    #Calls the function which calculates the metrics based on the present values of the liability under each path and the desired confidence interval
    metrics = liability_metrics(path_pv, confidence)

    #Prints the metrics returned by the previous function
    print(f"\nStochastic liability present valuation for {int(horizon)} years across 10000 paths")
    for i, j in metrics.items():
        print(f"{i:10s}: {j:8.2f}")

    #Calls a function which calculates the deterministic present value based upon the mean results for interest and inflation rates from the historical data that was found via OLS
    det_pv = deterministic_pv(base_annual_payment=liability_value, det_discount_rate=b_rate, det_escalation_rate=b_inflation, T=horizon, inflation_linked=link_status)

    #Prints comparisons between the metrics for the stochastic and deterministic present value, as well as the required capital buffer to cover the risk of the worst outcomes appearing
    print(f"\nDeterministic PV (flat b assumptions): {det_pv:.2f}")
    print(f"Stochastic mean PV: {metrics['mean']:.2f}")
    print(f"Difference (stochastic - deterministic): {metrics['mean'] - det_pv:.2f}")
    print(f"\nRequired capital buffer (VaR_{int(confidence*100)} - mean): {metrics[f"VaR_{int(confidence*100)}"] - metrics['mean']:.2f}")
    print(f"Required capital buffer (TVaR_{int(confidence*100)} - mean): {metrics[f"TVaR_{int(confidence*100)}"] - metrics['mean']:.2f}")

    #Creating a histogram to plot the distribution of liability present value across 10000 paths
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.hist(path_pv, bins=80, color='royalblue', alpha=0.7, edgecolor='white')

    #Plotting the mean for both the stochastic and deterministic cases
    ax.axvline(metrics['mean'], color='indigo', linewidth=2, linestyle='-', label=f"Stochastic mean PV = {metrics['mean']:.2f}")
    ax.axvline(det_pv, color='darkorange', linewidth=2, linestyle='--', label=f"Deterministic PV = {det_pv:.2f}")

    #Plotting the VaR and TVaR lines
    ax.axvline(metrics[f'VaR_{int(confidence*100)}'], color='lightcoral', linewidth=2, linestyle=':', label=f"VaR_{int(confidence*100)} = {metrics[f'VaR_{int(confidence*100)}']:.2f}")
    ax.axvline(metrics[f'TVaR_{int(confidence*100)}'], color='maroon', linewidth=2, linestyle=':', label=f"TVaR_{int(confidence*100)} = {metrics[f'TVaR_{int(confidence*100)}']:.2f}")

    #Creating the axis titles, legend and grid
    ax.set_title("Distribution of liability present value across 10000 paths", fontsize=12, fontweight='bold')
    ax.set_xlabel("Present value")
    ax.set_ylabel("Number of paths")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    if save_histogram == 1:
        fig2.savefig("liabilities_pv_histogram.png", dpi=150, bbox_inches='tight')
    else:
        pass
