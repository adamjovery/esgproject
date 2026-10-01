#Importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import yfinance as yf

#Stopping the print from showing up with "np.float64"
np.set_printoptions(legacy='1.25')

###########################
#First COVID-robust model

#Setting years we wish to follow within the historical data collection
firstyear = 2006
lastyear = 2026
hist_horizon = lastyear - firstyear
steps_yearly = 12
hist_steps_total = steps_yearly * hist_horizon

#Setting simulation parameters for esg
horizon = 10
steps_total = steps_yearly * horizon

#Intialising keys
model_go = False
liability_go = False
record = False

#Requires input from user as to when to begin the model, and whether to save the fan chart produced
while model_go != True:
    print(f"\nPlease enter 1 if you wish to begin running the first COVID-robust model ({firstyear} to {lastyear})")
    if input() == "1": 
        model_go = True
    else:
        print("\nModel not begun")
print("\nPlease enter 1 if you wish to save the fan chart to a file labelled: 'robust_fan_chart.png', \notherwise it will not be saved")
if input() == "1":
    print("\nFan chart will be saved")
    save_path_direction="robust_fan_chart.png"
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
    dt_hist = hist_horizon / hist_steps_total
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
    esg_inputs = hist_params(hist_rates, hist_inflation, hist_equity, dt_hist=dt_hist)
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
    fan_chart = plot_esg_fan_charts(results, T=horizon, subtitle = f"COVID-robust ESG ({firstyear} to {lastyear}): Simulated Paths", save_path=save_path_direction)
    
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

######################
#Second COVID-robust model

#Setting years we wish to follow within the historical data collection
firstyear2 = 2009
lastyear2 = 2019
hist_horizon2 = lastyear2 - firstyear2
steps_yearly2 = 12
hist_steps_total2 = steps_yearly2 * hist_horizon2

#Setting simulation parameters for esg
horizon2 = 10
steps_total2 = steps_yearly2 * horizon2

#Intialising keys
model_go2 = False
liability_go2 = False
record2 = False

#Requires input from user as to when to begin the second COVID-robust model, and whether to save the fan chart produced
while model_go2 != True:
    print(f"\nPlease enter 1 if you wish to begin running the second COVID-robust model ({firstyear2} to {lastyear2})")
    if input() == "1": 
        model_go2 = True
    else:
        print("\nModel not begun")
print("\nPlease enter 1 if you wish to save the fan chart to a file labelled: 'robust2_fan_chart.png', \notherwise it will not be saved")
if input() == "1":
    print("\nFan chart will be saved")
    save_path_direction2="robust2_fan_chart.png"
else:
    print("\nFan chart will not be saved")
    save_path_direction2 = None

#Main body of code
if model_go2 is True:
    dt_monthly2 = horizon2 / steps_total2
    dt_hist2 = hist_horizon2 / hist_steps_total2
    #Intialising dates
    start2 = datetime.datetime(firstyear2, 1, 1)
    end2 = datetime.datetime(lastyear2, 1, 1)

    #Raw series data collection
    rates_raw = web.DataReader('DGS10', 'fred', start2, end2) / 100 
    cpi_raw = web.DataReader('CPIAUCSL', 'fred', start2, end2)     
    equity_raw = yf.download('SPY', start=start2, end=end2)['Close']   
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
    esg_inputs = hist_params(hist_rates, hist_inflation, hist_equity, dt_hist=dt_hist2)
    print("\nCalibrated parameters for each model:")
    print("Rates:    ", esg_inputs["rate_params"])
    print("Inflation:", esg_inputs["inflation_params"])
    print("Equity:   ", esg_inputs["equity_params"])
    print("\nCorrelation matrix [rates, inflation, equity]:")
    print(np.round(esg_inputs["corr_matrix"], 3))
 
    #Simulating the esg forwards across 10000 paths monthly for ten years using the found parameters
    results = run_esg(T=horizon2, steps=steps_total2, paths=10000, corr_matrix=esg_inputs["corr_matrix"], rate_params=esg_inputs["rate_params"], inflation_params=esg_inputs["inflation_params"], equity_params=esg_inputs["equity_params"]) 
    rates, inflation, equity = results["rates"], results["inflation"], results["equity"]
    
    #Plotting the fan charts for the esg results
    fan_chart = plot_esg_fan_charts(results, T=horizon2, subtitle = f"COVID-robust ESG ({firstyear2} to {lastyear2}): Simulated Paths", save_path=save_path_direction2)
    
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

######################
#Third COVID-robust model

#Setting years we wish to follow within the historical data collection
firstyear3 = 1999
lastyear3 = 2019
hist_horizon3 = lastyear3 - firstyear3
steps_yearly3 = 12
hist_steps_total3 = steps_yearly3 * hist_horizon3

#Setting simulation parameters for esg
horizon3 = 10
steps_total3 = steps_yearly3 * horizon3

#Intialising keys
model_go3 = False
liability_go3 = False
record3 = False

#Requires input from user as to when to begin the third COVID-robust model, and whether to save the fan chart produced
while model_go3 != True:
    print(f"\nPlease enter 1 if you wish to begin running the third COVID-robust model ({firstyear3} to {lastyear3})")
    if input() == "1": 
        model_go3 = True
    else:
        print("\nModel not begun")
print("\nPlease enter 1 if you wish to save the fan chart to a file labelled: 'robust3_fan_chart.png', \notherwise it will not be saved")
if input() == "1":
    print("\nFan chart will be saved")
    save_path_direction3="robust3_fan_chart.png"
else:
    print("\nFan chart will not be saved")
    save_path_direction3 = None

#Main body of code
if model_go3 is True:
    dt_monthly3 = horizon3 / steps_total3
    dt_hist3 = hist_horizon3 / hist_steps_total3
    #Intialising dates
    start3 = datetime.datetime(firstyear3, 1, 1)
    end3 = datetime.datetime(lastyear3, 1, 1)

    #Raw series data collection
    rates_raw = web.DataReader('DGS10', 'fred', start3, end3) / 100 
    cpi_raw = web.DataReader('CPIAUCSL', 'fred', start3, end3)     
    equity_raw = yf.download('SPY', start=start3, end=end3)['Close']   
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
    esg_inputs = hist_params(hist_rates, hist_inflation, hist_equity, dt_hist=dt_hist3)
    print("\nCalibrated parameters for each model:")
    print("Rates:    ", esg_inputs["rate_params"])
    print("Inflation:", esg_inputs["inflation_params"])
    print("Equity:   ", esg_inputs["equity_params"])
    print("\nCorrelation matrix [rates, inflation, equity]:")
    print(np.round(esg_inputs["corr_matrix"], 3))
 
    #Simulating the esg forwards across 10000 paths monthly for ten years using the found parameters
    results = run_esg(T=horizon3, steps=steps_total3, paths=10000, corr_matrix=esg_inputs["corr_matrix"], rate_params=esg_inputs["rate_params"], inflation_params=esg_inputs["inflation_params"], equity_params=esg_inputs["equity_params"]) 
    rates, inflation, equity = results["rates"], results["inflation"], results["equity"]
    
    #Plotting the fan charts for the esg results
    fan_chart = plot_esg_fan_charts(results, T=horizon3, subtitle = f"COVID-robust ESG ({firstyear3} to {lastyear3}): Simulated Paths", save_path=save_path_direction3)
    
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