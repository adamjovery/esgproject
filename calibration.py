import numpy as np
import statsmodels.api as sm
#Function to perform OLS to acquire the parameters for both the Vasicek and Ornstein-Uhlenbeck models
def calibrate_mean_reverting(series, dt):
    #Rearranging variables and generating lagged variable
    x = np.asarray(series)
    x_lag = x[:-1]
    x_now = x[1:]
    y = x_now - x_lag

    #Performing OLS regression and recording alpha and beta
    X = sm.add_constant(x_lag)
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params

    #Extracting parameter values, and calculating sigma from residuals (minus two degrees of freedom in this)
    a = -beta / dt
    b = alpha / (a * dt)
    resid_std = np.std(model.resid, ddof=2)
    sigma = resid_std / np.sqrt(dt)
    
    #Returning parameter values, sigma, and model fit
    return {"a": a, "b": b, "sigma": sigma, "r_squared": model.rsquared}
 
#Function to conduct the calibration of GBM parameters - logged returns as in the model
def calibrate_gbm(price_series, dt):
    #Extracting prices and calculating the logged returns from them
    prices = np.asarray(price_series)
    log_returns = np.diff(np.log(prices))

    #Calculating mean and standard deviation of logged returns
    mean_log_return = np.mean(log_returns)
    std_log_return = np.std(log_returns, ddof=1)

    #Calculating sigma and mu from mean and standard deviation of logged returns
    sigma = std_log_return / np.sqrt(dt)
    mu = mean_log_return / dt + 0.5 * sigma**2

    #Returning mu and sigma
    return {"mu": mu, "sigma": sigma}

#Function to estimate the correlation matrix between the shocks driving each factor using the changes in each series for the mean-reverting models and log-returns for the GBM
def calibrate_correlation_matrix(rate_series, inflation_series, equity_series):
    #Calculating changes/log-returns in the series
    d_rates = np.diff(np.asarray(rate_series))
    d_inflation = np.diff(np.asarray(inflation_series))
    d_log_equity = np.diff(np.log(np.asarray(equity_series)))

    #Stacking the differences and calculating the pearson correlation coefficient
    data = np.vstack([d_rates, d_inflation, d_log_equity])
    corr_matrix = np.corrcoef(data)

    #Returns the calculated 3x3 correlation matrix in the order of rates/inflation/equity
    return corr_matrix
 