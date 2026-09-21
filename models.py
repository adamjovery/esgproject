# Overview of variables within the models
# r0: initial interest rate
# e0: intial equity prices
# i0: initial inflation rate
# a: mean reversion speed
# b: long-run mean
# mu: expected return of equity
# sigma: volatility
# T: horizon in years
# steps: time steps
# paths: number of scenarios

import numpy as np
#Function which simulates the interest rate path under the Vasicek model
#This model assumes interest rates follow a mean reverting stochastic process
def vasicek(r0, a, b, sigma, T, steps, paths, z):
    dt = T / steps

    #Intialising matrix of interest rates
    rates = np.zeros((steps + 1, paths))
    rates[0] = r0

    #Charting path of interest rates
    for t in range(1, steps + 1):
        rates[t] = (rates[t-1] + a * (b - rates[t-1]) * dt + sigma * np.sqrt(dt) * z[t-1])
    
    #Returning the path of the interest rates
    return rates

#Function which simulates the equity price level path following Geometric Brownian Motion
#This model assumes equity prices follow a stochastic process with the logarithm of equity prices following a Brownian motion with drift
def gbm_log(e0, mu, sigma, T, steps, paths, z):
    dt = T / steps

    #Intialising matrix of logged equity prices
    log_equity = np.zeros((steps + 1, paths))
    log_equity[0] = np.log(e0)

    #Charting path of logged equity prices
    for t in range(1, steps + 1):
        log_equity[t] = (log_equity[t-1] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[t-1])

    #Returning the logged equity prices path to the e
    return np.exp(log_equity)

#Function which simulates the path of inflation rates under the Ornstein-Uhlenbeck model
#This model assumes inflation rates follow a mean reverting stochastic process
def ornstein(i0, a, b, sigma, T, steps, paths, z):
    dt = T / steps

    #Intialising matrix of inflation rates
    rates = np.zeros((steps + 1, paths))
    rates[0] = i0

    #Charting path of inflation rates
    for t in range(1, steps + 1):
        rates[t] = (rates[t-1] + a * (b - rates[t-1]) * dt + sigma * np.sqrt(dt) * z[t-1])

    #Returning the path of inflation rates
    return rates