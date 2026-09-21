# esgproject
Python files which simulate an ESG, create a fan chart of the results, and then offers stochastic present value liability discounting and escalation, which provides a mean, median, VaR, and TVaR, as well as providing a comparison to a deterministic present value and creating a histogram of these results.

Install all of the ".py" files and run "simulate.py" to replicate the results of this model, which are displayed in "esg_fan_chart.png" and "liabilities_pv_histogram.png". The liability calculation for these results uses an inflation linked annuity of 1000 units of currency, with a confidence interval of 0.95.

This Economic Scenario Generator (ESG) which is built in Python follows three models across 10 years and 10,000 paths.
The three models which are linked together by their correlated shocks are as follows:
  1. Vasicek model to simulate the path of short-term interest rates - assuming they follow a mean reverting stochastic process.
  2. Geometric Brownian Motion model to simulate the path of equity price levels - assuming they follow a stochastic process with their logarithm following a Brownian motion with drift.
  3. Ornstein-Uhlenbeck model to simulate the path of inflation rates - assuming they follow a mean reverting stochastic process.

This is done by collecting the series data from FRED and Yahoo finance on the 10-Year Treasury yield, CPI Year on Year percentage change, and the SPY ETF close. This data is then cleaned and the parameters of the models built from this data via OLS for the mean reverting models, and through expected returns for the GBM. The correlation matrix for these models is also found through the Pearson correlation coefficients of the changes/log-returns within the series data.

The correlated standard normal shocks are then found via a Cholesky decomposition which is performed on the correlation matrix, with the result of this being multiplied by the shock dimension of an array of randomly generated shocks from a normal distribution. 

The found parameters and correlated shocks are then fed into the models, which simulate the next 10 years across 10,000 paths. The mean values of this in year 10 are then displayed, and a fan chart created for each of the variables. 

Following this, a stochastic liability present valuation can then be performed, with the annual value of the liability, the confidence interval for the VaR and TVaR, and whether the liability is inflation linked being inputted by the user. This then results in a range of metrics describing the stochastic liability present valuation, and compares it to a deterministic present valuation based upon the parameters derived from the historical data.

A histogram is then created, mapping the frequency of paths to the stochastic present values, with the stochastic and deterministic means as well as the VaR and TVaR levels also being plotted.
