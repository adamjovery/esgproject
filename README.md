# esgproject

Economic Scenario Generator 2.0

ESG using a GBM with GARCH for the volatility of equity returns, Vasicek for interest rates, and Ornstein and Uhlenbeck for inflation.
Option to: 
  1) Run a range of 4 ESG's with different historical data windows, showing their robustness to COVID, and choose advanced settings for these ESG's. 
  2) Run a pre-specified reproducible ESG with liability calculator

The pre-specified reproducible ESG's results are shown in the files: "reproducible_fan_chart_2006_to_2026.png" and in "reproducible_histogram_2006_to_2026.png"
These results are built on a historical data window of 2006 to 2026, a normally distributed GARCH in the GBM, a base annual liability payment of 1000 currency units, a confidence level of 0.95, and an inflation linked liability. 

Sample results of the range of 4 ESG's are shown in the files "window_range_fan_chart_year_range.png" and "window_range_histogram_year_range.png" 
These results are built on historical data windows of 1999 to 2019, 2009 to 2019, 2006 to 2026, and 2016 to 2026; a GARCH with a skewed Student's T distribution in the GBM; a base annual liability payment of 1000 currency units; a confidence level of 0.95; and an inflation linked liability.

This Economic Scenario Generator (ESG) which is built in Python follows three models across 10 years and 10,000 paths. The three models which are linked together by their correlated shocks are as follows:

Vasicek model to simulate the path of interest rates - assuming they follow a mean reverting stochastic process.
Geometric Brownian Motion model with GARCH volatility to simulate the path of equity price levels - assuming they follow a stochastic process with their logarithm following a Brownian motion with drift.
Ornstein-Uhlenbeck model to simulate the path of inflation rates - assuming they follow a mean reverting stochastic process.

This is done by collecting the series data on the chosen window from FRED and Yahoo finance based upon the 10-Year Treasury yield, CPI Year on Year percentage change, and the SPY ETF close. This data is then cleaned and the parameters of the models built from this data via OLS for the mean reverting models, and through expected returns for the GBM. The correlation matrix for these models is also found through the Pearson correlation coefficients of the changes/log-returns within the series data.

The correlated standard normal shocks are then found via a Cholesky decomposition which is performed on the correlation matrix, with the result of this being multiplied by the shock dimension of an array of randomly generated shocks from a normal distribution.

The found parameters and correlated shocks are then fed into the models, which simulate the next 10 years across 10,000 paths. The mean values of this in year 10 are then displayed, and a fan chart created for each of the variables.

Following this, a stochastic liability present valuation can then be performed, with the annual value of the liability, the confidence interval for the VaR and TVaR, and whether the liability is inflation linked being inputted by the user. This then results in a range of metrics describing the stochastic liability present valuation, and compares it to a deterministic present valuation based upon the parameters derived from the historical data.

A histogram is then created, mapping the frequency of paths to the stochastic present values, with the stochastic and deterministic means as well as the VaR and TVaR levels also being plotted.
