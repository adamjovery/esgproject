#################################################################################################################################################################
#Economic Scenario Generator 2.0.
#ESG using a GBM with GARCH for the volatility of equity returns, Vasicek for interest rates, and Ornstein and Uhlenbeck for inflation.
#Option to 1) Run a pre-specified simple ESG with liability calculator
#          2) Run a range of 4 ESG's with different windows, showing their robustness to COVID, and choose advanced settings for these ESG's. 
#################################################################################################################################################################
#Initialisation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import yfinance as yf
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm

np.set_printoptions(legacy='1.25')

horizon = 10
steps = 12
dt = 1 / steps
paths = 10000
covid_key = False

#################################################################################################################################################################
#Modelling functions

def vasicek(r0, a, b, sigma, dt, steps, paths, z):
    ir = np.zeros((steps + 1, paths))
    ir[0] = r0
    for t in range(1, steps + 1):
        ir[t] = (ir[t-1] + a * (b - ir[t-1]) * dt + sigma * np.sqrt(dt) * z[t-1])
        
    return ir

def gbm(e0, mu, sigma, dt, steps, paths, z):
    log_eq = np.zeros((steps + 1, paths))
    log_eq[0] = np.log(e0) 
    for t in range(1, steps + 1):
        log_eq[t] = (log_eq[t-1] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[t-1])
        
    return np.exp(log_eq)

def ornstein(i0, a, b, sigma, dt, steps, paths, z):
    inf = np.zeros((steps + 1, paths))
    inf[0] = i0
    for t in range(1, steps + 1):
        inf[t] = (inf[t-1] + a * (b - inf[t-1]) * dt + sigma * np.sqrt(dt) * z[t-1])
        
    return inf

#################################################################################################################################################################
#GARCH functions

def gbm_with_garch(e0, mu, garch_res, dt, steps, paths, z, seed=None):
    log_eq = np.zeros((steps + 1, paths))
    log_eq[0] = np.log(e0)
    days = 21
    dayt = dt/days
    sub = 1 / (1e4 * dayt)

    rand = np.random.default_rng(seed)
    alpha, beta = garch_res.params["alpha[1]"], garch_res.params["beta[1]"]
    omega = garch_res.params["omega"] * sub
    last_eps = garch_res.resid.iloc[-1]**2 * sub
    last_var = garch_res.conditional_volatility.iloc[-1]**2 * sub
    sto_sigma = np.sqrt(np.full(paths, omega + alpha * last_eps + beta * last_var))
    vol_draw = garch_draw(garch_res, rand)
    
    for t in range(1, steps + 1):
        month_eq = np.zeros(paths)
        w = vol_draw((days, paths))
        zdays = z[t-1] / np.sqrt(days) + (w - w.mean(axis=0))
        for i in range(days):
            month_eq += (mu - 0.5 * sto_sigma**2) * dayt + sto_sigma * np.sqrt(dayt) * zdays[i]
            sto_sigma = np.minimum(np.sqrt(omega + alpha * (sto_sigma * zdays[i])**2 + beta * sto_sigma**2), 1.5)
        log_eq[t] = (log_eq[t-1] + month_eq) 

    return np.exp(log_eq)
    
def garch_draw(garch_res, rand, zmax=6.0):
    dist = garch_res.model.distribution
    name = dist.name.lower()
    if "skew" in name:
        dist_params = np.array([garch_res.params["eta"], garch_res.params["lambda"]])
    elif "student" in name:
        dist_params = np.array([garch_res.params["nu"]])
    else:
        return lambda size: rand.standard_normal(size)

    grid_1 = norm.cdf(np.linspace(-7, 7, 4001))
    grid_2 = dist.ppf(grid_1, dist_params)

    n = 10**6
    ref = np.clip(np.interp((np.arange(n) + 0.5) / n, grid_1, grid_2), -zmax, zmax)
    m, s = ref.mean(), ref.std()

    def vol_draw(size):
        w = np.clip(np.interp(rand.random(size), grid_1, grid_2), -zmax, zmax)
        return (w - m) / s
        
    return vol_draw 

class GarchError(ValueError):
    pass

def check_garch(garch_res):
    a, b, w = garch_res.params["alpha[1]"], garch_res.params["beta[1]"], garch_res.params["omega"]
    if w <= 0 or a < 0 or b < 0:
        raise GarchError("\nThe GARCH has invalid parameters")
    if a + b >= 1:
        raise GarchError(f"\nThe GARCH is non-stationary (alpha+beta = {a+b:.4f})")
    if a + b > 0.999:
        print(f"\nThe GARCH persistence: {a+b:.4f} is very close to 1")

#################################################################################################################################################################
#Calibration functions

def calibrate_mr(series, dt, name=None):
    x = np.asarray(series)
    x_lag = x[:-1]
    x_now = x[1:]
    y = x_now - x_lag
    X = sm.add_constant(x_lag)
    model = sm.OLS(y, X).fit()
    
    alpha, beta = model.params
    a = -beta / dt
    if not (0 < a * dt < 1):
        raise ValueError(f"{name}: a*dt = {a*dt:.3f} is outside (0, 1)")
    b = alpha / (a * dt)
    resid_std = np.std(model.resid, ddof=2)
    sigma = resid_std / np.sqrt(dt)
    
    return {"a": a, "b": b, "sigma": sigma, "resid": model.resid}
 
def calibrate_gbm(eq_series, dt):
    prices = np.asarray(eq_series)
    log_returns = np.diff(np.log(prices))
    sigma = np.std(log_returns, ddof=1) / np.sqrt(dt)
    mean_log_returns = np.mean(log_returns)
    mu = mean_log_returns / dt + 0.5 * sigma**2

    return {"mu": mu, "sigma": sigma}

def calibrate_corr(ir_resid, inf_resid, eq_series):
    diff_log_eq = np.diff(np.log(np.asarray(eq_series)))
    stack_data = np.vstack([np.asarray(ir_resid), np.asarray(inf_resid), diff_log_eq])
    corr_matrix = np.corrcoef(stack_data)

    return corr_matrix

#################################################################################################################################################################
#Correlated shocks function

def corr_shocks(corr_matrix, steps, paths, seed=None):
    rho = corr_matrix.shape[0]
    cho = np.linalg.cholesky(corr_matrix)
    rand = np.random.default_rng(seed)
    ind_z = rand.standard_normal((rho, steps, paths))
    corr_z = np.einsum('ij,jkl->ikl', cho, ind_z)
   
    return corr_z

#################################################################################################################################################################
#Diagram functions

def fan_chart(paths, T, ax, title, ylabel, as_percent=False):
    steps = paths.shape[0] - 1
    x = np.linspace(0, T, steps + 1)
    p5 = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    
    ax.fill_between(x, p5, p95, color='forestgreen', alpha=0.2, label='5th-95th percentile')
    ax.fill_between(x, p25, p75, color='forestgreen', alpha=0.4, label='25th-75th percentile')
    ax.plot(x, p50, color='darkgreen', linewidth=2, label='Median')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Years')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.25)
    if as_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
 
def esg_fan_charts(results, T, chart_title, dist_type, subtitle=None, save_path=None, first_year=None, last_year=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fan_chart(results["ir"], T, axes[0], "Interest Rate (Vasicek)", "Interest rate", as_percent=True)
    fan_chart(results["inf"], T, axes[1], "Inflation Rate (Ornstein-Uhlenbeck)", "Inflation rate", as_percent=True)
    if dist_type != None:
        fan_chart(results["eq"], T, axes[2], f"Equity Level (log-GBM with GARCH [{dist_type}])", "Index level", as_percent=False)
    else:
        fan_chart(results["eq"], T, axes[2], f"Equity Level (log-GBM [Constant Vol.])", "Index level", as_percent=False)
    
    fig.suptitle(subtitle, fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path + str(first_year) + "_to_" + str(last_year) + ".png", dpi=150, bbox_inches='tight')
    else:
        pass
    return {chart_title: fig}

def liability_histogram(path_pv, deter_pv, flat_deter_pv, metrics, confidence, paths, first_year=None, last_year=None, save_histogram=None):
    conf = round(confidence*100)
    hist, ax = plt.subplots(figsize=(10, 6))
    ax.hist(path_pv, bins=80, color='forestgreen', alpha=0.7, edgecolor='white')
    ax.axvline(metrics['mean'], color='darkgreen', linewidth=2, linestyle='-', label=f"Stochastic mean PV = {metrics['mean']:.2f}")
    ax.axvline(deter_pv, color='darkslategrey', linewidth=2, linestyle='--', label=f"Deterministic PV = {deter_pv:.2f}")
    ax.axvline(flat_deter_pv, color='lightslategrey', linewidth=2, linestyle='--', label=f"Alt. Deterministic PV = {flat_deter_pv:.2f}")
    ax.axvline(metrics[f'VaR_{conf}'], color='turquoise', linewidth=2, linestyle=':', label=f"VaR_{conf} = {metrics[f'VaR_{conf}']:.2f}")
    ax.axvline(metrics[f'TVaR_{conf}'], color='teal', linewidth=2, linestyle=':', label=f"TVaR_{conf} = {metrics[f'TVaR_{conf}']:.2f}")
    ax.set_title(f"Distribution of liability present value across {paths} paths ({first_year} to {last_year})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Present value")
    ax.set_ylabel("Number of paths")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    hist.tight_layout()
    if save_histogram:
        hist.savefig(save_histogram + str(first_year) + "_to_" + str(last_year) + ".png", dpi=150, bbox_inches='tight')
    else:
        pass
    return {"hist_returned": hist} 

#################################################################################################################################################################
#Liability calculator functions

def dis_rate(ir_paths, dt):
    integral = np.zeros_like(ir_paths)
    integral[1:] = np.cumsum((ir_paths[:-1] + ir_paths[1:]) / 2 * dt, axis=0)
    
    return np.exp(-integral)
 
def esc_rate(inf_paths, dt):
    integral = np.zeros_like(inf_paths)
    integral[1:] = np.cumsum((inf_paths[:-1] + inf_paths[1:]) / 2 * dt, axis=0)
    
    return np.exp(integral)

def liability_value(dt, ir_paths, inf_paths, base_pay, T, steps, inf_linked):
    liability_paths = ir_paths.shape[1]
    dis_values = dis_rate(ir_paths, dt)
    esc_values = esc_rate(inf_paths, dt) if inf_linked else np.ones_like(inf_paths)
    
    path_pv = np.zeros(liability_paths)
    pay_ind = [i * steps for i in range(1, T + 1)]
    for i in pay_ind:
        nom_pay = base_pay * esc_values[i]
        path_pv += nom_pay * dis_values[i]

    return path_pv

def liability_metrics(path_pv, confidence):
    mean_pv = path_pv.mean()
    median_pv = np.median(path_pv)
    std_pv = path_pv.std()
    
    var_level = np.percentile(path_pv, confidence * 100)
    tail_pv = path_pv[path_pv >= var_level]
    tvar_level = tail_pv.mean()
    
    return {"mean": mean_pv, "median": median_pv, "std": std_pv, f"VaR_{round(confidence*100)}": var_level, f"TVaR_{round(confidence*100)}": tvar_level, "5th percentile": np.percentile(path_pv, 5), "95th percentile": np.percentile(path_pv, 95)}
 
def flat_det_pv(base_pay, det_dis, det_esc, T, inf_linked):
    flat_pv = 0.0
    for i in range(1, T + 1):
        nom_pay = base_pay * (np.exp(det_esc * i) if inf_linked else 1.0)
        flat_pv += nom_pay * np.exp(-det_dis * i)

    return flat_pv

def det_pv(dt, steps, horizon, base_pay, ir_params, inf_params, inf_linked):
    n = steps * horizon
    z0 = np.zeros((n, 1))
    ir = vasicek(**{**ir_params, "sigma": 0.0}, dt=dt, steps=n, paths=1, z=z0)
    inf = ornstein(**{**inf_params, "sigma": 0.0}, dt=dt, steps=n, paths=1, z=z0)
    pv = liability_value(dt, ir, inf, base_pay, horizon, steps, inf_linked)[0]
    
    return pv

def liability_sum_stats(metrics, horizon, paths, deter_pv, flat_deter_pv, confidence, first_year, last_year):
    print(f"\nStochastic liability present valuation for {horizon} years across {paths} paths ({first_year} to {last_year})")
    for i, j in metrics.items():
        print(f"{i:10s}: {j:8.2f}")    
    conf = round(confidence*100)
    var_conf = f"VaR_{conf}"
    tvar_conf = f"TVaR_{conf}"
    print(f"\nDeterministic PV (long-run b assumptions, starting from current discount and escalation rates):  {deter_pv:.2f}")
    print(f"\nAlt. Deterministic PV (flat long-run b assumptions):  {flat_deter_pv:.2f}")
    print(f"Stochastic mean PV:  {metrics['mean']:.2f}")
    print(f"Difference (stochastic - deterministic):  {metrics['mean'] - deter_pv:.2f}")
    print(f"\nRequired capital buffer (VaR_{conf} - mean):  {metrics[var_conf] - metrics['mean']:.2f}")
    print(f"Required capital buffer (TVaR_{conf} - mean):  {metrics[tvar_conf] - metrics['mean']:.2f}")

#################################################################################################################################################################
#Historical data functions

def hist_col(first_year, last_year):
    start = datetime.datetime(first_year, 1, 1)
    end = datetime.datetime(last_year, 1, 1)
    ir_raw = web.DataReader('DGS10', 'fred', start, end) / 100 
    cpi_raw = web.DataReader('CPIAUCSL', 'fred', start, end)     
    eq_raw = yf.download('SPY', start, end, progress=False)['Close'] 
    eq_raw.columns = eq_raw.columns.get_level_values(0)
    
    ir_m = ir_raw.resample('MS').last()
    cpi_m = cpi_raw.resample('MS').last()
    inf_m = 12 * np.log(cpi_m).diff()
    yoy_cpi = cpi_m.pct_change(12, fill_method = None)
    eq_m = eq_raw.resample('MS').last()
    com = pd.concat([ir_m, inf_m, eq_m], axis=1, keys=['ir', 'inf', 'eq']).dropna()

    cur_inf = float(np.log1p(yoy_cpi.iloc[:, 0].loc[com.index[-1]]))
    hist_ir = com['ir'].values.flatten()
    hist_inf = com['inf'].values.flatten()
    hist_eq = com['eq'].values.flatten()
    log_returns = 100 * np.log(eq_raw / eq_raw.shift(1)).dropna()
    
    return {"hist_ir": hist_ir, "hist_inf": hist_inf, "hist_eq": hist_eq, "log_returns": log_returns, "cur_inf": cur_inf}

def hist_params(hist_ir, hist_inf, hist_eq, dt, cur_inf):
    ir_fit = calibrate_mr(hist_ir, dt, name="interest rate")
    inf_fit = calibrate_mr(hist_inf, dt, name="inflation")
    eq_fit = calibrate_gbm(hist_eq, dt)
    corr_matrix = calibrate_corr(ir_fit["resid"], inf_fit["resid"], hist_eq)
    
    ir_params = dict(r0=hist_ir[-1], a=ir_fit["a"], b=ir_fit["b"], sigma=ir_fit["sigma"])
    inf_params = dict(i0=cur_inf, a=inf_fit["a"], b=inf_fit["b"], sigma=inf_fit["sigma"])
    eq_params = dict(e0=hist_eq[-1], mu=eq_fit["mu"], sigma=eq_fit["sigma"])
    
    return {"corr_matrix": corr_matrix, "ir_params": ir_params, "inf_params": inf_params, "eq_params": eq_params}
    
#################################################################################################################################################################
#ESG functions

def run_esg(dt, steps, paths, corr_matrix, ir_params, inf_params, eq_params, garch_res=None, garch_key=None, seed=None):
    seed_corr, seed_eq = np.random.SeedSequence(seed).spawn(2)
    z = corr_shocks(corr_matrix, steps, paths, seed_corr)
    ir = vasicek(**ir_params, dt=dt, steps=steps, paths=paths, z=z[0])
    inf = ornstein(**inf_params, dt=dt, steps=steps, paths=paths, z=z[1])
    if garch_key == True and garch_res is not None:
        eq_garch_params = {i: j for i, j in eq_params.items() if i != "sigma"}
        eq = gbm_with_garch(**eq_garch_params, garch_res=garch_res, dt=dt, steps=steps, paths=paths, z=z[2], seed=seed_eq)
    else:
        eq = gbm(**eq_params, dt=dt, steps=steps, paths=paths, z=z[2])
    return {"ir": ir, "inf": inf, "eq": eq}

def sum_stats(ir, inf, eq, ir_b, inf_b, eq_0, mu, first_year, last_year, horizon, paths, esg_inputs, dt):
    print("\n#################################################################################################################################################################")
    print(f"\nSummary of ESG in year {horizon} across {paths} paths for {first_year} to {last_year} (calibrations in brackets):")
    print(f"Mean interest rate:  {ir[-1].mean():.4f}  (calibrated target b: {ir_b:.4f})")
    print(f"Mean inflation rate:  {inf[-1].mean():.4f}  (calibrated target b: {inf_b:.4f})")
    print(f"Mean equity level:  {eq[-1].mean():.2f}  (calibrated expected value of equity: {eq_0*np.exp(mu*horizon):.2f})")

    def innovations(x, a, b, dt):
        return x[1:] - x[:-1] - a * (b - x[:-1]) * dt

    ir_inn = innovations(ir, esg_inputs["ir_params"]["a"], esg_inputs["ir_params"]["b"], dt)
    inf_inn = innovations(inf, esg_inputs["inf_params"]["a"], esg_inputs["inf_params"]["b"], dt)
    log_eq = np.log(eq)
    check_inn = np.isfinite(log_eq).all(axis=0)
    eq_inn = np.diff(log_eq[:, check_inn], axis=0)

    corr_rate_inf = np.corrcoef(ir_inn.flatten(), inf_inn.flatten())[0, 1]
    corr_rate_eq = np.corrcoef(ir_inn[:, check_inn].flatten(), eq_inn.flatten())[0, 1]
    target_rate_inf = esg_inputs["corr_matrix"][0, 1]
    target_rate_eq = esg_inputs["corr_matrix"][0, 2]

    print("\nRealised correlations (calibrations in brackets)")
    print(f"Interest rates vs inflation rates:  {corr_rate_inf:.3f}  (calibrated target: {target_rate_inf:.3f})")
    print(f"Interest rates vs equity levels:  {corr_rate_eq:.3f}  (calibrated target: {target_rate_eq:.3f})")
    
#################################################################################################################################################################
#User interface

while covid_key != True:
    print("\nPlease enter 1 to run a range of ESG's data windows with custom parameters and COVID comparisons for robustness, or enter 2 to run a single sample ESG with pre-determined specification \nand liability calculator")
    response = input()
    if response == "1":
        covid_robust = True
        covid_key = True
        liability_key = False
        payment_key = False
        confidence_key = False
        garch_key = False
        dist_key = False
        print("\nPlease enter 1 if you want to use GARCH to model the volatility in the GBM, \notherwise the volatility will be modelled as a constant")
        if input() == "1":
            print("\nGARCH will be used to model volatility in the GBM")
            garch_key = True
            while dist_key != True:
                print("\nPlease enter 1 if you want to use a normal distribution in the GARCH, enter 2 if you want to use Student's t distribution, and enter 3 if you want to use a \nskewed Student's t distribution")
                dist_resp = input()
                if dist_resp == "1":
                    dist_type = "normal"
                    dist_key = True
                    print("\nA normal distribution will be used in the GARCH")
                elif dist_resp == "2":
                    dist_type = "t"
                    dist_key = True
                    print("\nA Student's t distribution will be used in the GARCH")
                elif dist_resp == "3":
                    dist_type = "skewt"
                    dist_key = True
                    print("\nA skewed Student's t distribution will be used in the GARCH")
                else:
                    print("\nError please try again")
        else: 
            print("\nVolatility will be modelled as a constant in the GBM")
            dist_type = None
        print("\nPlease enter 1 if you want to perform liability calculations")
        if input() == "1":
            print("\nLiability calculations will be performed")
            liability_key = True
            print("\nPlease enter 1 if you want the ESG & liability summary statistics")
            if input () == "1":
                print("\nESG & liability summary statistics will be shown")
                sum_stats_key = True
            else:
                print("\nESG & liability summary statistics will not be shown")
                sum_stats_key = False
            while payment_key != True:
                print("\nPlease enter the current annual payment value of your liability in integer or decimal form")
                base_pay = input()
                error = 0
                try:
                    float(base_pay)
                except ValueError:
                    error = 1
                if error == 0:
                    base_pay = float(base_pay)
                    payment_key = True
                else:
                    print("\nError please try again")
            while confidence_key != True:
                print("\nPlease enter the desired confidence interval for the VaR and TVaR in decimal form, \ngreater than 0 and less than 1")
                error = 0
                confidence = input()
                try:
                    float(confidence)
                except ValueError:
                    error = 1
                if error == 0 and 1 > float(confidence) > 0:
                    confidence = float(confidence)
                    confidence_key = True
                else:
                    print("\nError please try again")
            print("\nPlease enter 1 if you want the liability to be inflation linked")
            if input() == "1":
                print("\nLiability recorded as inflation linked")
                inf_link = True
            else:
                print("\nLiability recorded as not inflation linked")
                inf_link = False
            print("\nPlease enter 1 if you wish to save the fan charts to a file labelled: 'covid_robust_fan_charts_year.png' & the liability PV histograms to a file labelled: 'covid_robust_histograms_year.png' \notherwise they will not be saved")
            if input() == "1":
                print("\nDiagrams will be saved")
                save_histogram = "covid_robust_histograms_"
                save_fan_chart = "covid_robust_fan_charts_"
            else:
                print("\nDiagrams will not be saved")
                save_histogram = None
                save_fan_chart = None
        else:
            print("\nLiability calculations will not be performed")
            print("\nPlease enter 1 if you want the ESG summary statistics")
            if input () == "1":
                print("\nESG summary statistics will be shown")
                sum_stats_key = True
            else:
                print("\nESG summary statistics will not be shown")
                sum_stats_key = False
            print("\nPlease enter 1 if you wish to save the fan charts to a file labelled: 'covid_robust_fan_charts_year.png', \notherwise it will not be saved")
            if input() == "1":
                print("\nFan chart will be saved")
                save_fan_chart = "covid_robust_fan_charts_"
            else:
                print("\nFan chart will not be saved")
                save_fan_chart = None
    elif response == "2":
        covid_robust = False
        covid_key = True
        inf_link = True
        garch_key = True

        first_year = 2006
        last_year = 2026
        base_pay = 1000
        confidence = 0.95
        seed = 1000000
        save_histogram = 0
        dist_type = "normal"

        print("\nPlease enter 1 if you wish to save the fan chart to a file labelled: 'simple_fan_chart_year.png' & the liability PV histogram \nto a file labelled: 'simple_histogram_year.png', otherwise they will not be saved")
        if input() == "1":
            print("\nDiagrams will be saved")
            save_histogram = "simple_histogram_"
            save_fan_chart = "simple_fan_chart_"
        else:
            print("\nDiagrams will not be saved")
            save_histogram = None
            save_fan_chart = None
        print(f"\nRunning pre-specified ESG using GARCH (Normal distribution) in a GBM for equity levels, Ornstein-Uhlenbeck for inflation, and Vasicek for interest rates. Based on {first_year} to {last_year} data, \nwith the liability calculator having an annual base liability of {base_pay} currency units, a confidence interval of {confidence}, and the liability being inflation linked")
    else:
        print("\nError please try again")

#################################################################################################################################################################
#Main
        
if covid_robust == True:
    first_years = ["1999", "2009", "2006", "2016"]
    last_years = ["2019", "2019", "2026", "2026"]
    
    for i, j in zip(first_years, last_years):
        try:
            hist_inputs = hist_col(int(i), int(j))
            esg_inputs = hist_params(hist_inputs["hist_ir"], hist_inputs["hist_inf"], hist_inputs["hist_eq"], dt, hist_inputs["cur_inf"])
        
            if garch_key == True:
                garch = arch_model(hist_inputs["log_returns"], mean="Constant", vol="GARCH", p=1, q=1, dist=dist_type)
                garch_res = garch.fit(disp="off")
                check_garch(garch_res)
                results = run_esg(dt, steps*horizon, paths, esg_inputs["corr_matrix"], esg_inputs["ir_params"], esg_inputs["inf_params"], esg_inputs["eq_params"], garch_res, garch_key)
            else:
                results = run_esg(dt, steps*horizon, paths, esg_inputs["corr_matrix"], esg_inputs["ir_params"], esg_inputs["inf_params"], esg_inputs["eq_params"])
            if sum_stats_key == True:
                sum_stats(results["ir"], results["inf"], results["eq"], esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], esg_inputs["eq_params"]["e0"], esg_inputs["eq_params"]["mu"], int(i), int(j), horizon, paths, esg_inputs, dt)
            else:
                pass
            fan_chart_created = esg_fan_charts(results, horizon, f"chart_{int(i)}", dist_type, subtitle=f"Economic Scenario Generator ({int(i)} to {int(j)}): Simulated Paths", save_path=save_fan_chart, first_year=int(i), last_year=int(j))
            plt.show()
            plt.close(fan_chart_created[f"chart_{int(i)}"])
        
            if liability_key == True:
                path_pv = liability_value(dt, results["ir"], results["inf"], base_pay, horizon, steps, inf_link)
                metrics = liability_metrics(path_pv, confidence)
                deter_pv = det_pv(dt, steps, horizon, base_pay, esg_inputs["ir_params"], esg_inputs["inf_params"], inf_link)
                flat_deter_pv = flat_det_pv(base_pay, esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], horizon, inf_link)
                if sum_stats_key == True:
                    liability_sum_stats(metrics, horizon, paths, deter_pv, flat_deter_pv, confidence, int(i), int(j))
                else: 
                    pass
                histogram_created = liability_histogram(path_pv, deter_pv, flat_deter_pv, metrics, confidence, paths, int(i), int(j), save_histogram)
                plt.show()
                plt.close(histogram_created["hist_returned"])
            else:
                pass
        except GarchError as err:
            if dist_type != "normal" and dist_type != None:
                print(err)
                print ("Therefore, changing the GARCH distribution to a normal distribution just for this window")
                dist_type_changed = "normal"
                garch = arch_model(hist_inputs["log_returns"], mean="Constant", vol="GARCH", p=1, q=1, dist=dist_type_changed)
                garch_res = garch.fit(disp="off")
                check_garch(garch_res)
                results = run_esg(dt, steps*horizon, paths, esg_inputs["corr_matrix"], esg_inputs["ir_params"], esg_inputs["inf_params"], esg_inputs["eq_params"], garch_res, garch_key)
                if sum_stats_key == True:
                    sum_stats(results["ir"], results["inf"], results["eq"], esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], esg_inputs["eq_params"]["e0"], esg_inputs["eq_params"]["mu"], int(i), int(j), horizon, paths, esg_inputs, dt)
                else:
                    pass
                fan_chart_created = esg_fan_charts(results, horizon, f"chart_{int(i)}", dist_type_changed, subtitle=f"Economic Scenario Generator ({int(i)} to {int(j)}): Simulated Paths", save_path=save_fan_chart, first_year=int(i), last_year=int(j))
                plt.show()
                plt.close(fan_chart_created[f"chart_{int(i)}"])
        
                if liability_key == True:
                    path_pv = liability_value(dt, results["ir"], results["inf"], base_pay, horizon, steps, inf_link)
                    metrics = liability_metrics(path_pv, confidence)
                    deter_pv = det_pv(dt, steps, horizon, base_pay, esg_inputs["ir_params"], esg_inputs["inf_params"], inf_link)
                    flat_deter_pv = flat_det_pv(base_pay, esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], horizon, inf_link)
                    if sum_stats_key == True:
                        liability_sum_stats(metrics, horizon, paths, deter_pv, flat_deter_pv, confidence, int(i), int(j))
                    else: 
                        pass
                    histogram_created = liability_histogram(path_pv, deter_pv, flat_deter_pv, metrics, confidence, paths, int(i), int(j), save_histogram)
                    plt.show()
                    plt.close(histogram_created["hist_returned"])
                else:
                    pass
            else:
                print(f"\nSkipping {i} to {j}: {err}")   
                plt.close('all')
            continue
        except ValueError as err:
            print(f"\n Therefore, skipping {i} to {j}: {err}")   
            plt.close('all')
            continue
else:
    hist_inputs = hist_col(first_year, last_year)
    garch = arch_model(hist_inputs["log_returns"], mean="Constant", vol="GARCH", p=1, q=1, dist=dist_type)
    garch_res = garch.fit(disp="off")
    check_garch(garch_res)
    
    esg_inputs = hist_params(hist_inputs["hist_ir"], hist_inputs["hist_inf"], hist_inputs["hist_eq"], dt, hist_inputs["cur_inf"])
    results = run_esg(dt, steps*horizon, paths, esg_inputs["corr_matrix"], esg_inputs["ir_params"], esg_inputs["inf_params"], esg_inputs["eq_params"], garch_res, garch_key, seed=seed)
    
    sum_stats(results["ir"], results["inf"], results["eq"], esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], esg_inputs["eq_params"]["e0"], esg_inputs["eq_params"]["mu"], first_year, last_year, horizon, paths, esg_inputs, dt)
    fan_chart_created = esg_fan_charts(results, horizon, "overall_chart", dist_type, subtitle=f"Economic Scenario Generator: Simulated Paths ({first_year} to {last_year})", save_path=save_fan_chart, first_year=first_year, last_year=last_year)
    plt.show()
    plt.close(fan_chart_created["overall_chart"])

    path_pv = liability_value(dt, results["ir"], results["inf"], base_pay, horizon, steps, inf_link)
    metrics = liability_metrics(path_pv, confidence)
    deter_pv = det_pv(dt, steps, horizon, base_pay, esg_inputs["ir_params"], esg_inputs["inf_params"], inf_link)
    flat_deter_pv = flat_det_pv(base_pay, esg_inputs["ir_params"]["b"], esg_inputs["inf_params"]["b"], horizon, inf_link)
    liability_sum_stats(metrics, horizon, paths, deter_pv, flat_deter_pv, confidence, first_year, last_year)
    liability_histogram(path_pv, deter_pv, flat_deter_pv, metrics, confidence, paths, first_year, last_year, save_histogram)
    plt.show()
    plt.close("all")
