import numpy as np
#Function to convert the short-term interest rate paths from the esg into stochastic discount factors 
def discount_factors(rate_paths, dt):
    #Intialising integral array of zeros to the same shape as the rate_paths array
    integral = np.zeros_like(rate_paths)

    #The negative of the integral (approximated via the trapezoidal rule) of the area under the rate path from time 0 to time t then put to the e gives us the array of discount factors
    integral[1:] = np.cumsum((rate_paths[:-1] + rate_paths[1:]) / 2 * dt, axis=0)
    return np.exp(-integral)
 
#Function to convert the inflation paths from the esg into a cumulative escalation factor
def escalation_factors(inflation_paths, dt):
    #Intialising integral array of zeros to the same shape as the inflation_paths array
    integral = np.zeros_like(inflation_paths)
    
    #The integral (approximated via the trapezoidal rule) of the area under the inflation path from time 0 to time t then put to the e gives us the array of esclation factors 
    integral[1:] = np.cumsum((inflation_paths[:-1] + inflation_paths[1:]) / 2 * dt, axis=0)
    return np.exp(integral)

#Function to calculate the stochastic present value of annual liability across the esg paths
def value_liability(rates, inflation, base_annual_payment, T, steps_per_year, inflation_linked):
    dt = 1 / steps_per_year
    #Finds the number of columns in the rates array and stores that in paths
    paths = rates.shape[1]

    #Calling the functions which calculate the arrays of discount factors and escalation factors, with the escalation factors being calculated only if the payments are inflation linked, otherwise an array in the same shape of all ones will be created instead
    discount_factor_values = discount_factors(rates, dt)
    escalation_factor_values = escalation_factors(inflation, dt) if inflation_linked else np.ones_like(rates)

    #Intialising present value path array and calculating end-of-year real payment index
    path_pv = np.zeros(paths)
    payment_indices = [i * steps_per_year for i in range(1, T + 1)]

    #Using end-of-year payment index to find the nominal payment in each year and add this to the present value path array
    for i in payment_indices:
        nominal_payment = base_annual_payment * escalation_factor_values[i]
        path_pv += nominal_payment * discount_factor_values[i]

    #Returns the present values of the paths
    return path_pv
 
#Function to summarise the stochastic present value distribution through the mean, percentiles, VaR, and TVaR
def liability_metrics(path_pv, confidence):
    #Calculating the mean, median, and standard deviation of the present values of the paths
    mean_pv = path_pv.mean()
    median_pv = np.median(path_pv)
    std_pv = path_pv.std()

    #Looks at the worst outcome, the value at risk, at a given confidence level
    var_level = np.percentile(path_pv, confidence * 100)

    #Finds each present value which is at least as bad as the VaR and also calculates their mean
    tail_pv = path_pv[path_pv >= var_level]
    tvar_level = tail_pv.mean()

    #Returns the calculated the metrics for the liability based upon the esg results
    return {"mean": mean_pv, "median": median_pv, "std": std_pv, f"VaR_{int(confidence*100)}": var_level, f"TVaR_{int(confidence*100)}": tvar_level, "5th percentile": np.percentile(path_pv, 5), "95th percentile": np.percentile(path_pv, 95)}
 
#Function to calculate a deterministic present value for comparison with the stochastic present value calculation
def deterministic_pv(base_annual_payment, det_discount_rate, det_escalation_rate, T, inflation_linked):
    pv = 0.0
    for i in range(1, T + 1):
        #Escalates the liability value by a deterministic inflation rate if the liability is inflation linked, otherwise it stays constant
        nominal_payment = base_annual_payment * ((1 + det_escalation_rate) ** i if inflation_linked else 1.0)
        
        #Uses the deterministic interest rate to discount the nominal yearly payment
        pv += nominal_payment / ((1 + det_discount_rate) ** i)

    #Return the deterministic present value
    return pv