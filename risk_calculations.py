#%% import the libraries
import pandas as pd
import numpy as np


#%% import the data
def read_pickle(instrument, startdate, enddate, resample_period):
    path_ = ("/home/jake/Code/kaiseki/local_data/" + instrument + "_pickle")
    df = pd.read_pickle(path_).loc[startdate:enddate].resample(resample_period).ffill()
    return df


class Risk_calc:
    def __init__(self):


def hist_value_at_risk(returns, weights, alpha, lookback):
    sim_returns = returns[-lookback:] * weights
    sim_returns_sum = sim_returns.sum(axis=1)
    return np.percentile(sim_returns_sum, 100 * (1-alpha))

def conditional_value_at_risk(returns, weights, alpha, lookback):
    # Call out to our existing function
    hvar = hist_value_at_risk(returns, weights, alpha, lookback=lookback)
    returns = returns.fillna(0.0)
    sim_returns = returns[-lookback:] * weights
    sim_returns_sum = sim_returns.sum(axis=1)
    return np.nanmean(sim_returns_sum[sim_returns_sum < hvar])

def brexit_stress(returns, weights):
    brexit_rets = pd.DataFrame(index=returns.index)
    brexit_rets = returns['2016-6-24':'2016-6-24']
    brexit_stress_sim = brexit_rets.dot(weights)
    return brexit_stress_sim


#%%
def risk_results(returns, pct_positions):
    return_value = {}
    return_value['HVaR_95_2y'] = hist_value_at_risk(returns, pct_positions, 0.95, 504)
    return_value['CVaR_95_2y'] = conditional_value_at_risk(returns, pct_positions, 0.95, 504)
    return_value['HVaR_99_2y'] = hist_value_at_risk(returns, pct_positions, 0.99, 504)
    return_value['CVaR_99_2y'] = conditional_value_at_risk(returns, pct_positions, 0.99, 504)
    return_value['portfolio_notional'] = np.sum(np.absolute(pct_positions))
    return_value['Brexit_Stress'] = brexit_stress(returns, pct_positions)
    return return_value


#%% use tabulate to create portfolio stats table
def print_risk_results(portfolio):
    from tabulate import tabulate
    headers = ["Metric", "Risk % Portfolio"]
    print(tabulate([(keys, values) for keys, values in portfolio.items()], headers = headers))