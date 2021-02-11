#%% import the libraries
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy import stats

#%% import the data
instrument = "hourly_fx_close"
startdate = '2010-01-01'
enddate = '2018-12-31'
resample_period = "1h"

def read_pickle(instrument, startdate, enddate, resample_period):
    path_ = ("/home/jake/Code/kaiseki/local_data/" + instrument + "_pickle")
    df = pd.read_pickle(path_).loc[startdate:enddate].resample(resample_period).last()
    return df

fx_c = read_pickle(instrument, startdate, enddate, resample_period)


fx_c_rets = fx_c.pct_change(1)

#%% run mean reversion model

fx_c_ranks = fx_c_rets.rank(axis=1, ascending=False)
df_long_signal = np.sign(fx_c_ranks[fx_c_ranks<=1])
df_short_signal = np.sign(fx_c_ranks[fx_c_ranks>=9])*(-1)
long_rets = df_long_signal.shift(1) * fx_c_rets*(-1)
short_rets = df_short_signal.shift(1) * fx_c_rets*(-1)
long_rets_sum = long_rets.mean(axis=1)
short_rets_sum = short_rets.mean(axis=1)
mr_pnl = (long_rets_sum + short_rets_sum) / 2
mr_pnl_daily = mr_pnl.resample("1b").sum()


mr_cum_pnl = mr_pnl.cumsum()
mr_cum_pnl.plot()
plt.show()







#%% create csv files (need to delete old ones)
fx_c_ranks.to_csv('temp_mr_ranks.csv')
fx_c_rets.to_csv('temp_mr_rets.csv')
long_rets_sum.to_csv('temp_long_rets.csv')
short_rets_sum.to_csv('temp_short_rets.csv')
mr_pnl.to_csv('temp_mr_pnl.csv')
mr_pnl_daily.to_csv('temp_mr_pnl_daily.csv')
#%% plot

#%%
market_vol = mr_pnl.rolling(window=50, center=False).std()

#%%
plt.scatter(market_vol.iloc[-2000:],mr_pnl.iloc[-2000:])
plt.show()


#%%
df_plots = pd.concat([mr_cum_pnl, market_vol], axis=1)
df_plots.plot(subplots=True)
plt.show()
