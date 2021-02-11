
#%% import the libraries
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy import stats

#%% import the data
fx_data = pd.read_csv('C:\\Users\\JakeMoore\\Documents\\kaiseki\\Kaiseki_models\\intradaydata.csv', parse_dates=['Date'], index_col='Date')

#%% create new dataframe of fx returns
fx_c = pd.concat([fx_data['USDJPY_C'], fx_data['USDGBP_C'], fx_data['USDEUR_C'], fx_data['USDNOK_C'],
               fx_data['USDSEK_C'], fx_data['USDCHF_C'], fx_data['USDAUD_C'], fx_data['USDCAD_C']], axis=1)
fx_c_rets = fx_c.pct_change(1)

fx_c_rets_norm = stats.zscore(fx_c_rets, axis=1, ddof=1)
fx_c_rets_norm = pd.DataFrame(data=fx_c_rets_norm, index=fx_c.index)

# So far not using Z scores / Normalized as columns are wierd

fx_c_ranks = fx_c_rets.rank(axis=1, ascending=False)
df_long_signal = fx_c_ranks[fx_c_ranks==1]
df_short_signal = np.sign(fx_c_ranks[fx_c_ranks==8])*(-1)
long_rets = df_long_signal.shift(1) * fx_c_rets*(-1)
short_rets = df_short_signal.shift(1) * fx_c_rets*(-1)
long_rets_sum = long_rets.sum(axis=1)
short_rets_sum = short_rets.sum(axis=1)
mr_pnl = long_rets_sum + short_rets_sum
mr_cum_pnl = (1 + mr_pnl).cumprod()

#%% create csv files (need to delete old ones)
fx_c_ranks.to_csv('temp_mr_ranks.csv')
fx_c_rets.to_csv('temp_mr_rets.csv')
long_rets.to_csv('temp_long_rets.csv')
short_rets.to_csv('temp_short_rets.csv')
mr_pnl.to_csv('temp_mr_pnl.csv')
#%% plot
mr_cum_pnl.plot()
plt.show()

#%% import SMA2 2 returns
sma2rets = pd.read_csv('C:\\Users\\JakeMoore\\Documents\\kaiseki\\temp_tf_SMA2.csv', parse_dates=['Date'], index_col='Date')
mr_rets_excel = pd.read_csv('C:\\Users\\JakeMoore\\Documents\\kaiseki\\temp_mr_excel_pnl.csv', parse_dates=['Date'], index_col='Date')

two_model_returns = pd.concat([sma2rets, mr_rets_excel], join='inner', axis=1)
two_model_returns.to_csv('temp_two_models2.csv')