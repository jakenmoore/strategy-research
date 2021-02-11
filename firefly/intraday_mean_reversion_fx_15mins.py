#%% import the libraries
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy import stats

#%%
def read_pickle(instrument, startdate, enddate, resample_period):
    path_ = ('/home/jake/Code/kaiseki/local_data/' + instrument + "_pickle")
    df = pd.read_pickle(path_).loc[startdate:enddate].resample(resample_period).last()
    return df

#%% load some data
instrument = ""
startdate = '2015-01-01'
enddate = '2018-12-31'
resample_period = "15t"


usdeur = read_pickle("EURUSD", startdate, enddate, resample_period)
usdeur= usdeur[['open','high','low','close']]
usdeur= 1/usdeur
usdeur.columns = [['usdeur_open','usdeur_high','usdeur_low','usdeur_close']]

usdaud = read_pickle("AUDUSD", startdate, enddate, resample_period)
usdaud= usdaud[['open','high','low','close']]
usdaud= 1/usdaud
usdaud.columns = [['usdaud_open','usdaud_high','usdaud_low','usdaud_close']]

usdcad = read_pickle("USDCAD", startdate, enddate, resample_period)
usdcad= usdcad[['open','high','low','close']]
usdcad.columns = [['usdcad_open','usdcad_high','usdcad_low','usdcad_close']]

usdchf = read_pickle("USDCHF", startdate, enddate, resample_period)
usdchf= usdchf[['open','high','low','close']]
usdchf.columns = [['usdchf_open','usdchf_high','usdchf_low','usdchf_close']]

usdjpy = read_pickle("USDJPY", startdate, enddate, resample_period)
usdjpy= usdjpy[['open','high','low','close']]
usdjpy.columns = [['usdjpy_open','usdjpy_high','usdjpy_low','usdjpy_close']]

usdgbp = read_pickle("GBPUSD", startdate, enddate, resample_period)
usdgbp= usdgbp[['open','high','low','close']]
usdgbp= 1/usdgbp
usdgbp.columns = [['usdgbp_open','usdgbp_high','usdgbp_low','usdgbp_close']]

usdnok = read_pickle("USDNOK", startdate, enddate, resample_period)
usdnok= usdnok[['open','high','low','close']]
usdnok.columns = [['usdnok_open','usdnok_high','usdnok_low','usdnok_close']]

usdsek = read_pickle("USDSEK", startdate, enddate, resample_period)
usdsek= usdsek[['open','high','low','close']]
usdsek.columns = [['usdsek_open','usdsek_high','usdsek_low','usdsek_close']]

usdnzd = read_pickle("NZDUSD", startdate, enddate, resample_period)
usdnzd= usdnzd[['open','high','low','close']]
usdnzd= 1/usdnzd
usdnzd.columns = [['usdnzd_open','usdnzd_high','usdnzd_low','usdnzd_close']]




#%%
 # put all the dataframes together

df = pd.concat([usdeur,usdaud,usdcad,usdchf,usdjpy,usdgbp,usdnok,usdsek],axis=1)

df = pd.concat([usdeur['usdeur_close'],usdaud['usdaud_close'],usdcad['usdcad_close'],usdchf['usdchf_close'],usdjpy['usdjpy_close'],usdgbp['usdgbp_close'],usdnok['usdnok_close'],usdsek['usdsek_close'], usdnzd['usdnzd_close']],axis=1)
df.index = df.index.to_pydatetime()
df.sort_index(inplace=True)
df = df[df.index.dayofweek < 5]

fx_c = df.dropna()

#
fx_c_rets = fx_c.pct_change()

 # run mean reversion model

fx_c_ranks = fx_c_rets.rank(axis=1, ascending=False)
df_long_signal = np.sign(fx_c_ranks[fx_c_ranks<=1])
df_short_signal = np.sign(fx_c_ranks[fx_c_ranks>=9])*(-1)
long_rets = df_long_signal.shift(1) * fx_c_rets*(-1)
short_rets = df_short_signal.shift(1) * fx_c_rets*(-1)
long_rets_sum = long_rets.mean(axis=1)
short_rets_sum = short_rets.mean(axis=1)
mr_pnl = (long_rets_sum + short_rets_sum) / 2



print(np.mean(mr_pnl)*100)

mr_cum_pnl = mr_pnl.cumsum()
mr_cum_pnl.plot()
plt.show()
















#%% create csv files (need to delete old ones)

mr_pnl_daily = mr_pnl.resample("1b").sum()

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
