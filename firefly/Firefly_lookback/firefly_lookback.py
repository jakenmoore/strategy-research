
import pandas as pd
import matplotlib.pyplot as plt
from Kaiseki_Utilities.read_data import read_pickle
import numpy as np
#
# #%% load some data
# instrument = "EURUSD"
# startdate = '2000-01-01'
# enddate = '2018-12-31'
# resample_period = "1t"
#
# usdeur = read_pickle("EURUSD", startdate, enddate, resample_period)
# usdeur= usdeur[['open','close']]
# usdeur= 1/usdeur
# usdeur.columns = [['usdeur_open','usdeur_close']]
#
# usdaud = read_pickle("AUDUSD", startdate, enddate, resample_period)
# usdaud= usdaud[['open','close']]
# usdaud= 1/usdaud
# usdaud.columns = [['usdaud_open','usdaud_close']]
#
# #usdnzd = read_pickle("NZDUSD", startdate, enddate, resample_period)
# #usdnzd= usdnzd[['close']]
# #usdnzd= 1/usdnzd
# #usdnzd.columns = [['usdnzd']]
#
# usdcad = read_pickle("USDCAD", startdate, enddate, resample_period)
# usdcad= usdcad[['open','close']]
# usdcad.columns = [['usdcad_open','usdcad_close']]
#
# usdchf = read_pickle("USDCHF", startdate, enddate, resample_period)
# usdchf= usdchf[['open','close']]
# usdchf.columns = [['usdchf_open','usdchf_close']]
#
# usdjpy = read_pickle("USDJPY", startdate, enddate, resample_period)
# usdjpy= usdjpy[['open','close']]
# usdjpy.columns = [['usdjpy_open','usdjpy_close']]
#
# usdgbp = read_pickle("GBPUSD", startdate, enddate, resample_period)
# usdgbp= usdgbp[['open','close']]
# usdgbp= 1/usdgbp
# usdgbp.columns = [['usdgbp_open','usdgbp_close']]
#
# usdnok = read_pickle("USDNOK", startdate, enddate, resample_period)
# usdnok= usdnok[['open','close']]
# usdnok.columns = [['usdnok_open','usdnok_close']]
#
# usdsek = read_pickle("USDSEK", startdate, enddate, resample_period)
# usdsek= usdsek[['open','close']]
# usdsek.columns = [['usdsek_open','usdsek_close']]
#
# #put all the dataframes together
# df = pd.concat([usdeur,usdaud,usdcad,usdchf,usdjpy,usdgbp,usdnok,usdsek],axis=1)
# df.index = df.index.to_pydatetime()
# df  = df.dropna()
#
# #open
# df_open = df.filter(regex='_open', axis=1).rename(columns = lambda x : str(x)[:-5])
# df_open =  df_open.resample('1H').first()
#
# #close
# df_close = df.filter(regex='_close', axis=1).rename(columns = lambda x : str(x)[:-6])
# df_close =  df_close.resample('1H').last()
#
# df_open = df_close.shift(1)

df = pd.read_csv('C:\\Repos\\Kaiseki\\intradaydata\\intradaydata.csv', index_col='Date', parse_dates=True)
df = df.replace(to_replace=-1, method='ffill')
df.index = df.index.to_pydatetime()

df_close = df.filter(regex='_C', axis=1).rename(columns = lambda x : str(x)[:-2])
df_open = df.filter(regex='_O', axis=1).rename(columns = lambda x : str(x)[:-2])
df_high = df.filter(regex='_H', axis=1).rename(columns = lambda x : str(x)[:-2])
df_low = df.filter(regex='_L', axis=1).rename(columns = lambda x : str(x)[:-2])

#calculate the % returns
df_ret = df_close / df_open -1

#calculate the row mean
df_ret_mean = df_ret.rolling(20).mean()

#calculate the row stdev
df_ret_std = df_ret.rolling(24).std()

#calculate teh x scores
df_ret_z = (df_ret  - df_ret_mean ) / df_ret_std


#calculate the ranks
df_rank =  df_ret_z.rank(axis=1)

#calculate signal
df_signal =np.where(df_rank == 8,-1,0)
df_signal =np.where(df_rank == 1,1,df_signal)
df_signal = pd.DataFrame(data=df_signal, index=df_rank.index,columns   =df_rank.columns )

#calculate the return
df_strat_ret = df_ret.shift(-1) * df_signal
df_strat_ret['TOTAL']  = df_strat_ret.sum(axis=1) #- 0.00008
df_strat_ret['Cum_TOTAL'] = df_strat_ret['TOTAL'] .cumsum()
df_strat_ret['DollarValue'] = df_strat_ret['TOTAL']*100000 -8  #8 is the cost ie 4 * max(2bps or 2$)
df_strat_ret['DollarValue']=df_strat_ret['DollarValue'].cumsum()

#First plot - without stops

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Cumulative Return', color=color)
ax1.plot(df_strat_ret[['Cum_TOTAL']], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Dollar Value', color=color)  # we already handled the x-label with ax1
ax2.plot(df_strat_ret[['DollarValue']], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# rotate and align the tick labels so they look better
fig.autofmt_xdate()
plt.title('Without Stops')
plt.show()




