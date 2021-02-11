import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib

#for making changes to the module
import utilities.Database_Utils
importlib.reload(utilities.Database_Utils)
from utilities.Database_Utils import  getTrades, prepareDB,getHourlyBarsPivoted, getPositions,Dollarise, getOrders
from utilities.PnL_Utils import  MarkToMarket

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

prepareDB()

tickers = ['EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','USDJPY','USDNOK','NZDUSD','USDSEK']
df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,'2018-01-01 00:00','2019-06-01 00:00',dollarise=True)

# Set the default volatility values
_Vol = {}
_Vol["USDJPY"] = 0.000835;
_Vol["USDGBP"] = 0.000729;
_Vol["USDEUR"] = 0.000666;
_Vol["USDNOK"] = 0.000986;
_Vol["USDSEK"] = 0.000967;
_Vol["USDCHF"] = 0.000749;
_Vol["USDAUD"] = 0.000947;
_Vol["USDNZD"] = 0.000947;
_Vol["USDCAD"] = 0.000949;

#calculate the % returns
df_ret = (df_close / df_close)-1

#calculate the row mean
df_ret_mean = df_ret.mean(axis=1)
df_ret_mean_rolling = df_ret.rolling(25).mean()

#calculate the row stdev
df_ret_std = df_ret.std(axis=1)

#calculate the z scores
df_ret_z = df_ret.sub(df_ret_mean, axis='rows')
df_ret_z = df_ret_z.div(df_ret_std, axis='rows')

#calculate the ranks
df_rank =  df_ret_z.rank(axis=1)

#calculate signal
df_signal =np.where(df_rank == 9,-1,0)
df_signal =np.where(df_rank ==1 ,1,df_signal)

df_signal = pd.DataFrame(data=df_signal, index=df_rank.index,columns=df_rank.columns)

#separate dataframe for longs and sorts
df_signal_s = pd.DataFrame(data=np.where((df_rank == 9) & (df_ret_z >=1) ,-1,0), index=df_rank.index,columns = df_rank.columns )
df_signal_l = pd.DataFrame(data=np.where((df_rank ==1) & (df_ret_z <=-1),1,0), index=df_rank.index,columns = df_rank.columns )

# Zero out the singal at 10PM
df_signal_s.loc[df_signal_s.at_time('22:00').index] = 0
df_signal_l.loc[df_signal_s.at_time('22:00').index] = 0

# Calculate the stop levels
df_vol = df_open.pct_change().rolling(24).std().shift(1) #add shif to avoid look ahead bias

for col in df_vol.columns:
    df_vol[[col]] = df_vol[[col]].fillna(value=_Vol[col])

factorStop = 1.5

## Stops
df_Long_stop_level =df_open*(1-df_vol*factorStop)
df_short_stop_level =df_open*(1+df_vol*factorStop)

# Take profits

#df_long_ret = np.where((df_low < df_Long_stop_level) & (df_signal_l.shift(1) == 1),
#                                (df_Long_stop_level/df_open) - 1,
#                                (df_close/df_open)-1)
#& (df_signal_s == -1)
#df_short_ret = np.where((df_high > df_short_stop_level) & (df_signal_s.shift(1) == -1),
#                                (df_short_stop_level/df_open)-1,
#                                (df_close/df_open)-1)

df_long_ret = np.where((df_low < df_Long_stop_level) & (df_signal_l.shift(1) == 1),
                                (df_close/df_open) - 1,
                                (df_close/df_open)-1)
#& (df_signal_s == -1)
df_short_ret = np.where((df_high > df_short_stop_level) & (df_signal_s.shift(1) == -1),
                                (df_close/df_open)-1,
                                (df_close/df_open)-1)



df_long_ret = pd.DataFrame(data=df_long_ret, index=df_open.index,columns   =df_open.columns )
df_short_ret = pd.DataFrame(data=df_short_ret, index=df_open.index,columns   =df_open.columns )

#the actual prices at which we close the trades
df_effective_close_short = np.where((df_high > df_short_stop_level) & (df_signal_s.shift(1) == -1),df_short_stop_level,df_close)
df_effective_close_long = np.where((df_low < df_Long_stop_level) & (df_signal_l.shift(1) == 1),df_Long_stop_level,0)

df_effective_close = df_effective_close_short + df_effective_close_long
df_effective_close = pd.DataFrame(data=df_effective_close, index=df_low.index, columns=df_low.columns)

#df_effective_close = df_close
#df_effective_close.loc[df_high > df_short_stop_level] = df_short_stop_level
#df_effective_close.loc[df_low > df_Long_stop_level] = df_Long_stop_level

df_strat_ret_stops = + df_long_ret.shift(-1) * df_signal_l + df_short_ret.shift(-1) * df_signal_s
df_signal = df_signal_l + df_signal_s
df_strat_ret_stops['TOTAL']  = (df_strat_ret_stops.sum(axis=1) - df_signal_l.astype(bool).sum(axis=1) * 0.00004 - df_signal_s.astype(bool).sum(axis=1) * 0.00004) \
                               / (df_signal_l.astype(bool).sum(axis=1) + df_signal_s.astype(bool).sum(axis=1))

df_strat_ret_stops['TOTAL']  = df_strat_ret_stops['TOTAL'] *100
df_strat_ret_stops['Cum_TOTAL'] = df_strat_ret_stops['TOTAL'] .cumsum()
df_strat_ret_stops['High_water_mark'] = df_strat_ret_stops['Cum_TOTAL'].cummax()
df_strat_ret_stops['Draw_Down'] = df_strat_ret_stops['High_water_mark'] - df_strat_ret_stops['Cum_TOTAL']

df_strat_ret_stops['DollarValue'] = df_strat_ret_stops['TOTAL']*100000
df_strat_ret_stops['DollarValue']=df_strat_ret_stops['DollarValue'].cumsum()


# Strategy statistics
print("Max DrawDown " + str(np.max(df_strat_ret_stops['Draw_Down'])) + "%")
print("Annualised return " + str(np.mean(df_strat_ret_stops['TOTAL']) * 250 * 22) + "%")
print("Annualised volatility " + str(np.std(df_strat_ret_stops['TOTAL']) * np.sqrt(250 * 22)) + "%")

### Plot  - with stops
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Cumulative Return %', color=color)
ax1.plot(df_strat_ret_stops[['Cum_TOTAL']], color=color)
ax1.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# rotate and align the tick labels so they look better
fig.autofmt_xdate()
plt.title('G10 FX relative value MR')
plt.show()
