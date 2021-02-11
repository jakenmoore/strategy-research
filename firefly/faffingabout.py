import pandas as pd
import matplotlib.pyplot as plt
from Kaiseki_Utilities.read_data import read_pickle
import numpy as np
import importlib

#for making changes to the module
import Kaiseki_Utilities.Database_Utils
importlib.reload(Kaiseki_Utilities.Database_Utils)
from Kaiseki_Utilities.Database_Utils import  getHourlyBars, prepareDB,getHourlyBarsPivoted

#read the trades file

prepareDB()
#df = getHourlyBars('EURUSD','2019-04-01 00:00','2019-04-06 00:00')
tickers = ['EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','USDJPY','USDNOK','NZDUSD','USDSEK']
df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,'2018-01-01 00:00','2019-04-11 00:00',dollarise=True)


corr =  df_close.corr(method='pearson')

#df_close[['USDGBP']].plot()
#plt.show()

#set the default volatility values
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
df_ret = (df_close / df_open)-1

#calculate the row mean
df_ret_mean = df_ret.mean(axis=1)
df_ret_mean_rolling = df_ret.rolling(25).mean()

#calculate the row stdev
df_ret_std = df_ret.std(axis=1)

#calculate teh x scores
df_ret_z = df_ret.sub(df_ret_mean, axis='rows')
df_ret_z = df_ret_z.div(df_ret_std, axis='rows')

#calculate the ranks
df_rank =  df_ret_z.rank(axis=1)

#calculate signal
df_signal =np.where(df_rank == 8,-1,0)
df_signal =np.where(df_rank == 1,1,df_signal)
df_signal = pd.DataFrame(data=df_signal, index=df_rank.index,columns   =df_rank.columns )

#separate dataframe for longs and sorts
df_signal_s = pd.DataFrame(data=np.where((df_rank == 8) & (df_ret_z >=1) ,-1,0), index=df_rank.index,columns   =df_rank.columns )
df_signal_l = pd.DataFrame(data=np.where((df_rank == 1) & (df_ret_z <=-1),1,0), index=df_rank.index,columns   =df_rank.columns )

# Zero out the singal at 10PM
df_signal_s.loc[df_signal_s.at_time('22:00').index] = 0
df_signal_l.loc[df_signal_s.at_time('22:00').index] = 0

# Calculate the stop levels
df_vol = df_open.pct_change().rolling(24).std().shift(1) #add shif to avoid look ahead bias


for col in df_vol.columns:
    df_vol[[col]] = df_vol[[col]].fillna(value=_Vol[col])

# Constant volatility model
#df_vol = df_open.pct_change()[1:100].std() #add shif to avoid look ahead bias


factorStop = 1.5
factorTP = 1

## Stops
df_Long_stop_level =df_open*(1-df_vol*factorStop)
df_short_stop_level =df_open*(1+df_vol*factorStop)

# Take profits
# Todo - this is a problem when vol is 0
df_Long_tp_level =(df_open*(1+df_vol*factorTP))
df_short_tp_level =(df_open*(1-df_vol*factorTP))

df_long_ret = np.where(df_low < df_Long_stop_level,
                                (df_Long_stop_level/df_open) - 1,
                                (df_close/df_open)-1)

#df_long_ret = np.where(df_high > df_Long_tp_level,
#                      (df_Long_tp_level/df_open) -1,
#                      df_long_ret)

df_short_ret = np.where(df_high > df_short_stop_level,
                                (df_short_stop_level/df_open)-1,
                                (df_close/df_open)-1)

#df_short_ret = np.where(df_low < df_short_tp_level,
#                                ((df_short_tp_level/df_open) - 1),
#                       df_short_ret)

df_long_ret = pd.DataFrame(data=df_long_ret, index=df_low.index,columns   =df_low.columns )
df_short_ret = pd.DataFrame(data=df_short_ret, index=df_low.index,columns   =df_low.columns )


df_strat_ret_stops = + df_long_ret.shift(-1) * df_signal_l + df_short_ret.shift(-1) * df_signal_s
df_strat_ret_stops['TOTAL']  = (df_strat_ret_stops.sum(axis=1) - df_signal_l.astype(bool).sum(axis=1) * 0.00004 - df_signal_s.astype(bool).sum(axis=1) * 0.00004) \
                               / (df_signal_l.astype(bool).sum(axis=1) + df_signal_s.astype(bool).sum(axis=1))

df_strat_ret_stops[['TOTAL']] = df_strat_ret_stops[['TOTAL']].fillna(0)

df_strat_ret_stops['TOTAL']  = df_strat_ret_stops['TOTAL'] *100
df_strat_ret_stops['Cum_TOTAL'] = df_strat_ret_stops['TOTAL'] .cumsum()
df_strat_ret_stops['High_water_mark'] = df_strat_ret_stops['Cum_TOTAL'].cummax()
df_strat_ret_stops['Draw_Down'] = df_strat_ret_stops['High_water_mark'] - df_strat_ret_stops['Cum_TOTAL']

df_strat_ret_stops['DollarValue'] = df_strat_ret_stops['TOTAL']*100000
df_strat_ret_stops['DollarValue']=df_strat_ret_stops['DollarValue'].cumsum()


#df_strat_ret_stops = RunStrategy(df_close,df_open)

df_strat_ret_stops.to_csv('returns.csv')

# Max Draw Down
print(np.max(df_strat_ret_stops['Draw_Down']))

#last 12 months annualize return
np.mean(df_strat_ret_stops['TOTAL']) * 250 * 22

#
np.std(df_strat_ret_stops['TOTAL']) * np.sqrt(250 * 22)


### Second plot  - with stops
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Cumulative Return %', color=color)
ax1.plot(df_strat_ret_stops[['Cum_TOTAL']], color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:blue'
#ax2.set_ylabel('Dollar Value', color=color)  # we already handled the x-label with ax1
#ax2.plot(df_strat_ret_stops[['DollarValue']], color=color)
#ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# rotate and align the tick labels so they look better
fig.autofmt_xdate()
plt.title('G10 FX relative value MR')
plt.show()

# # scatterplot of signal strength vs return
#
# df_ret_renamed  =df_strat_ret_stops.shift(1).rename(columns = lambda x : "ret_" + str(x))
# df_signal_str  =  df_ret_z
# df_draw = pd.concat([df_ret_renamed,df_signal_str],axis=1)
# df_draw = df_draw.dropna()#
#
# df_draw = df_draw.loc[(df_draw['USDEUR'] != 0).index,['USDEUR','ret_USDEUR']]
# ax1 = df_draw.plot(kind='scatter', x='USDEUR', y='ret_USDEUR', color='r')
# plt.show()

# ##model to regress signal strength with teh expected return
#
# import statsmodels.api as sm#
# X = df_draw["USDEUR"]
# y = df_draw["ret_USDEUR"]#
#
# # Note the difference in argument order
# model = sm.OLS(y, X).fit()
# #predictions = model.predict(X) # make the predictions by the model#
#
# # Print out the statistics
# model.summary()






