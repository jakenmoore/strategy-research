import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


#load the data

df_open=0
df_close=0
df_high =0
df_low = 0

# DATASET 1
# df = pd.read_csv('C:\\Repos\\Kaiseki\\intradaydata\\intradaydata.csv', index_col='Date', parse_dates=True)

# # #DATASET 2
# df = pd.read_csv('C:\\Users\\Tommy\\Kaiseki\\Jake Moore - Systems\\data\\intraday\\intraday_data_merged_08.02.2019.csv', index_col='Date', parse_dates=True,dayfirst=True)
# df = df[df.columns.drop(list(df.filter(regex='USDNZD')))]
# #
# # #Data preprocessing for dataset 1 & 2
# df = df.replace(to_replace=-1, method='ffill')
# df.index = df.index.to_pydatetime()
# #
# df_close = df.filter(regex='_C', axis=1).rename(columns = lambda x : str(x)[:-2])
# df_open = df.filter(regex='_O', axis=1).rename(columns = lambda x : str(x)[:-2])
# df_high = df.filter(regex='_H', axis=1).rename(columns = lambda x : str(x)[:-2])
# df_low = df.filter(regex='_L', axis=1).rename(columns = lambda x : str(x)[:-2])
#
# ##END OF Data preprocessing for dataset 1 & 2


#DATASET 3
import utilities.Database_Utils
from utilities.Database_Utils import  getTrades, prepareDB,getHourlyBarsPivoted, getPositions,Dollarise, getOrders
from utilities.PnL_Utils import  MarkToMarket
#
#prepareDB()

#tickers = ['AUDUSD','USDCAD','USDCHF','EURUSD','GBPUSD','USDJPY','USDNOK','USDSEK']
#df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,'2018-04-05 00:00','2019-02-08 00:00',dollarise=True)

#ticker  = 'USDSEK'
#df_close[ticker + '2'] = df_close2[ticker]
#df_close[[ticker ,ticker + '2']].plot()
#plt.show()
# END OF DATASET 3


#DATASET 4
from utilities.read_data import read_pickle
instrument = "EURUSD"
startdate = '2000-01-01'
enddate = '2018-12-31'
resample_period = "1t"

usdeur = read_pickle("EURUSD", startdate, enddate, resample_period)
usdeur= usdeur[['open','high','low','close']]
usdeur= 1/usdeur
usdeur.columns = [['usdeur_open','usdeur_high','usdeur_low','usdeur_close']]

usdaud = read_pickle("AUDUSD", startdate, enddate, resample_period)
usdaud= usdaud[['open','high','low','close']]
usdaud= 1/usdaud
usdaud.columns = [['usdaud_open','usdaud_high','usdaud_low','usdaud_close']]

#usdnzd = read_pickle("NZDUSD", startdate, enddate, resample_period)
#usdnzd= usdnzd[['close']]
#usdnzd= 1/usdnzd
#usdnzd.columns = [['usdnzd']]

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

#put all the dataframes together
df = pd.concat([usdeur,usdaud,usdcad,usdchf,usdjpy,usdgbp,usdnok,usdsek],axis=1)
df.index = df.index.to_pydatetime()
df  = df.dropna()
df.columns = df.columns.get_level_values(0)

#open
df_open = df.filter(regex='_open', axis=1).rename(columns = lambda x : str(x)[:-5])
df_open =  df_open.resample('1H').first()

df_close =  df_open.shift(-1)

df_high = df.filter(regex='_high', axis=1).rename(columns = lambda x : str(x)[:-5])
df_high =  df_high.resample('1H').max()

df_low = df.filter(regex='_low', axis=1).rename(columns = lambda x : str(x)[:-4])
df_low =  df_low.resample('1H').min()
# END DATASET 4

_Vol = {}
_Vol["usdjpy"] = 0.000835;
_Vol["usdgbp"] = 0.000729;
_Vol["usdeur"] = 0.000666;
_Vol["usdnok"] = 0.000986;
_Vol["usdsek"] = 0.000967;
_Vol["usdchf"] = 0.000749;
_Vol["usdaud"] = 0.000947;
_Vol["usdnzd"] = 0.000947;
_Vol["usdcad"] = 0.000949;

#calculate the % returns
df_ret = df_close / df_open - 1

#calculate the row mean
df_ret_mean = df_ret.mean(axis=1)

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
df_signal_s = pd.DataFrame(data=np.where((df_rank == 8) ,-1,0), index=df_rank.index,columns   =df_rank.columns )
df_signal_l = pd.DataFrame(data=np.where((df_rank == 1) ,1,0), index=df_rank.index,columns   =df_rank.columns )


#calculate the stop levels
df_vol = df_close.pct_change().rolling(24).std().shift(1)
for col in df_vol.columns:
    df_vol[[col]] = df_vol[[col]].fillna(value=_Vol[col])
    df_vol[[col]] = df_vol[[col]].replace(0,value=_Vol[col])

factorStop = 1

## Stops
df_Long_stop_level =(df_open*(1-df_vol*factorStop))
df_short_stop_level =(df_open*(1+df_vol*factorStop))

df_long_ret = np.where((df_low <= df_Long_stop_level) & (df_signal_l.shift(1) == 1),
                       df_Long_stop_level/df_open -1,
                                0)

df_long_ret = np.where((df_low > df_Long_stop_level) & (df_signal_l.shift(1) == 1),
                       df_close/df_open -1,
                                df_long_ret)


df_short_ret = np.where((df_high >= df_short_stop_level) & (df_signal_s.shift(1) == -1),
                                 df_short_stop_level/df_open - 1,
                                0)

df_short_ret = np.where((df_high < df_short_stop_level) & (df_signal_s.shift(1) == -1),
                                 df_close/df_open - 1,
                                df_short_ret)

df_long_ret = pd.DataFrame(data=df_long_ret, index=df_low.index,columns   =df_low.columns )
df_short_ret = pd.DataFrame(data=df_short_ret, index=df_low.index,columns   =df_low.columns )

#calculate the return
df_strat_ret_stops = df_long_ret * df_signal_l.shift(1) + df_short_ret * df_signal_s.shift(1)
df_strat_ret_stops['TOTAL']  = df_strat_ret_stops.sum(axis=1)
df_strat_ret_stops['Count'] = df_signal_l.astype(bool).sum(axis=1) + df_signal_s.astype(bool).sum(axis=1)
df_strat_ret_stops['Count'] = df_strat_ret_stops['Count'].replace(0,1)
df_strat_ret_stops['Cost'] = df_strat_ret_stops['Count'] * 0.00002
df_strat_ret_stops['TOTAL_NET'] = df_strat_ret_stops['TOTAL']
df_strat_ret_stops['TOTAL_NET'] = df_strat_ret_stops['TOTAL_NET'] - df_strat_ret_stops['Cost']
df_strat_ret_stops['TOTAL_NET'] = df_strat_ret_stops['TOTAL_NET'] / df_strat_ret_stops['Count']
df_strat_ret_stops['Cum_TOTAL'] = df_strat_ret_stops['TOTAL_NET'] .cumsum()


df_strat_ret_stops=df_strat_ret_stops.fillna(0)
sharpe = np.mean(df_strat_ret_stops['TOTAL_NET']) * 24 * 250 / (np.std(df_strat_ret_stops['TOTAL_NET']) * np.sqrt(250*24))

### Second plot  - with stops
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
print(sharpe)
