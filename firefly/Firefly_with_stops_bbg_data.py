
import pandas as pd
import matplotlib.pyplot as plt
#from utilities.read_data import read_pickle
import numpy as np


# #%% load some data
# instrument = "EURUSD"
# startdate = '2000-01-01'
# enddate = '2018-12-31'
# resample_period = "1t"
#
# usdeur = read_pickle("EURUSD", startdate, enddate, resample_period)
# usdeur= usdeur[['open','high','low','close']]
# usdeur= 1/usdeur
# usdeur.columns = [['usdeur_open','usdeur_high','usdeur_low','usdeur_close']]
#
# usdaud = read_pickle("AUDUSD", startdate, enddate, resample_period)
# usdaud= usdaud[['open','high','low','close']]
# usdaud= 1/usdaud
# usdaud.columns = [['usdaud_open','usdaud_high','usdaud_low','usdaud_close']]
#
# #usdnzd = read_pickle("NZDUSD", startdate, enddate, resample_period)
# #usdnzd= usdnzd[['close']]
# #usdnzd= 1/usdnzd
# #usdnzd.columns = [['usdnzd']]
#
# usdcad = read_pickle("USDCAD", startdate, enddate, resample_period)
# usdcad= usdcad[['open','high','low','close']]
# usdcad.columns = [['usdcad_open','usdcad_high','usdcad_low','usdcad_close']]
#
# usdchf = read_pickle("USDCHF", startdate, enddate, resample_period)
# usdchf= usdchf[['open','high','low','close']]
# usdchf.columns = [['usdchf_open','usdchf_high','usdchf_low','usdchf_close']]
#
# usdjpy = read_pickle("USDJPY", startdate, enddate, resample_period)
# usdjpy= usdjpy[['open','high','low','close']]
# usdjpy.columns = [['usdjpy_open','usdjpy_high','usdjpy_low','usdjpy_close']]
#
# usdgbp = read_pickle("GBPUSD", startdate, enddate, resample_period)
# usdgbp= usdgbp[['open','high','low','close']]
# usdgbp= 1/usdgbp
# usdgbp.columns = [['usdgbp_open','usdgbp_high','usdgbp_low','usdgbp_close']]
#
# usdnok = read_pickle("USDNOK", startdate, enddate, resample_period)
# usdnok= usdnok[['open','high','low','close']]
# usdnok.columns = [['usdnok_open','usdnok_high','usdnok_low','usdnok_close']]
#
# usdsek = read_pickle("USDSEK", startdate, enddate, resample_period)
# usdsek= usdsek[['open','high','low','close']]
# usdsek.columns = [['usdsek_open','usdsek_high','usdsek_low','usdsek_close']]
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
# df_close2 =  df_open.shift(-1)
#
# #df_close = df.filter(regex='_close', axis=1).rename(columns = lambda x : str(x)[:-6])
# #df_close =  df_close.resample('1H').last()
#
#
# df_high = df.filter(regex='_high', axis=1).rename(columns = lambda x : str(x)[:-5])
# df_high =  df_high.resample('1H').max()
#
# df_low = df.filter(regex='_low', axis=1).rename(columns = lambda x : str(x)[:-4])
# df_low =  df_low.resample('1H').min()

#df = pd.read_csv('C:\\Repos\\Kaiseki\\intradaydata\\intradaydata.csv', index_col='Date', parse_dates=True)
df = pd.read_csv('C:\\Users\\Tommy\\Kaiseki\\Jake Moore - Systems\\data\\intraday\\intraday_data_merged_08.02.2019.csv', index_col='Date', parse_dates=True,dayfirst=True)
df = df[df.columns.drop(list(df.filter(regex='USDNZD')))]
df = df.replace(to_replace=-1, method='ffill')
df.index = df.index.to_pydatetime()

#dont trade in after 10PM
df = df.between_time('00:00', '22:00')
df_close = df.filter(regex='_C', axis=1).rename(columns = lambda x : str(x)[:-2])
df_open = df.filter(regex='_O', axis=1).rename(columns = lambda x : str(x)[:-2])
df_high = df.filter(regex='_H', axis=1).rename(columns = lambda x : str(x)[:-2])
df_low = df.filter(regex='_L', axis=1).rename(columns = lambda x : str(x)[:-2])



#tickers = ['EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','USDJPY','USDNOK','USDSEK']
#df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,'2018-01-01 00:00','2019-06-01 00:00',dollarise=True)



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

# Constant volatility model
#df_vol = df_open.pct_change()[1:100].std() #add shif to avoid look ahead bias

factorStop = .1
factorTP = 1

## Stops
df_Long_stop_level =df_open*(1-df_vol*factorStop)
df_short_stop_level =df_open*(1+df_vol*factorStop)

# Take profits
# Todo - this is a problem when vol is 0
df_Long_tp_level =(df_open*(1+df_vol*factorTP))
df_short_tp_level =(df_open*(1-df_vol*factorTP))

df_long_ret = np.where((df_low < df_Long_stop_level) & (df_signal_l.shift(1) == 1),
                                (df_close/df_open) - 1,
                                (df_close/df_open)-1)

df_short_ret = np.where((df_high > df_short_stop_level) & (df_signal_s.shift(1) == -1),
                                (df_close/df_open)-1,
                                (df_close/df_open)-1)


df_long_ret = pd.DataFrame(data=df_long_ret, index=df_low.index,columns   =df_low.columns )
df_short_ret = pd.DataFrame(data=df_short_ret, index=df_low.index,columns   =df_low.columns )


df_strat_ret_stops = + df_long_ret.shift(-1) * df_signal_l + df_short_ret.shift(-1) * df_signal_s
df_strat_ret_stops['TOTAL']  = (df_strat_ret_stops.sum(axis=1) - df_signal_l.astype(bool).sum(axis=1) * 0.00004 - df_signal_s.astype(bool).sum(axis=1) * 0.00004) \
                               / (df_signal_l.astype(bool).sum(axis=1) + df_signal_s.astype(bool).sum(axis=1))

df_strat_ret_stops['TOTAL']  = df_strat_ret_stops['TOTAL'] *100
df_strat_ret_stops['Cum_TOTAL'] = df_strat_ret_stops['TOTAL'] .cumsum()
df_strat_ret_stops['High_water_mark'] = df_strat_ret_stops['Cum_TOTAL'].cummax()
df_strat_ret_stops['Draw_Down'] = df_strat_ret_stops['High_water_mark'] - df_strat_ret_stops['Cum_TOTAL']
df_strat_ret_stops['DollarValue'] = df_strat_ret_stops['TOTAL']*100000
df_strat_ret_stops['DollarValue']=df_strat_ret_stops['DollarValue'].cumsum()


sharpe  = 250 * 24 *np.mean(df_strat_ret_stops['TOTAL']) / (np.std(df_strat_ret_stops['TOTAL']) *np.sqrt(24*250))


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






