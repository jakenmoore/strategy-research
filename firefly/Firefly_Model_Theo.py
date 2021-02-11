import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


#load the data
#df = pd.read_csv('C:\\Repos\\Kaiseki\\markets.csv', index_col='Timestamp', parse_dates=True)
df = pd.read_csv('C:\\Repos\\Kaiseki\\intradaydata\\intradaydata.csv', index_col='Date', parse_dates=True)
#df = pd.read_csv('C:\\Repos\\Kaiseki\\intradaydata.csv', index_col='Timestamp', parse_dates=True, dayfirst=True)
df = df.replace(to_replace=-1, method='ffill')
df.index = df.index.to_pydatetime()

df_close = df.filter(regex='_C', axis=1).rename(columns = lambda x : str(x)[:-2])
df_open = df.filter(regex='_O', axis=1).rename(columns = lambda x : str(x)[:-2])
df_high = df.filter(regex='_H', axis=1).rename(columns = lambda x : str(x)[:-2])
df_low = df.filter(regex='_L', axis=1).rename(columns = lambda x : str(x)[:-2])

#calculate the % returns
df_ret = df_close / df_open -1

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
df_signal_s = pd.DataFrame(data=np.where(df_rank == 8,-1,0), index=df_rank.index,columns   =df_rank.columns )
df_signal_l = pd.DataFrame(data=np.where(df_rank == 1,1,0), index=df_rank.index,columns   =df_rank.columns )


#calculate the stop levels
df_vol = df_close.pct_change().rolling(24).std().shift(1)

factorStop = 1

## Stops
df_Long_stop_level =(df_open*(1-df_vol*factorStop))
df_short_stop_level =(df_open*(1+df_vol*factorStop))

df_long_ret = np.where(df_low < df_Long_stop_level,
                       df_Long_stop_level/df_open -1,
                                (df_close/df_open)-1)

#df_long_ret = np.where(df_high > df_Long_tp_level,
#                       df_Long_tp_level/df_open -1,
#                       df_long_ret)

df_short_ret = np.where(df_high > df_short_stop_level,
                                 -1*(df_short_stop_level/df_open - 1),
                                -1*((df_close/df_open)-1))

#df_short_ret = np.where(df_low < df_short_tp_level,
#                                 -1 *(df_short_tp_level/df_open - 1),
#                        df_short_ret)

df_long_ret = pd.DataFrame(data=df_long_ret, index=df_low.index,columns   =df_low.columns )
df_short_ret = pd.DataFrame(data=df_short_ret, index=df_low.index,columns   =df_low.columns )


#calculate the return
df_strat_ret = df_ret.shift(1) * df_signal
df_strat_ret['TOTAL']  = df_strat_ret.sum(axis=1)
df_strat_ret['Cum_TOTAL'] = df_strat_ret['TOTAL'] .cumsum()
df_strat_ret['DollarValue'] = df_strat_ret['TOTAL']*100000 - 8
df_strat_ret['DollarValue']=df_strat_ret['DollarValue'].cumsum()

df_strat_ret_stops = df_long_ret.shift(1) * df_signal_l - df_short_ret.shift(1) * df_signal_s
df_strat_ret_stops['TOTAL']  = df_strat_ret_stops.sum(axis=1)
df_strat_ret_stops['Cum_TOTAL'] = df_strat_ret_stops['TOTAL'] .cumsum()
df_strat_ret_stops['DollarValue'] = df_strat_ret_stops['TOTAL']*100000 - 8
df_strat_ret_stops['DollarValue']=df_strat_ret_stops['DollarValue'].cumsum()


#First plot - without stops

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Cumulative Return', color=color)
ax1.plot(df_strat_ret.index, df_strat_ret['Cum_TOTAL'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Dollar Value', color=color)  # we already handled the x-label with ax1
ax2.plot(df_strat_ret.index, df_strat_ret['DollarValue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# rotate and align the tick labels so they look better
fig.autofmt_xdate()
plt.title('Without Stops')
plt.show()

### Second plot  - with stops

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Cumulative Return', color=color)
ax1.plot(df_strat_ret_stops.index, df_strat_ret_stops['Cum_TOTAL'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Dollar Value', color=color)  # we already handled the x-label with ax1
ax2.plot(df_strat_ret_stops.index, df_strat_ret_stops['DollarValue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# rotate and align the tick labels so they look better
fig.autofmt_xdate()
plt.title('With Stops')
plt.show()
