import pandas as pd

#%%
co_merged = pd.read_csv('/home/jake/Code/kaiseki/local_data/co_merged.csv')
co_merged.set_index("Date_Time", inplace=True)

#%%
co_trades_020218 = pd.read_csv('/home/jake/Code/kaiseki/local_data/TradesCO02022018.csv',
                               parse_dates=[['Date', 'Time']])
co_trades_020218.set_index('Date_Time', inplace=True)

#%%
co_OBL1_020218 = pd.read_csv('/home/jake/Code/kaiseki/local_data/ordersCO02022018.csv',
                             parse_dates=[['Date', 'Time']])
co_OBL1_020218.set_index('Date_Time', inplace=True)

#%%
co_trades_020218.drop("Market Flag", inplace=True, axis=1)
co_OBL1_020218.drop("Market Flag", inplace=True, axis=1)
co_OBL1_020218.drop("Quote Condition", inplace=True, axis=1)

#%%
co_trades_020218.sort_index(inplace=True)
co_OBL1_020218.sort_index(inplace=True)

#%%
co_OBL1_020218_resampled = co_OBL1_020218.resample("10ms").last()

#%%
co_merge_020218 = co_trades_020218.join(co_OBL1_020218_resampled, how="left")

#%%
co_merge_020218.loc[co_merge_020218.loc[:,'Price'] >= co_merge_020218.loc[:,'Ask Price'] ,'Bid/Ask']  ="BUY"
co_merge_020218.loc[co_merge_020218.loc[:,'Price'] <= co_merge_020218.loc[:,'Bid Price'] ,'Bid/Ask']  ="SELL"



#%%  Tom's Code

import pandas as pd
import numpy as np

# %%
co_trades_020218 = pd.read_csv('C:\TickData\TickWrite7\DATA\TradesCO02022018.csv', parse_dates=[['Date', 'Time']])
co_trades_020218.set_index('Date_Time', inplace=True)
# %%
co_OBL1_020218 = pd.read_csv('C:\TickData\TickWrite7\DATA\ordersCO02022018.csv', parse_dates=[['Date', 'Time']])
co_OBL1_020218.set_index('Date_Time', inplace=True)

# %%
co_trades_020218.drop("Market Flag", inplace=True, axis=1)
co_OBL1_020218.drop("Market Flag", inplace=True, axis=1)
co_OBL1_020218.drop("Quote Condition", inplace=True, axis=1)

# %%
co_trades_020218.sort_index(inplace=True)
co_OBL1_020218.sort_index(inplace=True)

# %%
co_OBL1_020218_resampled = co_OBL1_020218.resample("1ms").last()

# %%

co_merged = pd.read_csv('/home/jake/Code/kaiseki/local_data/co_merged.csv')
co_merged.set_index("Date_Time", inplace=True)
co_merge_020218 = co_trades_020218.join(co_OBL1_020218_resampled, how="left")
co_merge_020218.to_csv('merged.csv')


#%%
co_merged.loc[:, 'Volume_Weighted_Mid'] = (co_merged.loc[:, 'Bid Price'] * co_merged.loc[:,'Ask Size']
                                                 + co_merged.loc[:, 'Ask Price'] * co_merged.loc[:,'Bid Size']) \
                                                / (co_merged.loc[:, 'Bid Size'] + co_merged.loc[:,'Ask Size'])

co_merged.loc[:, 'Buy/Sell'] = 0
co_merged.loc[co_merged.loc[:, 'Price'] >= co_merged.loc[:, 'Volume_Weighted_Mid'], 'Buy/Sell'] = 1
co_merged.loc[co_merged.loc[:, 'Price'] <= co_merged.loc[:, 'Volume_Weighted_Mid'], 'Buy/Sell'] = -1
co_merged.loc[:, 'Signed Volume'] =co_merged.loc[:, 'Buy/Sell'] * co_merged.loc[:, 'Volume']

#%% resampling

co_merged.index = pd.to_datetime(co_merged.index, unit='ns')
#%%
#1merge first and then resample
co_merged_1m = co_merged.resample("1Min").agg({'Signed Volume':'sum','Bid Price':'last','Ask Price':'last', 'Volume': 'sum'})
# co_merged_resampled_minutes

windows = [10,20,50]
for window in windows:
        co_merged_1m["trade_flow_{}".format(window)] = co_merged_1m['Signed Volume'].rolling(window).sum()




