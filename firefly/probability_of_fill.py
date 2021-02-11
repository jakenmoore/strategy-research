'''
Author Jake Moore
------
Tests being filled on limit orders for entry
and
Stop losses for exit
------
'''

TODO: code up the limit order algorithm
TODO: add in the trading signals for the MR and TF systems
TODO: turn all the code into functions

import pandas as pd
import matplotlib.pyplot as plt
from read_data import read_pickle
import numpy as np

#%% load some data
instrument = "EURUSD"
startdate = '2000-01-01'
enddate = '2018-12-31'
resample_period = "1t"

eurusd = read_pickle("EURUSD", startdate, enddate, resample_period)
usdjpy = read_pickle("USDJPY", startdate, enddate, resample_period)

#%% concatenates two dataframes and keeps the original df names, not necessary for this
keys = ['usdjpy', 'eurusd']
both = pd.concat([usdjpy, eurusd], axis=1, keys=keys)

#%%
'''
This is to try to calculate the probability of being filled on the bid or the offer
Looks like 7% of the time the close that minute is the low for the next hour
The average low after any 1 minute is 5-6bps
Is it possible to say which is more likey to come true; no fill or a fill 2bps lower?
'''

df = eurusd # can change to other pairs

df['low60'] = df.loc[:,'low'].rolling(60).min().shift(-60) # this is the low in the next 60 mins
df['lower_60'] = ((df['close'])/df['low60'])-1
close_to_low = df['lower_60']

above_zero_count = ((0 < close_to_low)).sum()
count = len(df['lower_60'])
prob_bid_fill = (count / above_zero_count)-1
test1 = df['lower_60']

#%% plot a subset
small = df.loc['2018-05-01':'2018-05-05']
small.plot(y=['close', 'low60'])

#%%
''' begin the stop loss and take profit algorithms
one extension could be to cut the hour up into 15 minute chunks and reduce TP and SL for each period. 
This code uses a rolling standard deviation to calculate the level for the stop loss. I have chosen 1/5th
of the stdev as the level to stop, as we are picking up pennies with this approach. 
'''

np.random.seed(42) # keeps the random signals constant

# take out the position for normal OHLC
eurusd_1h = (eurusd.resample('1H').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                        'close': 'last', 'position': 'last'}))

# create random position
eurusd_1h['position'] = np.sign(np.random.normal(size=len(eurusd_1h)))

# define stop loss percent, currently 24 hours, could be longer, like 4 days
eurusd_1h['losscalc']= eurusd_1h['close'].pct_change().rolling(24).std() # could reduce this, 0.5*stdev

# calculate pnl for long trades with stop loss calculated above
eurusd_1h['long_ret'] = np.where(eurusd_1h['low'] < (eurusd_1h['open']*(1-eurusd_1h['losscalc'].shift(1))),
                                 -eurusd_1h['losscalc'].shift(1),
                                (eurusd_1h['close']/eurusd_1h['open'])-1)

# calculates the strategy pnl with and without stop losses
eurusd_1h['long_with_stop'] = np.where(eurusd_1h['position']>0, eurusd_1h['long_ret'], 0)
eurusd_1h['long_no_stop'] = np.where(eurusd_1h['position']>0, (eurusd_1h['close']/eurusd_1h['open'])-1, 0)

# calculate pnl for short trades with stop loss calculated above
eurusd_1h['short_ret'] = np.where(eurusd_1h['high'] > (eurusd_1h['open']*(1+eurusd_1h['losscalc'].shift(1))),
                                  -eurusd_1h['losscalc'].shift(1),
                                  ((eurusd_1h['close']/eurusd_1h['open'])-1)*-1)

# calculates the strategy pnl with and without stop losses
eurusd_1h['short_with_stop'] = np.where(eurusd_1h['position']<0, eurusd_1h['short_ret'], 0)
eurusd_1h['short_no_stop'] = np.where(eurusd_1h['position']<0, ((eurusd_1h['close']/eurusd_1h['open'])-1)*-1, 0)

# print the stats
print("longs no stop", eurusd_1h['long_no_stop'].sum())
print("longs with stop", eurusd_1h['long_with_stop'].sum())
print("shorts no stop", eurusd_1h['short_no_stop'].sum())
print("shorts with stop", eurusd_1h['short_with_stop'].sum())

# plot a chart of the sum combined pnl
print("combined annual return", (eurusd_1h['long_with_stop']+eurusd_1h['short_with_stop']).mean()*(252*24))
print("combined annual stdev",(eurusd_1h['long_with_stop']+eurusd_1h['short_with_stop']).std()*np.sqrt(252*24))

(eurusd_1h['long_with_stop']+eurusd_1h['short_with_stop']).cumsum().plot(title='combined sum pnl gain')

