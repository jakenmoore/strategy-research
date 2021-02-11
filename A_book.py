# %% Import
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq

import math
import datetime
from datetime import timedelta
import scipy.stats

# %% Set the file names


filename_futures_quotes = "/Volumes/GoogleDrive/Shared drives/data/jeff_usdjpy_20200728_quotes.csv"
filename_futures_trades = "/Volumes/GoogleDrive/Shared drives/data/jeff_usdjpy_20200727_20200731_trades_clients"

date_filter = '2020-07-28'


# %% Define the ofi
def ofi(quotes, level):
    """Returns Order Flow Imbalance for given levels of the orderbook"""
    qdf = quotes.copy()
    bid_price_label = 'Bid' + str(level)
    offer_price_label = 'Offer' + str(level)
    bid_qty_label = 'Bid' + str(level) + 'Qty'
    offer_qty_label = 'Offer' + str(level) + 'Qty'

    qdf['prev_bidprice'] = qdf[bid_price_label].shift()
    qdf['prev_bidsize'] = qdf[bid_qty_label].shift()
    qdf['prev_askprice'] = qdf[offer_price_label].shift()
    qdf['prev_asksize'] = qdf[offer_qty_label].shift()

    # Fix any missing/invalid data
    qdf.replace([np.inf, np.NINF], np.nan, inplace=True)
    qdf.fillna(method="ffill", inplace=True)
    qdf.fillna(method="bfill", inplace=True)

    bid_geq = qdf[bid_price_label] >= qdf['prev_bidprice']
    bid_leq = qdf[bid_price_label] <= qdf['prev_bidprice']
    ask_geq = qdf[offer_price_label] >= qdf['prev_askprice']
    ask_leq = qdf[offer_price_label] <= qdf['prev_askprice']

    qdf['ofi'] = np.zeros(len(qdf))
    qdf['ofi'].loc[bid_geq] += qdf[bid_qty_label].loc[bid_geq]
    qdf['ofi'].loc[bid_leq] -= qdf['prev_bidsize'].loc[bid_leq]
    qdf['ofi'].loc[ask_geq] += qdf['prev_asksize'].loc[ask_geq]
    qdf['ofi'].loc[ask_leq] -= qdf[offer_qty_label].loc[ask_leq]
    return qdf['ofi']


# %% VWM
def vwm_single(df_all, level):
    bid_price_label = 'Bid' + str(level)
    offer_price_label = 'Offer' + str(level)
    bid_qty_label = 'Bid' + str(level) + 'Qty'
    offer_qty_label = 'Offer' + str(level) + 'Qty'
    return (df_all[bid_qty_label] * df_all[offer_price_label] + df_all[bid_price_label] * df_all[offer_qty_label]) / (
                df_all[bid_qty_label] + df_all[offer_qty_label])


def vwm_0_to_4(df_all):
    df_all['weighted_bid_5'] = (df_all["Bid0Qty"] * df_all["Bid0"] + df_all["Bid1Qty"] * df_all["Bid1"] + df_all[
        "Bid2Qty"] * df_all["Bid2"] + df_all["Bid3Qty"] * df_all["Bid3"] + df_all["Bid4Qty"] * df_all["Bid4"]) / (
                                           df_all["Bid0Qty"] + df_all["Bid1Qty"] + df_all["Bid2Qty"] + df_all[
                                       "Bid3Qty"] + df_all["Bid4Qty"])
    df_all['weighted_offer_5'] = (df_all["Offer0Qty"] * df_all["Offer0"] + df_all["Offer1Qty"] * df_all["Offer1"] +
                                  df_all["Offer2Qty"] * df_all["Offer2"] + df_all["Offer3Qty"] * df_all["Offer3"] +
                                  df_all["Offer4Qty"] * df_all["Offer4"]) / (
                                             df_all["Offer0Qty"] + df_all["Offer1Qty"] + df_all["Offer2Qty"] + df_all[
                                         "Offer3Qty"] + df_all["Offer4Qty"])
    df_all['weighted_bid_notional_5'] = df_all["Bid0Qty"] + df_all["Bid1Qty"] + df_all["Bid2Qty"] + df_all["Bid3Qty"] + \
                                        df_all["Bid4Qty"]
    df_all['weighted_offer_notional_5'] = df_all["Offer0Qty"] + df_all["Offer1Qty"] + df_all["Offer2Qty"] + df_all[
        "Offer3Qty"] + df_all["Offer4Qty"]
    return (df_all['weighted_bid_5'] + df_all['weighted_offer_5']) / 2


# %% Read the order book data for futures
df_quotes = pd.read_csv(filename_futures_quotes)
df_quotes['t'] = pd.to_datetime(df_quotes['t'], errors='coerce')
df_quotes.set_index("t", inplace=True)
# df_all  = df_all.resample("100ms").last()

df_quotes["Offer0Qty"] = df_quotes["Offer0Qty"].astype('float')
df_quotes["Offer0"] = df_quotes["Offer0"].astype('float')
df_quotes["Bid0"] = df_quotes["Bid0"].astype('float')
df_quotes["Bid0Qty"] = df_quotes["Bid0Qty"].astype('float')

df_quotes["Offer1Qty"] = df_quotes["Offer1Qty"].astype('float')
df_quotes["Offer1"] = df_quotes["Offer1"].astype('float')
df_quotes["Bid1"] = df_quotes["Bid1"].astype('float')
df_quotes["Bid1Qty"] = df_quotes["Bid1Qty"].astype('float')

df_quotes["Offer2Qty"] = df_quotes["Offer2Qty"].astype('float')
df_quotes["Offer2"] = df_quotes["Offer2"].astype('float')
df_quotes["Bid2"] = df_quotes["Bid2"].astype('float')
df_quotes["Bid2Qty"] = df_quotes["Bid2Qty"].astype('float')

df_quotes["Offer3Qty"] = df_quotes["Offer3Qty"].astype('float')
df_quotes["Offer3"] = df_quotes["Offer3"].astype('float')
df_quotes["Bid3"] = df_quotes["Bid3"].astype('float')
df_quotes["Bid3Qty"] = df_quotes["Bid3Qty"].astype('float')

df_quotes["Offer4Qty"] = df_quotes["Offer4Qty"].astype('float')
df_quotes["Offer4"] = df_quotes["Offer4"].astype('float')
df_quotes["Bid4"] = df_quotes["Bid4"].astype('float')
df_quotes["Bid4Qty"] = df_quotes["Bid4Qty"].astype('float')

# add mid
df_quotes['mid'] = (df_quotes['Bid0'] + df_quotes['Offer0']) / 2
df_quotes['mid_change'] = ((df_quotes['Bid0'] + df_quotes['Offer0']) / 2.0).pct_change()

# resrict the data to a single day only
df_quotes = df_quotes.loc[date_filter]

# remove the duplicates
df_quotes = df_quotes.loc[~df_quotes.index.duplicated(keep='last')]

# resample
# df_quotes = df_quotes.resample("100ms").last().ffill()
# df_quotes = df_quotes

# cast to float
df_quotes["Offer0Qty"] = df_quotes["Offer0Qty"].astype('float')
df_quotes["Offer0"] = df_quotes["Offer0"].astype('float')
df_quotes["Bid0"] = df_quotes["Bid0"].astype('float')
df_quotes["Bid0Qty"] = df_quotes["Bid0Qty"].astype('float')

# %%#######################
# OFI using top x levels 
########################

# The OFI needs to be done an a resampled dataset
df_quotes = df_quotes.resample("100ms").last().ffill()

# 100 period MA works well 
df_quotes['ofi'] = ofi(df_quotes, 0)
df_quotes['ofi_signal'] = np.where(df_quotes['ofi'].rolling(100).mean() > 0, 1, -1)

# %%#######################
# OFI using top x levels performance
########################

# Shift the signal
df_quotes['ofi_signal'] = df_quotes['ofi_signal'].shift(1)
df_quotes['ofi_pnl'] = (df_quotes['ofi_signal'] * df_quotes['mid_change'])
print("Cumulative PnL " + str(df_quotes['ofi_pnl'].cumsum().iloc[-1]))

# Plot
df_quotes['ofi_pnl'].cumsum().resample("1T").last().plot()

# %%add a suffix
df_quotes = df_quotes.add_suffix('_quotes')

# %% Read the data
# Load trades
df_trades = pd.read_csv(filename_futures_trades + ".csv")

# Function that adds the rolling long ash short position to the dataframe and average fill
df_trades['buys_factor'] = 0  # delete after use
df_trades['sells_factor'] = 0  # delete after use

df_trades.loc[df_trades['side'] == 'sell', 'sells_factor'] = df_trades['fillPrice'] * df_trades['filledQuantity']
df_trades.loc[df_trades['side'] == 'buy', 'buys_factor'] = df_trades['fillPrice'] * df_trades['filledQuantity']
df_trades.loc[df_trades['side'] == 'sell', 'cum_sell'] = df_trades.loc[
    df_trades['side'] == 'sell', 'filledQuantity'].cumsum()
df_trades.loc[df_trades['side'] == 'buy', 'cum_buys'] = df_trades.loc[
    df_trades['side'] == 'buy', 'filledQuantity'].cumsum()
df_trades.loc[df_trades['side'] == 'sell', 'average_sell_price'] = df_trades['sells_factor'].cumsum() / df_trades[
    'cum_sell']
df_trades.loc[df_trades['side'] == 'buy', 'average_buy_price'] = df_trades['buys_factor'].cumsum() / df_trades[
    'cum_buys']

# Set the index
df_trades["timestamp"] = df_trades["transactTime"]
df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])

df_trades.set_index("timestamp", inplace=True)
df_trades.sort_index(inplace=True)

# drop unused columns
# df_trades.to_parquet(filename_trades + ".parq", compression="snappy")

# df_trades = df_trades[['fillPrice','filledQuantity','side','cum_sell','average_sell_price','cum_buys','average_buy_price']]
df_trades = df_trades.loc[date_filter]

# %% Calcuate the weighted averages for the trades
import numpy as np


def weighted_average(group):
    weights = group['filledQuantity']
    height = group['fillPrice']
    return np.average(height, weights=weights)


grouped = df_trades.groupby(['timestamp', 'side']).apply(weighted_average).unstack()
grouped = grouped.add_suffix('_price')
grouped2 = df_trades.groupby(['timestamp', 'side'])['filledQuantity'].apply(sum).unstack()
grouped2 = grouped2.add_suffix('_qty')
grouped = pd.concat([grouped, grouped2], axis=1, sort=True)
grouped = grouped[['buy_price', 'sell_price', 'buy_qty', 'sell_qty']]
grouped.columns = ['OFFER_price', 'BID_price', 'OFFER_qty', 'BID_qty']
grouped.index = pd.to_datetime(grouped.index)

# %% Join the CFDs with the future quotes
result_final = pd.concat([grouped, df_quotes], axis=1, sort=True)
result_final = result_final[
    ['BID_price', 'OFFER_price', 'BID_qty', 'OFFER_qty', 'ofi_signal_quotes', 'Bid0_quotes', 'Offer0_quotes',
     'mid_quotes']]
result_final.sort_index(inplace=True)

cols = ['Bid0_quotes', 'Offer0_quotes', 'ofi_signal_quotes']
result_final.loc[:, cols] = result_final.loc[:, cols].fillna(method="ffill")

# Calculate the mid
result_final['mid_quotes'] = (result_final['Bid0_quotes'] + result_final['Offer0_quotes']) / 2

cols = ['BID_qty', 'OFFER_qty']
result_final.loc[:, cols] = result_final.loc[:, cols].fillna(0)

# Trim the dataframe to the point of the first trade
result_final = result_final.loc[result_final.index >= grouped.index[0]]

# %%
# del grouped
# del grouped2
# del df_quotes
# del df_trades

# %%
result_final['delayed_bid'] = np.nan
result_final['delayed_offer'] = np.nan
result_final['delayed_mid'] = result_final['mid_quotes']

delayed_mid_index = result_final.columns.get_loc('delayed_mid')
delayed_bid_index = result_final.columns.get_loc('delayed_bid')
delayed_offer_index = result_final.columns.get_loc('delayed_offer')
BID_price_index = result_final.columns.get_loc('BID_price')
OFFER_price_index = result_final.columns.get_loc('OFFER_price')
ofi_signal_quotes_index = result_final.columns.get_loc('ofi_signal_quotes')
Bid0_CFD_index = result_final.columns.get_loc('Bid0_quotes')
Offer0_CFD_index = result_final.columns.get_loc('Offer0_quotes')

# 'BID_price','OFFER_price','BID_qty','OFFER_qty','Bid0_CFD', 'Offer0_CFD','ofi_signal_quotes'
result_final_columns = result_final.columns
array = result_final.values
for i in range(array.shape[0]):
    # We want to sell but the signal says up
    if (array[i, BID_price_index] > 0) & (array[i, ofi_signal_quotes_index] > 0):
        # Delay the execution
        for i_inner in range(i, array.shape[0]):
            if array[i_inner, ofi_signal_quotes_index] < 0:
                array[i, delayed_bid_index] = array[i_inner, Bid0_CFD_index]
                array[i, delayed_mid_index] = (array[i_inner, Bid0_CFD_index] + array[i_inner, Offer0_CFD_index]) / 2
                break
    if (array[i, OFFER_price_index] > 0) & (array[i, ofi_signal_quotes_index] < 0):
        # Delay the execution
        for i_inner in range(i, array.shape[0]):
            if array[i_inner, ofi_signal_quotes_index] > 0:
                array[i, delayed_offer_index] = array[i_inner, Offer0_CFD_index]
                array[i, delayed_mid_index] = (array[i_inner, Bid0_CFD_index] + array[i_inner, Offer0_CFD_index]) / 2
                break

result_final = pd.DataFrame(data=array, index=result_final.index, columns=result_final_columns)

del array

# %% Look at the savings

mean = df_trades['filledQuantity'].mean()
result_final['saving'] = 0
result_final.loc[result_final['BID_qty'] > 0, 'saving'] = result_final['delayed_mid'] - result_final['mid_quotes']
result_final.loc[result_final['OFFER_qty'] > 0, 'saving'] = result_final['mid_quotes'] - result_final['delayed_mid']
result_final['saving'].cumsum().plot()

result_final['saving_pct'] = 0
result_final.loc[result_final['BID_qty'] > 0, 'saving_pct'] = (result_final['delayed_mid'] - result_final[
    'mid_quotes']) / result_final['mid_quotes']
result_final.loc[result_final['OFFER_qty'] > 0, 'saving_pct'] = (result_final['mid_quotes'] - result_final[
    'delayed_mid']) / result_final['delayed_mid']
result_final['saving_pct'].cumsum().plot()

(result_final['saving_pct'] * mean).cumsum().plot()

# result_final.loc[(result_final['OFFER_qty'] > 0) | (result_final['BID_qty'] > 0),'saving']
# result_final.loc[result_final['saving']>0,'saving'].count() / result_final.loc[result_final['saving']<0,'saving'].count()
