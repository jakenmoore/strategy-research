import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to read the pickle files
def read_pickle(instrument, startdate, enddate, resample_period):
    path_ = ("/Users/jakemoore/Code/data/" + instrument + "_pickle")
    df = pd.read_pickle(path_).loc[startdate:enddate].resample(resample_period).ffill()
    return df


# DATASET
startdate = '2017-01-01'
enddate = '2018-12-31'
download_period = "1t"  # do not change

eurusd = read_pickle("EURUSD", startdate, enddate, download_period)
eurusd = eurusd[['open', 'high', 'low', 'close']]


def generate_features(df):
    """Generate various vols, trend and trigger levels"""
    df['o_to_h'] = np.log(df['high'] / df['open'])
    df['o_to_l'] = np.log(df['low'] / df['open'])
    df['c_to_c'] = np.log(df['close'] / df['close'].shift(1))
    df['c_to_c_vol'] = df['c_to_c'].rolling(120).std()
    df['o_to_h_vol'] = df['o_to_h'].rolling(120).std()
    df['o_to_l_vol'] = df['o_to_l'].rolling(120).std()
    df['high_trigger'] = df['open'] * (1 + (2 * df['o_to_h_vol'].shift(1)))
    df['low_trigger'] = df['open'] * (1 - (2 * df['o_to_l_vol'].shift(1)))
    df['6hr_ma'] = df['close'].rolling(6).mean()
    df['trend_sig'] = np.where(df['open'] > df['6hr_ma'], 1, -1)
    return df


def trading_logic(df):
    """Trading logic and pnl for breakout strategy"""
    if df['high'] >= df['high_trigger']:
        df['long_pnl'] = np.log(df['close'] / df['high'])
    else:
        df['long_pnl'] = 0

    # if df['low'] <= df['low_trigger']:
    #     df['short_pnl'] = (np.log(df['close'] / df['low'])) * -1
    # else:
    #     df['short_pnl'] = 0

    # add in final stage, trend_sig 
    return df


eurusd = generate_features(eurusd)
eurusd = trading_logic(eurusd)

