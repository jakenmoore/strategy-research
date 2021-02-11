import numpy as np


def data_processing(signal_tick_data, trigger_tick_data, reval_tick_data, resample_period, ma_samples):
    # Aggregate tick data
    signal_tick_data['spread'] = signal_tick_data['Offer0'] - signal_tick_data['Bid0']
    signal_mid_price_series = (signal_tick_data.loc[:, 'Bid0'] + signal_tick_data.loc[:, 'Offer0']) / 2
    trigger_mid_price_series = (trigger_tick_data.loc[:, 'Bid0'] + trigger_tick_data.loc[:, 'Offer0']) / 2
    reval_mid_price_series = (reval_tick_data.loc[:, 'Bid0'] + reval_tick_data.loc[:, 'Offer0']) / 2
    bar_sampler = signal_mid_price_series.resample(resample_period)
    hloc = bar_sampler.ohlc()
    # Derived columns
    hloc['o_to_h'] = (hloc['high'] / hloc['open'] - 1)
    hloc['o_to_l'] = (hloc['low'] / hloc['open'] - 1)
    hloc['c_to_c'] = hloc['close'].pct_change()
    hloc['o_to_h_vol'] = hloc['o_to_h'].rolling(ma_samples).std()
    hloc['o_to_l_vol'] = hloc['o_to_l'].rolling(ma_samples).std()
    hloc['c_to_c_vol'] = hloc['c_to_c'].rolling(ma_samples).std()
    hloc['6_hr_ma'] = hloc['close'].rolling(6).mean()
    hloc['6_hr_ma_sig'] = np.where(hloc['close'].shift(1) > hloc['6_hr_ma'].shift(1), 1, -1)
    return signal_tick_data, signal_mid_price_series, trigger_mid_price_series, reval_mid_price_series, hloc


#################
# Trigger check
#################

def mid_trigger_price(timestamp, where, trigger_mid_price_series=None, reval_mid_price_series=None):
    index = trigger_mid_price_series[timestamp: timestamp + timestamp.freq].where(where).dropna().first_valid_index()
    if index == None:
        return None
    result = reval_mid_price_series.asof(index)
    #     print(result)
    return result


def trigger_check(row, high_trigger_col, low_trigger_col, high_side):
    triggers = [
        ['High', mid_trigger_price(row.name, lambda price: price.gt(row[high_trigger_col])),
         +high_side, row[high_trigger_col]],
        ['Low', mid_trigger_price(row.name, lambda price: price.lt(row[low_trigger_col])),
         -high_side, row[low_trigger_col]]
    ]

    triggers = [t for t in triggers if t[1] is not None]

    triggers.sort(key=lambda t: t[1])

    if not triggers:
        return None

    return triggers[0]





