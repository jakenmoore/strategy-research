def signal_from_mid():
    df_all["xtx_bid"] = xtx_prices["Bid0"]
    df_all["xtx_offer"] = xtx_prices["Offer0"]
    df_all["xtx_mid"] = (df_all["xtx_bid"] + df_all["xtx_offer"]) / 2
    df_all["xtx_signal"] = np.where(df_all["xtx_mid"] > df_all["mid"], 1, -1)
    df_all["xtx_signal"] = df_all["xtx_signal"].shift(1)
    df_all["xtx_pnl"] = df_all["xtx_signal"] * df_all["mid_change"]
    df_all[["xtx_pnl", "adapt_pnl"]].cumsum().resample("1T").last().plot()


df_all["xtx_bid"] = xtx_prices["Bid0"]
df_all["xtx_offer"] = xtx_prices["Offer0"]
df_all["xtx_mid"] = (df_all["xtx_bid"] + df_all["xtx_offer"]) / 2
df_all["xtx_signal"] = np.where(df_all["xtx_mid"] > df_all["mid"], 1, -1)
df_all["xtx_signal"] = df_all["xtx_signal"].shift(1)
df_all["xtx_pnl"] = df_all["xtx_signal"] * df_all["mid_change"]
df_all[["xtx_pnl", "adapt_pnl"]].cumsum().resample("1T").last().plot()


def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

#%% model
# need to put all_rets outside the for loop so that each return series can be added
all_rets = pd.DataFrame(index=fx_c.index)
all_positions = pd.DataFrame(index=fx_c.index)
combined_rets = pd.DataFrame(index=fx_c.index)

win1 = 6
win2 = 24

win3 = 8
win4 = 32

win5 = 10
win6 = 40

win7 = 12
win8 = 48

win9 = 14
win10 = 56

win11 = 16
win12 = 64

win20 = 20
win200 = 200

for columnName in fx_c.columns:
    instrument = fx_c[columnName]
    signals = pd.DataFrame(index=fx_c.index)
    signal_20_200 = pd.DataFrame(index=fx_c.index)
    params = pd.DataFrame(index=fx_c.index)
    results = pd.DataFrame(index=fx_c.index)

    # create the signal parameters
    # Param 1
    params['avgwin1'] = instrument.ewm(span=win1).mean()
    params['avgwin2'] = instrument.ewm(span=win2).mean()
    params['volwin2'] = instrument.rolling(window=win2).std()

    # Param 2
    params['avgwin3'] = instrument.ewm(span=win3).mean()
    params['avgwin4'] = instrument.ewm(span=win4).mean()
    params['volwin4'] = instrument.rolling(window=win4).std()

    # Param 3
    params['avgwin5'] = instrument.ewm(span=win5).mean()
    params['avgwin6'] = instrument.ewm(span=win6).mean()
    params['volwin6'] = instrument.rolling(window=win6).std()

    # Param 4
    params['avgwin7'] = instrument.ewm(span=win7).mean()
    params['avgwin8'] = instrument.ewm(span=win8).mean()
    params['volwin8'] = instrument.rolling(window=win8).std()

    # Param 5
    params['avgwin9'] = instrument.ewm(span=win9).mean()
    params['avgwin10'] = instrument.ewm(span=win10).mean()
    params['volwin10'] = instrument.rolling(window=win10).std()

    # Param 6
    params['avgwin11'] = instrument.ewm(span=win11).mean()
    params['avgwin12'] = instrument.ewm(span=win12).mean()
    params['volwin12'] = instrument.rolling(window=win12).std()

    # Param 20_200
    params['20'] = instrument.ewm(span=win20).mean()
    params['200'] = instrument.ewm(span=win200).mean()

    # Param switch
    params['switch'] = params['volwin2'] - params['volwin8']
    params['zscore'] = zscore(instrument.pct_change(),96)
    params['zscoreswitch'] = (np.where(params['zscore'] > 1, -1, 0) + np.where(params['zscore'] < -1, 1, 0))

    #create trading signals

    signals['sig1'] = (params['avgwin1'] - params['avgwin2']) / params['volwin2']
    signals['sig2'] = (params['avgwin3'] - params['avgwin4']) / params['volwin4']
    signals['sig3'] = (params['avgwin5'] - params['avgwin6']) / params['volwin6']
    signals['sig4'] = (params['avgwin7'] - params['avgwin8']) / params['volwin8']
    signals['sig5'] = (params['avgwin9'] - params['avgwin10']) / params['volwin10']
    signals['sig6'] = (params['avgwin11'] - params['avgwin12']) / params['volwin12']

    #regime switching
    signals_sigswitch1 = np.where(params['switch'] > 0, signals['sig1'], signals['sig6'])
    signals_sigswitch2 = np.sign(params['zscoreswitch'] + signals_sigswitch1)

    # seprate series for 20 v 200
    signal_20_200 = params['20'] - params['200']

    # create the orders. Produces different ways of testing orders.
    signals['orders'] = signals.mean(axis=1)
    signals['orders1'] = signals.mean(axis=1).round(1)
    signals['orders2'] = np.sign(signals.mean(axis=1))
    signals['orders3'] = signals['orders'].apply(lambda x: 1 if x>=1 else (-1 if x <=-1 else 0))
    signals['orders20_200'] = np.sign(signal_20_200)
    signals['size'] = .01/((instrument.pct_change(1).rolling(window=120).std())*5).shift(1)
