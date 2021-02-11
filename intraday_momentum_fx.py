#%% import the libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


#%%
def read_pickle(instrument, startdate, enddate, resample_period):
    path_ = ('/Users/jakemoore/Code/kaiseki/data/' + instrument + "_pickle")
    df = pd.read_pickle(path_).loc[startdate:enddate].resample(resample_period).last()
    return df

#%% load some data
instrument = ""
startdate = '2015-01-01'
enddate = '2018-12-31'
resample_period = "1t"

#%%

eurusd = read_pickle("EURUSD", startdate, enddate, resample_period)
eurusd= eurusd[['open', 'high', 'low', 'close']]
eurusd= 1 / eurusd
eurusd.columns = [['eurusd_open', 'eurusd_high', 'eurusd_low', 'eurusd_close']]

usdaud = read_pickle("AUDUSD", startdate, enddate, resample_period)
usdaud= usdaud[['open','high','low','close']]
usdaud= 1/usdaud
usdaud.columns = [['usdaud_open','usdaud_high','usdaud_low','usdaud_close']]

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

usdnzd = read_pickle("NZDUSD", startdate, enddate, resample_period)
usdnzd= usdnzd[['open','high','low','close']]
usdnzd= 1/usdnzd
usdnzd.columns = [['usdnzd_open','usdnzd_high','usdnzd_low','usdnzd_close']]

#%% put all the dataframes together

df = pd.concat([eurusd, usdaud, usdcad, usdchf, usdjpy, usdgbp, usdnok, usdsek], axis=1)

df = pd.concat([eurusd['usdeur_close'], usdaud['usdaud_close'], usdcad['usdcad_close'], usdchf['usdchf_close'], usdjpy['usdjpy_close'], usdgbp['usdgbp_close'], usdnok['usdnok_close'], usdsek['usdsek_close'], usdnzd['usdnzd_close']], axis=1)
df.index = df.index.to_pydatetime()
df.sort_index(inplace=True)
df = df[df.index.dayofweek < 5]

fx_c = df.dropna()

#%% import the data and create new dataframe of fx returns
# fx_data = pd.read_csv('C:\\Users\\JakeMoore\\Documents\\kaiseki\\Kaiseki_models\\intradaydata.csv', parse_dates=['Date'], index_col='Date')
#
# fx_c = pd.concat([fx_data['USDCAD_C'], fx_data['USDGBP_C'], fx_data['USDEUR_C'], fx_data['USDNOK_C'],
#             fx_data['USDSEK_C'], fx_data['USDAUD_C'], fx_data['USDJPY_C']], axis=1)

#%% create performance stats
def performance_stats(returns):
    window = 252*24*1440
    annual_return = returns.mean() * window
    annual_stdev = returns.std() * np.sqrt(window)
    sharpe = annual_return/annual_stdev
    cum_return = (1 + returns).cumprod()
    max_pnl = cum_return.rolling(window, min_periods=1).max()
    max_dd = cum_return - max_pnl
    print("Ann Return " + str(round(annual_return, 2)))
    print("Ann St Dev " + str(round(annual_stdev, 2)))
    print("Sharpe " + str(round(sharpe, 2)))
    print("Max DD " + str(round(max_dd.min(), 2)))

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


    # create positions using raw signals, rounded, or >1, <-1
    # orders3 is >1, <-1  which produces far fewer trading signals
    results['price'] = instrument
    results['pctchg'] = instrument.pct_change()
    results['momentum_rets'] = results['pctchg'] * signals_sigswitch2.shift(1) * signals['size']
    results['mom_cum_pnl'] = (1 + results['momentum_rets']).cumprod()


    # Add in trend folliwing to the PNL
    results['pnlavgwin2'] = results['mom_cum_pnl'].ewm(span=win2).mean()
    results['pnlsig'] = np.where(results['mom_cum_pnl'] > results['pnlavgwin2'], 0, 1)
    results['pnl2'] = results['momentum_rets'] * results['pnlsig'].shift(1)


    all_rets[columnName] = results['pnl2'] # replaced momentum_rets


    all_positions[columnName] = signals['orders20_200']


combined_rets = all_rets.mean(axis=1)
combined_cum_rets = (1 + combined_rets).cumprod()
combined_cum_rets.plot(grid=True)
plt.show()
performance_stats(combined_rets)


#%%

rets_daily = combined_rets.resample('B').sum()
rets_daily.to_csv('rets_daily.csv')

#%%
combined_rets.to_csv('fxmomrets.csv')

combined_rets.to_pickle('fxsmom')

#%% look at individual returns
test = pd.DataFrame(index=fx_c)
test = pd.DataFrame(index=fx_c.index)
test['usdeur'] = fx_c['USDEUR_C']
test['usdeur_position'] = all_positions['USDEUR_C']
test.plot()

#%% Generate breakout from R Carver blog post

class Breakout:
    def __init__(self, name: str, periods: int, smoothing: int, **kwargs):
        """
        Uses a continuous version of a breakout model
        Same approach as momentum about, various different
        lookbacks and the signal is the distance from the min or max

        Args
        name : output column
        periods : lookback periods

        """

        self.name = name
        self.periods = periods
        self.smoothing = smoothing

    def __call__(self, instrument, context, *args, **kwargs):
        rolling = instrument.rolling(self.periods)
        roll_min, roll_max = rolling.min(), rolling.max()
        roll_mean = (roll_min + roll_max) / 2
        signal = 2 *  (instrument - roll_mean) / (roll_max - roll_min)
        return signal.ewm(span=self.smoothing).mean().iloc[-1]

bout = Breakout(eurusd['usdeur_close'], 50, 10)



#%% begin work on Markov Switch


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from io import BytesIO

#%%
fx_pair = usdjpy


raw = usdjpy['usdjpy_close'].pct_change(1)
raw = raw[raw.index.dayofweek < 5]
raw = raw.dropna()
# dta_kns = raw.loc[:'2018'] - raw.loc[:'2018'].mean()
dta_kns = raw.loc['2010':'2012'] - raw.loc['2010':'2012'].mean()


# Plot the dataset
dta_kns['usdjpy_close'].plot(title='Excess returns', figsize=(12, 3))

# Fit the model
mod_kns = sm.tsa.MarkovRegression(dta_kns, k_regimes=2, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()



#%%
fig, axes = plt.subplots(3, figsize=(10,7))

ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime')

ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a high-variance regime')

ax = axes[2]
ax.plot(usdjpy['usdjpy_close'])
ax.set(title='USDJPY')

fig.tight_layout()
plt.show()

jake = sm.tsa.MarkovRegression()

#%%
windows = len(features)
res = data.copy()
for w in range(windows // 2, windows):
    model.fit(features.iloc[:w], labels.iloc[:w])
    res.iloc[w]['PRED'] = model.predict(features.iloc[w:w + 1])

res['strategy'] = res['PRED'] * res['returns']
res[['returns', 'strategy']].cumsum().apply(np.exp).plot()



#%% Mean reversion for market making



eurusd.columns = ['eurusd_open', 'eurusd_high', 'eurusd_low', 'eurusd_close']

#clean up the data a bit
eurusd = eurusd.loc[:, ['eurusd_close']]
eurusd = eurusd.fillna(method='ffill')

#%% create moving average

max_length = 25
eurusd['MA_slow'] = eurusd['eurusd_close'].rolling(max_length).mean()

#%% create signal
eurusd['signal']= 0
eurusd.loc[(eurusd['eurusd_close'] > eurusd['MA_slow']), 'signal'] = -1
eurusd.loc[(eurusd['eurusd_close'] < eurusd['MA_slow']), 'signal'] = 1

#%% returns & strategy returns
eurusd['return'] = (eurusd['eurusd_close'] - eurusd['eurusd_close'].shift(1)) / eurusd['eurusd_close'].shift(1)
eurusd['strategy_return'] = eurusd['return'] * eurusd['signal'].shift(1)
eurusd['strategy_return_cum'] = eurusd['strategy_return'].cumsum()

#%%
#Get a subset of the data for charting
eurusd_subset = eurusd.resample('60t').last()

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#Chart the returns on the strategy
trace1 = go.Scatter(x=eurusd_subset.index, y=eurusd_subset['strategy_return_cum'], name ='Returns_cum')
data = [trace1]
fig = go.Figure(data=data)
plot(fig, filename = 'Returns_cum.html')