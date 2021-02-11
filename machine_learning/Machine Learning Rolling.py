import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('seaborn')
import cufflinks as cf
cf.set_config_file(offline=True)

#%% download stock data
# sym = 'AAPL.O'
# fn = 'http://hilpisch.com/tr_eikon_ffm_workshop.csv'
# data = pd.read_csv(fn, index_col=0, parse_dates=True)[sym]
# data['returns'] = np.log(data[sym] / data[sym].shift(1))



df = pd.read_pickle('/home/jake/Code/kaiseki/local_data/hourly_fx_close_pickle')

#%%
data = df['usdeur_close']
data['close'] = df['usdeur_close']


#%%
data = np.log(data['close'] / data['close'].shift(1))

#%%
data = data.dropna()

#%%
all_rets = pd.DataFrame(index=data.index)
all_positions = pd.DataFrame(index=data.index)
combined_rets = pd.DataFrame(index=data.index)



#%% Create bins
data.dropna(inplace=True)
np.digitize(data, bins=[0])
lags = 5

mu = data.loc['2010-01-01':'2011-12-31'].mean()
v = data.loc['2010-01-01':'2011-12-31'].std()

bins = [mu - v, mu, mu + v]
bins = [0]

#%% create lags
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = np.digitize(data.shift(lag), bins=True)
    cols.append(col)

#%% feature creation v1
data = data.iloc[lags:]
(len(bins) + 1) ** lags
features = data
labels = np.sign(data)

#%% import and set up model
from sklearn import linear_model
from sklearn.svm import SVC

# model = linear_model.LogisticRegression(C=1)
model = SVC(C=1)
model.fit(features, labels)

#%%
# TODO returns doesnt exist
data['PRED'] = model.predict(features)
data['strategy'] = data['PRED'] * data['returns']
np.exp(data[['returns', 'strategy']].sum())
data[['returns', 'strategy']].cumsum().apply(np.exp).plot()

# %% expanding training
windows = len(features)
res = data.copy()
for w in range(windows // 2, windows):
    model.fit(features.iloc[:w], labels.iloc[:w])
    res.iloc[w]['PRED'] = model.predict(features.iloc[w:w + 1])

res['strategy'] = res['PRED'] * res['returns']
res[['returns', 'strategy']].cumsum().apply(np.exp).plot()


