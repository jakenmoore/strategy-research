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

#%% try with FX data
# fx_data = pd.read_csv('import numpy as np
# import pandas as pd
# from pylab import plt
# plt.style.use('seaborn')
# import cufflinks as cf
# cf.set_config_file(offline=True)

#%% download stock data
# sym = 'AAPL.O'
# fn = 'http://hilpisch.com/tr_eikon_ffm_workshop.csv'
# data = pd.read_csv(fn, index_col=0, parse_dates=True)[sym]
# data['returns'] = np.log(data[sym] / data[sym].shift(1))

#%% try with FX data
fx_data = pd.read_csv('/Users/jakemoore/OneDrive - Kaiseki/systems/code/Kaiseki_models/intradaydata.csv', parse_dates=['Date'], index_col='Date')



fx_c = pd.concat([fx_data['USDCAD_C'], fx_data['USDGBP_C'], fx_data['USDEUR_C'], fx_data['USDNOK_C'],
            fx_data['USDSEK_C'], fx_data['USDAUD_C'], fx_data['USDJPY_C']], axis=1)

data = fx_data['USDEUR_C']

data = pd.DataFrame(data)
data['returns'] = np.log(data / data.shift(1))

#%%
all_rets = pd.DataFrame(index=data.index)
all_positions = pd.DataFrame(index=data.index)
combined_rets = pd.DataFrame(index=data.index)



#%% Create bins
data.dropna(inplace=True)
np.digitize(data['returns'], bins=[0])
lags = 5

mu = data['returns'].mean()
v = data['returns'].std()

bins = [mu - v, mu, mu + v]
bins = [0]

#%% create lags
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = np.digitize(data['returns'].shift(lag), bins=bins)
    cols.append(col)

#%% feature creation v1
data = data.iloc[lags:]
(len(bins) + 1) ** lags
features = data[cols]
labels = np.sign(data['returns'])

#%% import and set up model
from sklearn import linear_model
from sklearn.svm import SVC

# model = linear_model.LogisticRegression(C=1)
model = SVC(C=1)
model.fit(features, labels)

#%%
# data['PRED'] = model.predict(features)
# data['strategy'] = data['PRED'] * data['returns']
# np.exp(data[['returns', 'strategy']].sum())
# data[['returns', 'strategy']].cumsum().apply(np.exp).plot()

# %% expanding training
windows = len(features)
res = data.copy()
for w in range(windows // 2, windows):
    model.fit(features.iloc[:w], labels.iloc[:w])
    res.iloc[w]['PRED'] = model.predict(features.iloc[w:w + 1])

res['strategy'] = res['PRED'] * res['returns']
res[['returns', 'strategy']].cumsum().apply(np.exp).plot()

fx_c = pd.concat([fx_data['USDCAD_C'], fx_data['USDGBP_C'], fx_data['USDEUR_C'], fx_data['USDNOK_C'],
            fx_data['USDSEK_C'], fx_data['USDAUD_C'], fx_data['USDJPY_C']], axis=1)

data = fx_data['USDEUR_C']

data = pd.DataFrame(data)
data['returns'] = np.log(data / data.shift(1))

#%%
all_rets = pd.DataFrame(index=data.index)
all_positions = pd.DataFrame(index=data.index)
combined_rets = pd.DataFrame(index=data.index)



#%% Create bins
data.dropna(inplace=True)
np.digitize(data['returns'], bins=[0])
lags = 20

mu = data['returns'].mean()
v = data['returns'].std()

bins = [mu - v, mu, mu + v]
bins = [0]

#%% create lags
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = np.digitize(data['returns'].shift(lag), bins=bins)
    cols.append(col)

#%% feature creation v1
data = data.iloc[lags:]
(len(bins) + 1) ** lags
features = data[cols]
labels = np.sign(data['returns'])

#%% import and set up model
from sklearn import linear_model
from sklearn.svm import SVC

# model = linear_model.LogisticRegression(C=1)
model = SVC(C=1)
model.fit(features, labels)

#%%
# data['PRED'] = model.predict(features)
# data['strategy'] = data['PRED'] * data['returns']
# np.exp(data[['returns', 'strategy']].sum())
# data[['returns', 'strategy']].cumsum().apply(np.exp).plot()

# %% expanding training
windows = len(features)
res = data.copy()
for w in range(windows // 2, windows):
    model.fit(features.iloc[:w], labels.iloc[:w])
    res.iloc[w]['PRED'] = model.predict(features.iloc[w:w + 1])

res['strategy'] = res['PRED'] * res['returns']
res['strategy'].cumsum().apply(np.exp).plot()
