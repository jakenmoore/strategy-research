import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uuid import uuid4
from json import dumps
from math import floor
from importlib import reload
from itertools import product
import talib as ta
import yaml
import arrow
import numpy as np
import os
from numpy import random
import pandas as pd
import psycopg2 as pg
from psycopg2 import extensions
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogFormatter, StrMethodFormatter
import seaborn as sns

from sklearn.metrics import (classification_report, accuracy_score, log_loss,
                             roc_auc_score, roc_curve, auc, mean_absolute_error,
                             mean_squared_error, r2_score, explained_variance_score,
                             mean_squared_log_error, f1_score)
#%% load data
df = pd.read_pickle('/home/jake/Code/kaiseki/local_data/hourly_fx_hloc_pickle')
df = df.loc['2017':'2018']

#%%
feat_eng = df[['usdeur_open', 'usdeur_high', 'usdeur_low', 'usdeur_close']]
feat_eng.columns = ['open', 'high', 'low', 'close']


#%%
feat_eng['price_velocity'] = feat_eng['close'].pct_change()
feat_eng['price_acceleration'] =  feat_eng['price_velocity'].pct_change()
macd, macdsignal, macdhist = ta.MACD(feat_eng['close'], fastperiod=12, slowperiod=26, signalperiod=9)
feat_eng['macd'] = macd
feat_eng['macdsignal'] = macdsignal
feat_eng['macdhist'] = macdhist

for lookback in range(5, 51, 5):
    feat_eng['ema_{}'.format(lookback)] = ta.EMA(feat_eng['close'], timeperiod=lookback)
    upperband, middleband, lowerband = ta.BBANDS(feat_eng['close'], timeperiod=lookback, nbdevup=2, nbdevdn=2, matype=0)
    feat_eng['bband_up_{}'.format(lookback)] = upperband
    feat_eng['bband_mid_{}'.format(lookback)] = middleband
    feat_eng['bband_low_{}'.format(lookback)] = lowerband
    feat_eng['adx_{}'.format(lookback)] = ta.ADXR(feat_eng['high'], feat_eng['low'], feat_eng['close'], timeperiod=lookback)
    feat_eng['rocp_{}'.format(lookback)] = ta.ROCP(feat_eng['close'], timeperiod=lookback)


#%% Fix the data, replace any invalid values with 0
feat_eng.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

#%% Plot clustermap

sns.clustermap(feat_eng.corr())

#%% Creat labels

label = feat_eng.copy()[['close']]
label = label.shift(-1)
label.columns = ['label']
label = label.iloc[:-1]
label.tail()

#%%

# Align features data to match label
start, end = label.index.min(), label.index.max()
feat_eng = feat_eng.loc[start:end]
feat_eng.tail()

#%% create test / train split

from sklearn.model_selection import train_test_split
# NOTE: MAKE SURE shuffle=False for TS data
X_train, X_test, y_train, y_test = train_test_split(feat_eng, label, shuffle=False, test_size=0.85)
y_true = y_test.copy()


#%% Standardize Data: zscore,etc
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

# Initialize the standardizer
scaler = StandardScaler().fit(X_train)

# Transform the data
# NOTE: Do NOT fit on X_test, only fit on X_train
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

#Transform the target
scaler_label = StandardScaler().fit(y_train)
y_train = scaler_label.transform(y_train)

#%%
# Review the scaled data
pd.DataFrame(X_train, columns=feat_eng.columns).tail()

pd.DataFrame(y_train, columns=label.columns).tail()

#%% Create models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Create and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Review output - needs to be transformed
y_pred[0:10]

#%%
# Apply inverse scaler to predictions (get us back to $)
y_pred = scaler_label.inverse_transform(y_pred)

y_pred[0:10]

#%% review results
pred = pd.DataFrame(y_pred, index=y_true.index, columns=['predicted'])
pred = pred.shift(1)
y_true['predicted'] = pred

y_true.dropna(inplace=True)


