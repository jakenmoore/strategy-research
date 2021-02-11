#%% import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

#%% connect to FXCM
import fxcmpy
TOKEN = '9f02b3ebed3020bdc675fb7bdfad7ee1aed8cd4f'
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo')
con.is_connected()

#%% list instruments available
instruments = con.get_instruments()
print(instruments)

#%% define functions we need
def candles(instrument):
    data = instrument[['askopen', 'askhigh', 'asklow', 'askclose']]
    data.columns = ['open', 'high', 'low', 'close']
    return data
def midclose(instument):
    mid = pd.DataFrame(instument[['askclose', 'bidclose']].mean(axis=1), columns=['midclose'])
    return mid


#%%
lags = 6
def generate_features(df, lags):
    df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_%s' % lag
        df[col] = np.sign(df['Returns'].shift(lag))
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols

#%% get data and add mid
data = con.get_candles('EUR/USD', period='H1', number=5000)



# generate featrure
data['Mid'] = midclose(data)
data, cols = generate_features(data, lags)
labels = np.sign(data['Returns'])

# Support Vector Machines
from sklearn import svm
model = svm.SVC(C=100)
#%%
model.fit(data[cols], labels)

#%%
model.fit(np.sign(data[cols]), np.sign(data['Returns']))
pred = model.predict(np.sign(data[cols]))

data['position'] = pred
data['strategy'] = data['position'] * data['Returns']
data[['Returns', 'strategy']].cumsum().apply(np.exp).plot()


#%% Make a prediction using the last values
model.predict(data[cols].iloc[-1].values.reshape(1, -1))

#%% Set up Automated trading
to_show = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']
ticks = 0
position = 0
tick_data = pd.DataFrame()
tick_resam = pd.DataFrame()

#%%
def automated_trading(data, df):
    # TODO : add in PNL calculation. Add in new global variable which keeps position info.
    global lags, model, ticks
    global tick_data, tick_resam, to_show
    global position
    ticks += 1
    t = datetime.datetime.now()
    if ticks % 1 == 0:
        print('%3d | %s | %7.5f | %7.5f' % (ticks, str(t.time()),
                                            data['Rates'][0], data['Rates'][1]))
    # Collect tick data
    tick_data = tick_data.append(pd.DataFrame({'Bid': data['Rates'][0], 'Ask': data['Rates'][1],
                                               'High': data['Rates'][2], 'Low': data['Rates'][3]}, index=[t]))

    # resample tick data
    tick_resam = tick_data[['Bid', 'Ask']].resample('10ms', label='right').last().ffill()
    tick_resam['Mid'] = tick_resam.mean(axis=1)

    if len(tick_resam) > lags + 2:
        # generate signal
        tick_resam, cols = generate_features(tick_resam, lags)
        tick_resam['Prediction'] = model.predict(tick_resam[cols])
        # enter long position
        if tick_resam['Prediction'].iloc[-2] >= 0 and position == 0:
            print('going long (first time)')
            position = 1
            order = con.create_market_buy_order('EUR/USD', 25)
        elif tick_resam['Prediction'].iloc[-2] >= 0 and position == -1:
            print('going long')
            position = 1
            order = con.create_market_buy_order('EUR/USD', 50)
        # emter a short position
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position == 0:
            print('going short (first time)')
            position = -1
            order = con.create_market_sell_order('EUR/USD', 25)
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position == 1:
            print('going short')
            position = -1
            order = con.create_market_sell_order('EUR/USD', 50)


    if ticks > 50:
        con.unsubscribe_market_data('EUR/USD')
        print('closing all positions')
        try:
            con.close_all()
        except:
            pass


#%% start trading model

con.subscribe_market_data('EUR/USD', (automated_trading,))


#%%
tick_resam.tail()

#%% test model with live data
tick_resam, cols = generate_features(tick_resam, lags)

#%%
tick_resam['prediction'] = model.predict(tick_resam[cols])
