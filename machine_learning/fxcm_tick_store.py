#%% import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

from arctic import TICK_STORE
from arctic import Arctic
import pandas as pd
import pytz
from datetime import datetime
from pymongo import MongoClient


#%% connect to MongoDB
myclient = MongoClient('mongodb://localhost:27017/')

db = Arctic('localhost')
lib = db.initialize_library('FX', lib_type=TICK_STORE)
library = db['FX']


#%% connect to FXCM
import fxcmpy
TOKEN = '9f02b3ebed3020bdc675fb7bdfad7ee1aed8cd4f'
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo')
con.is_connected()


#%%
def automated_trading(data, df):
    global lags, model, ticks
    global tick_data, tick_resam, to_show
    global position
    ticks += 1
    t = datetime.now()
    timezone = pytz.timezone("Europe/London")
    d_aware = timezone.localize(t)
    if ticks % 1 == 0:
        print('%3d | %s | %7.5f | %7.5f' % (ticks, str(d_aware.time()),
                                            data['Rates'][0], data['Rates'][1]))
    # Collect tick data
    tick_data = tick_data.append(pd.DataFrame({'Bid': data['Rates'][0], 'Ask': data['Rates'][1],
                                               'High': data['Rates'][2], 'Low': data['Rates'][3]}, index=[d_aware]))

    # resample tick data
    tick_resam = tick_data[['Bid', 'Ask']].resample('1s', label='right').last().ffill()
    tick_resam['Mid'] = tick_resam.mean(axis=1)


    if ticks > 5:
        library.write('EURUSD', tick_data, metadata={'source': 'FXCM'})
        con.unsubscribe_market_data('EUR/USD')
        print('closing connection')
        try:
            con.close_all()
        except:
            pass


#%% start trading model
ticks = 0
position = 0
tick_data = pd.DataFrame()
tick_resam = pd.DataFrame()
con.subscribe_market_data('EUR/USD', (automated_trading,))


