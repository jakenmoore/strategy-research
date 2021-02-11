#%%
from pymongo import MongoClient
from arctic import Arctic
import quandl
import matplotlib.pyplot as plt
import numpy as np

"""
in the terminal 
sudo service mongod start
mongo # starts the mongodb shell
show dbs
use arctic
show collections
"""

#%%
myclient = MongoClient('mongodb://localhost:27017/')
mydb = myclient["testdb"]

#%% check it exists
print(myclient.list_database_names())

#%%
# Connect to Local MONGODB
store = Arctic('localhost')


#%%
store.initialize_library('cryptofeed-test')
library = store['cryptofeed-test']

#%%
trades = library.read("trades")

raw_trades = trades.data
trades = trades.data
#%%
# Filter out exchanges that are not bitmex
trades = trades[trades["feed"] == "BITMEX"]
trades.drop("id", inplace=True, axis=1)


trades["direction"] = trades.apply(lambda r: -1.0 * np.abs(r["amount"]) if
                                        r["side"] == "sell" else np.abs(r["amount"]), axis=1)



trades['long_sig'] = np.where(trades['direction'].rolling(50).sum() > 100000, 1, 0)
trades['short_sig'] = np.where(trades['direction'].rolling(50).sum() < -100000, -1, 0)

trades['returns'] = trades['price'].pct_change()

trades['long_pnl'] = trades['long_sig'].shift(1) * trades['returns']
trades['short_pnl'] = trades['short_sig'].shift(1) * trades['returns']
trades['total_pnl'] = trades['long_pnl'] + trades['short_pnl']
trades['cum_pnl'] = (1 + trades['total_pnl']).cumprod()
trades['cum_pnl'].plot()
plt.show()

#%% descriptive stats
trades['amount'].describe()

#%%
big_trades = trades[trades["amount"] > 1e3][["amount", "side", "price"]]
big_trades.head()

#%%

trades[['price', 'direction']].plot(subplots=True)
plt.show()