
import pandas as pd
# import sqlalchemy
import mysql.connector
from mysql.connector import Error
import numpy as np

#Create the engine
connection = mysql.connector.connect(host='ns3142407.ip-51-77-118.eu',
                             database='kaiseki_prices',
                             user='remote_user',
                             password='Password*8')
connection.is_connected()
#%%
with open('C:\\Repos\\Kaiseki Repo\\research\\firefly\\sqlforjake.txt', 'r') as file:
    stmt = file.read().strip()
#%%
def getLatestPrices():
    cursor = connection.cursor()
    cursor.execute(stmt)
    result = cursor.fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = [i[0] for i in cursor.description]
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df
#%%
trades = getLatestPrices()
#%%
trades = trades.drop_duplicates(subset='trace_id', keep='first')
trades = trades.sort_index(ascending=True)
#%%
trades = trades.sort_index(ascending=True)
trades['pnl'] = np.sign(trades['Quantity']) * trades['RealisedReturn']

#%%
trades['time'] = trades.index
trades['time'] = trades['time'].dt.floor("H")
#%%
trades_pivoted = trades.pivot_table(index='time', columns='symbol',values = 'Quantity',aggfunc=np.sum)
trades_pivoted = trades_pivoted.fillna(0)
#%%
cursor = connection.cursor()
cursor.execute("SELECT (bid + offer) / 2 as mid, timestamp, symbol FROM kaiseki_prices.shapshots_hourly sh, kaiseki_prices.symbols s "\
               "WHERE sh.symbol_id = s.id")
rows = cursor.fetchall()
prices = pd.DataFrame(rows)
prices.columns = [i[0] for i in cursor.description]
prices.set_index('timestamp', inplace=True)
#%%
prices_pivoted = prices.pivot_table(index='timestamp', columns='symbol', values = 'mid', aggfunc=np.sum)
prices_pivoted.index.names = ['time']

#%%
returns = prices_pivoted.pct_change()
#%%
returns.reset_index(inplace=True)
returns.set_index('time', inplace=True)
trades_pivoted.reset_index(inplace=True)
trades_pivoted.set_index('time', inplace=True)
#%%
returns = returns.loc[trades_pivoted.index,:]
df = trades_pivoted * returns
#%%
trades_pivoted.tail()
#%%
returns.tail()
#%%
df.tail()
