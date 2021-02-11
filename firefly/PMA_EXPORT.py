
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib

#for making changes to the module
import utilities.Database_Utils
importlib.reload(utilities.Database_Utils)
from utilities.Database_Utils import  getTrades, prepareDB,getHourlyBarsPivoted, getPositions,Dollarise, getOrders,getTradesAugmented
from utilities.PnL_Utils import  MarkToMarket

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

prepareDB()
start_date = '2019-10-01 00:00'
end_date   = '2019-10-10 00:00'
trades_df  = getTrades(start_date,end_date)
oi_df  = getTradesAugmented(start_date,end_date)

#create a dataframe for export
df_Export = pd.DataFrame()
df_Export['Ccy Pair'] = oi_df['symbol']
df_Export['Transaction Type'] = np.where(oi_df['quantity'] >=0, 'Purchase', 'Sell')
df_Export['FX Type'] = 'FX Spot'
df_Export['Currency (Security name)'] =  oi_df['symbol'].str.slice(0, 3)
df_Export['Strategy 1'] = oi_df['Strategy']
df_Export['Strategy 2'] = oi_df['Strategy']
df_Export['Client Fund'] = 'SRQ'
df_Export['Trader'] = 'Algo'
df_Export['Rate'] = oi_df['price']
df_Export['Quantity/ Buy Amount'] = oi_df['quantity']
df_Export['Trade/Sell currency'] =  oi_df['symbol'].str.slice(3, 6)
df_Export['Counterparty'] = 'IB'
df_Export['Custodian'] = 'DB'
df_Export['Trade Date'] = oi_df.index.date
df_Export['Settlement Date'] = oi_df.index.date
df_Export['Settlement Date'] = oi_df.index.date
df_Export['Fixing Date'] = ''
df_Export['Fixing Rate'] = ''
df_Export['Delivery Currency'] = ''
df_Export['Notes'] = oi_df['Trace']
df_Export['Data Migration Reference']  = 0
df_Export['Consideration  (Sell Amount)']= -1 * oi_df['quantity'] * oi_df['price']



