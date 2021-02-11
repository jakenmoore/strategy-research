
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib

#for making changes to the module
import utilities.Database_Utils
importlib.reload(utilities.Database_Utils)
from utilities.Database_Utils import  getTrades, prepareDB,getHourlyBarsPivoted, getPositions,Dollarise, getOrders
from utilities.PnL_Utils import  MarkToMarket

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

prepareDB()

# TO BE RUN DAILY

#%% Set the default volatility values and get data

_Vol = {}
_Vol["USDJPY"] = 0.000835;
_Vol["USDGBP"] = 0.000729;
_Vol["USDEUR"] = 0.000666;
_Vol["USDNOK"] = 0.000986;
_Vol["USDSEK"] = 0.000967;
_Vol["USDCHF"] = 0.000749;
_Vol["USDAUD"] = 0.000947;
_Vol["USDNZD"] = 0.000947;
_Vol["USDCAD"] = 0.000949;

start_date = '2019-06-05 00:00'
end_date   = '2019-06-06 00:00'
tickers = ['EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','USDJPY','USDNOK','NZDUSD','USDSEK']

df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,start_date,end_date,dollarise=True)



## Check which orders got filled
df_trades = getTrades(start_date,end_date)
df_orders = getOrders(start_date,end_date)
df_orders = df_orders.loc[df_orders['order_type'] == 'LMT']
df_orders = df_orders.loc[df_orders['orderStatus'] == 'Submitted']

df_fills = pd.merge(df_orders,df_trades, how='outer', left_on='order_id', right_on = 'order_id')
df_fills.to_csv('fills.csv')



#%% Check the slippage on market orders

df_trades = getTrades(start_date,end_date)
df_orders = getOrders(start_date,end_date)
df_orders = df_orders.loc[df_orders['order_type'] == 'MKT']
df_fills = pd.merge(df_orders,df_trades, how='inner', left_on='order_id', right_on = 'order_id')
df_fills.loc[df_fills['quantity_x'] >= 0, 'Slippage'] = df_fills.loc[df_fills['quantity_x'] > 0, 'price_y'] \
                                                       - df_fills.loc[df_fills['quantity_x'] > 0, 'offer_x']
df_fills.loc[df_fills['quantity_x'] <= 0, 'Slippage'] = df_fills.loc[df_fills['quantity_x'] < 0, 'price_y'] \
                                                       - df_fills.loc[df_fills['quantity_x'] < 0, 'bid_x']

print('Slippage on Market Orders')
print(df_fills[['symbol_x','Slippage']])


#%% Model performance
df_strat_result,df_signal,df_effective_close = RunStrategy(df_close, df_open, _Vol)
trace1 = go.Scatter(x=df_strat_result.index, y=df_strat_result['Cum_TOTAL'], name ='Firefly Performance')
layout = go.Layout(title='Firefly model performance')
data = [trace1]
fig = go.Figure(data=data, layout=layout)
plot(fig, filename = 'model-performance')

#%% conve-rt the model dataframes into a list of trades

def getTradesFromSignal(df_signal,df_open,df_effective_close):
    df_signal_shifted = df_signal.shift(1)
    trades = []
    for i in range(len(df_signal_shifted.index)):
        index = df_signal_shifted.index[i]
        for column in df_signal_shifted.columns:
        # crate a trade
            if (df_signal_shifted.loc[index, column] == 1) | (df_signal_shifted.loc[index, column] == -1) :
                #opening trade
                trade = {'datetime':index,'Buy/Sell':df_signal_shifted.loc[index,column],'price':df_open.loc[index,column],'symbol':column,'open':True}
                #closing trade
                trade2 = {'datetime': index, 'Buy/Sell': -1* df_signal_shifted.loc[index, column], 'price': df_effective_close.loc[index, column],
                         'symbol': column,'open':False}
                #closing trade
                trades.append(trade)
                trades.append(trade2)
                print(trade)

    df_model_trades = pd.DataFrame(trades)
    df_model_trades.index =  df_model_trades['datetime']
    return df_model_trades
df_model_trades = getTradesFromSignal(df_signal,df_open,df_effective_close)


#%% chart the prices for a symbol vs trades
symbol = 'USDNOK'
# Prices used in the model
trace1 = go.Scatter(x=df_open.index, y=df_open[symbol],name='Model Open Price')
# Trades
df_trades_single_asset = df_trades.loc[df_trades['symbol']==symbol]
trace2 = go.Scatter(x=df_trades_single_asset.index, y=df_trades_single_asset['price'],name='Trades')

# Limit Orders
#df_orders_single_asset = df_orders.loc[(df_orders['symbol']==symbol) & (df_orders['order_type']=='LMT')]
#trace3 = go.Scatter(x=df_orders_single_asset.index, y=df_orders_single_asset['price'],name='Limit Orders')

# Model Trades
df_model_trades_buys = df_model_trades.loc[(df_model_trades['symbol']==symbol) & (df_model_trades['Buy/Sell']==1) ]
trace4 = go.Scatter(x=df_model_trades_buys.index, y=df_model_trades_buys['price'],name='Model Trades',mode = 'markers',marker={'color': 'green', 'symbol': 15, 'size': 10})

df_model_trades_sells = df_model_trades.loc[(df_model_trades['symbol']==symbol)& (df_model_trades['Buy/Sell']==-1)]
trace5 = go.Scatter(x=df_model_trades_sells.index, y=df_model_trades_sells['price'],name='Model Trades',mode = 'markers',marker={'color': 'red', 'symbol': 15, 'size': 10})

layout = go.Layout(title=symbol)
data = [trace1,trace2,trace4,trace5]
fig = go.Figure(data=data, layout=layout)
plot(fig, filename = 'model-performance')






#%%get the realised performance
df_Realised = MarkToMarket(df_close,df_trades)

# A chart with the actual PnL
# % of trades filled
# average spread paid on the limits


# average spread paid on the market orders
    # get all market orders
    # get the underlying fills
    # check the difference between where the price was and where we got filled

# average slippage on the stops
# chart of bid/offer spread
# figure out historical data on IB
# automatic process for filling out the missing data
# master list of securities that we will save to the DB
# At what point to we cancel the orders (need minute by minute data for that)
# A Probability of being filled on the limit order
# B probability of mean reversion
# A * B and choose the best one












