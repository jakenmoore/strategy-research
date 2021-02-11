import sqlalchemy
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import Column, String, Integer, Date
from sqlalchemy import *
from sqlalchemy import Table


def prepareDB(server = 'localhost',username='root', password='root'):
    #Create the engine
    global engine
    engine = sqlalchemy.create_engine('mysql+mysqlconnector://remote_user:Password*8@ns3142407.ip-51-77-118.eu/kaiseki_prices')
    #engine = sqlalchemy.create_engine('mysql://root:root@localhost/kaiseki_prices')
    global metadata
    metadata = MetaData()
    global connection
    connection = engine.connect()
    global session
    session = sessionmaker(engine)

def GetOrCreateTicker (ticker):
    # Reflect a table
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)
    # Check if the ticker exists
    stmt = select([symbolsTable])
    stmt = stmt.where(symbolsTable.columns.symbol==ticker)
    results = connection.execute(stmt).fetchall()
    #if the ticker doesnt exist, create it
    if len(results) ==0:
        symbol ={'id':0,'symbol':ticker}
        stmt = symbolsTable.insert().values(symbol)
        connection.execute(stmt)
        session.commit()
        #retrieve the id of the ticker
        stmt = select([symbolsTable])
        stmt = stmt.where(symbolsTable.columns.symbol==ticker)
        results = connection.execute(stmt).fetchall()
    if len(results) > 0:
        return results[0]['id']
    return "None"

def insertDailyPrices(ticker, crosses):
    # Get ticker Id
    tickerId = GetOrCreateTicker( ticker)
    # Reflect the table
    dailyPricesTable = Table('prices_daily', metadata, autoload=True, autoload_with=engine)
    # rename columns if needed
    crosses['timestamp'] = crosses.index
    crosses['symbol_id'] = tickerId
    crosses=crosses.rename(columns = {'PX_HIGH':'high'})
    crosses=crosses.rename(columns = {'PX_LAST':'close'})
    crosses=crosses.rename(columns = {'PX_OPEN':'open'})
    crosses=crosses.rename(columns = {'PX_LOW':'low'})
    crosses_dict  = crosses.to_dict('records')
    #insert data = must be new data
    for item in crosses_dict:
        stmt  = dailyPricesTable.insert().values(item)
        connection.execute(stmt)
    session.commit()
    return

def insertIntradayPrices(ticker, crosses):
    # Get ticker Id
    tickerId = GetOrCreateTicker(ticker)
    # Reflect the table
    dailyPricesTable = Table('prices_intraday', metadata, autoload=True, autoload_with=engine)
    # rename columns if needed
    crosses['symbol_id'] = tickerId
    crosses_dict  = crosses.to_dict('records')
    #insert data = must be new data
    for item in crosses_dict:
        stmt  = dailyPricesTable.insert().values(item)
        connection.execute(stmt)
    session.commit()
    return

def getPricesIntraday(ticker, startDate, endDate):
    # Reflect the table
    dailyPricesTable = Table('prices_intraday', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol== ticker
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id
                           , dailyPricesTable.columns.timestamp <= endDate
                           ,dailyPricesTable.columns.timestamp >= startDate))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol', 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getHourlyBars(ticker, startDate, endDate):
    # Reflect the table
    dailyPricesTable = Table('bars_hourly', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol== ticker
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id
                           , dailyPricesTable.columns.timestamp <= endDate
                           ,dailyPricesTable.columns.timestamp >= startDate))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop([ 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getHourlyBarsPivoted(tickers, startDate, endDate,dollarise = False):
    # Reflect the table
    dailyPricesTable = Table('bars_hourly', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol.in_(tickers)
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id
                           , dailyPricesTable.columns.timestamp <= endDate
                           ,dailyPricesTable.columns.timestamp >= startDate))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop([ 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.replace(to_replace=-1, method='ffill')

    #pivot the data and split into separate dataframes
    df = df.pivot(columns='symbol')
    df.dropna(inplace=True)
    df.columns = ['_'.join(col) for col in df.columns]

    df_close = df.filter(regex='close', axis=1).rename(columns=lambda x: str(x)[6:])
    if dollarise:
        df_close = Dollarise(df_close)

    df_open = df.filter(regex='open', axis=1).rename(columns=lambda x: str(x)[5:])
    if dollarise:
        df_open = Dollarise(df_open)

    df_high = df.filter(regex='high', axis=1).rename(columns=lambda x: str(x)[5:])
    if dollarise:
        df_high = Dollarise(df_high)

    df_low = df.filter(regex='low', axis=1).rename(columns=lambda x: str(x)[4:])
    if dollarise:
        df_low = Dollarise(df_close)

    return df_open,df_close,df_high,df_low

def getMinBarsPivoted(tickers, startDate, endDate,dollarise = False):
    # Reflect the table
    dailyPricesTable = Table('bars_minute', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol.in_(tickers)
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id
                           , dailyPricesTable.columns.timestamp <= endDate
                           ,dailyPricesTable.columns.timestamp >= startDate))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop([ 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.replace(to_replace=-1, method='ffill')

    #pivot the data and split into separate dataframes
    df = df.pivot(columns='symbol')
    df.dropna(inplace=True)
    df.columns = ['_'.join(col) for col in df.columns]

    df_close = df.filter(regex='close', axis=1).rename(columns=lambda x: str(x)[6:])
    if dollarise:
        df_close = Dollarise(df_close)

    df_open = df.filter(regex='open', axis=1).rename(columns=lambda x: str(x)[5:])
    if dollarise:
        df_open = Dollarise(df_open)

    df_high = df.filter(regex='high', axis=1).rename(columns=lambda x: str(x)[5:])
    if dollarise:
        df_high = Dollarise(df_high)

    df_low = df.filter(regex='low', axis=1).rename(columns=lambda x: str(x)[4:])
    if dollarise:
        df_low = Dollarise(df_close)

    return df_open,df_close,df_high,df_low

def Dollarise(df_close):
        df_close_dollarised = df_close.copy(deep=True)
        for col in df_close.columns:
            if col[0:3] != "USD":
                column_name = col[3:] + col[0:3]
                df_close_dollarised = df_close_dollarised.drop(col, axis=1)
                df_close_dollarised[column_name] = 1 / df_close[col]
        return df_close_dollarised

def getPrices(ticker, startDate, endDate):
    # Reflect the table
    dailyPricesTable = Table('prices_daily', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol== ticker
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id
                           , dailyPricesTable.columns.timestamp <= endDate
                           ,dailyPricesTable.columns.timestamp >= startDate))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol', 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getPrices(ticker):
    # Reflect the table
    dailyPricesTable = Table('prices_daily', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([dailyPricesTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.symbol== ticker
                           ,symbolsTable.columns.id ==dailyPricesTable.columns.symbol_id))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol', 'symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getSignals():
    # Reflect the table
    signalsTable = Table('signals', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([symbolsTable,signalsTable ])
    stmt = stmt.where(symbolsTable.columns.id == signalsTable.columns.symbol_id)

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol_id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getTrades(startDate, endDate):
    # Reflect the table
    tradesTable = Table('ordersandtrades', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([tradesTable,symbolsTable])
    stmt = stmt.where(and_(
        symbolsTable.columns.id == tradesTable.columns.symbol_id,
        tradesTable.columns.timestamp <= endDate
                           ,tradesTable.columns.timestamp >= startDate
                           ,tradesTable.columns.What_happened == 'Trade'))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol_id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getOrders(startDate, endDate):
    # Reflect the table
    tradesTable = Table('ordersandtrades', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([tradesTable,symbolsTable])
    stmt = stmt.where(and_(
        symbolsTable.columns.id == tradesTable.columns.symbol_id,
        tradesTable.columns.timestamp <= endDate
                           ,tradesTable.columns.timestamp >= startDate
                           ,tradesTable.columns.What_happened == 'Order'))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol_id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def getLatestPrices():
    # Reflect the table
    stmt = 'SELECT * FROM kaiseki_prices.shapshots_min sm, kaiseki_prices.symbols s ' \
           'where sm.symbol_id =s.id ' \
           'and  timestamp  = (select max(timestamp) from kaiseki_prices.shapshots_min) '

    print(stmt)
    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def getPositions():
    # Reflect the table
    positionsTable = Table('positions', metadata, autoload=True, autoload_with=engine)
    symbolsTable = Table('symbols', metadata, autoload=True, autoload_with=engine)

    stmt = select([positionsTable,symbolsTable])
    stmt = stmt.where(and_(symbolsTable.columns.id ==positionsTable.columns.symbol_id))

    result = connection.execute(stmt).fetchall()
    # Create a DataFrame from the results: df
    df = pd.DataFrame(result)
    # Set Column names
    df.columns = result[0].keys()
    df = df.drop(['symbol_id','id'], axis=1)
    df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

prepareDB()

df_open, df_close, df_high, df_low = getHourlyBarsPivoted(['GBPUSD'], "01-09-2019 00:00", "01-11-2019 00:00",dollarise = False)
start_date = '2019-06-05 00:00'
end_date   = '2019-06-06 00:00'
tickers = ['EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','USDJPY','USDNOK','NZDUSD','USDSEK']

df_open, df_close, df_high, df_low = getHourlyBarsPivoted(tickers,start_date,end_date,dollarise=True)
