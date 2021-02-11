


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