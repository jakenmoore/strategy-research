import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import datetime
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


# TODO  - check not for teh front crack but for the back
# TODO  - sweep the parameters
# TODO Calculate the returns properly
# TODO test on hourly data with higher frequency of trading

###!!!!! todo - FROM THE FIRST 6 GASOIL CRACKS BUYS THE CHEAPEST AND SELL THE MOST EXPENSIVE

def plot_price_series(df, ts1, ts2,ts2_scaling_factor = 1):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2]/ts2_scaling_factor, label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()

def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2018, 1, 1), datetime.datetime(2018, 12, 31))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()

def readTickDataFile(filename):
    df2 = pd.read_csv(filename, parse_dates={'datetime': [1, 2]})
    df2['contract_as_date'] = df2['Symbol'].apply(
        lambda x: dt.datetime(2000 + int(x[-2:]), 'FGHJKMNQUVXZ'.find(x[-3:-2]) + 1, 1))
    df2['month_symbol'] = df2['Symbol'].apply(lambda x: x[-3:])
    #df2.set_index('datetime', inplace=True)
    #df2 = df2.at_time('16:00')
    return df2

import glob
gasoil_files = [filename for filename in glob.glob("C:\\TickData\\TickWrite7\\DATA\\GO*.csv")]

df = None
for fileName in gasoil_files:
    gasoildf =  readTickDataFile(fileName)
    gasoildf['Mid'] = (gasoildf['Close Ask Price'] + gasoildf['Close Bid Price'])/2
    gasoildf.set_index('datetime', inplace=True)
    gasoildf = gasoildf.at_time('16:00')
    gasoildf['Return'] = gasoildf['Mid'].pct_change().shift(-1)
    gasoildf = gasoildf[['Mid','Return','month_symbol','Symbol','contract_as_date']]
    #remove the last 2 days before the roll
    gasoildf = gasoildf.iloc[:-2]
    if df is None:
        df = gasoildf
    else:
        df = df.append(gasoildf)

#Make a dataframe of brent
df_brent = None
brent_files = [filename for filename in glob.glob("C:\\TickData\\TickWrite7\\DATA\\CO*.csv")]

for fileName in brent_files:
    brent_temp =  readTickDataFile(fileName)
    brent_temp['Mid'] = (brent_temp['Close Ask Price'] + brent_temp['Close Bid Price'])/2
    # set index for brent
    brent_temp.set_index('datetime', inplace=True)
    brent_temp = brent_temp.at_time('16:00')
    brent_temp['Return'] = brent_temp['Mid'].pct_change().shift(-1)

    brent_temp = brent_temp[['Mid','Return','month_symbol','Symbol','contract_as_date']]
    #remove the last 2 days before the roll
    brent_temp = brent_temp.iloc[:-2]
    if df_brent is None:
        df_brent = brent_temp
    else:
        df_brent = df_brent.append(brent_temp)

df_brent["datetime_2"] = df_brent.index
df_brent["datetime_plus_6"] = df_brent["datetime_2"].dt.date + pd.DateOffset(months=6)
df_brent["datetime_plus_6"] = df_brent["datetime_plus_6"].apply(lambda x: x.replace(day=1))
df_brent  = df_brent.loc[df_brent["datetime_plus_6"].dt.date == df_brent["contract_as_date"].dt.date,:]

#Make a continuous dataframe of Brent
df_brent = df_brent.sort_values('contract_as_date', ascending=True).drop_duplicates(['datetime_2'])\
    .sort_values('datetime', ascending=True)

#Join the two dataframes
df_merged = df.merge(df_brent, how='inner', right_index=True, left_index=True,suffixes=('_GO', '_CO'))
df_merged = df_merged.loc[df_merged['month_symbol_GO'] == df_merged['month_symbol_CO']]


#plot the timeseries
plot_price_series(df_merged, "Mid_CO", "Mid_GO",7.45)

#plot the scatterplot
plot_scatter_series(df_merged, "Mid_CO", "Mid_GO")

# Calculate optimal hedge ratio "beta"
#res = pd.stats.api.ols(y=df['WLL'], x=df["AREX"])
res = sm.OLS(df_merged["Mid_CO"],df_merged["Mid_GO"]).fit()
print(res.params)
beta_hr = res.params["Mid_GO"]

# Calculate the residuals of the linear combination
df_merged["res"] = df_merged["Mid_CO"] -   beta_hr *df_merged["Mid_GO"]

# Plot the residuals
plot_residuals(df_merged)

import statsmodels.tsa.stattools as ts
cadf = ts.adfuller(df_merged["res"])
print(cadf)

result = sm.tsa.stattools.coint(df_merged["Mid_CO"], df_merged["Mid_GO"]) # get conintegration
pvalue = result[1]


#Results critical value of -0.15619662921824984, > 1% statistic of -3.4314407137058147
# - we cannot reject the null that they are not contegrated ie not cointegrated on the daily basis as such

# (-0.15619662921824984,
#  0.9435454187103106,
#  4,
#  251,
#  {'1%': -3.4566744514553016,
#   '5%': -2.8731248767783426,
#   '10%': -2.5729436702592023},
#  17.202604485329857)

