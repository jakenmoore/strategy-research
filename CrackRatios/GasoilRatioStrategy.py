import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import datetime

# TODO  - check not for teh front crack but for the back
# TODO  - sweep the parameters
# TODO Calculate the returns properly
# TODO test on hourly data with higher frequency of trading

###!!!!! todo - FROM THE FIRST 6 GASOIL CRACKS BUYS THE CHEAPEST AND SELL THE MOST EXPENSIVE



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

# Calculate gasoil crack and the ratios

width = 2
MA_length = 10
STD_length = 10



#df_merged['Crack_Mid'] = df_merged['Mid_GO'] /7.45 - df_merged['Mid_CO']
#df_merged['Crack_STD'] = df_merged['Crack_Mid'].rolling(STD_length).std().shift(1)
#df_merged['Crack_MA'] = df_merged['Crack_Mid'].rolling(MA_length).mean().shift(1)
#df_merged['Crack_Upper'] = df_merged['Crack_MA'] + width* df_merged['Crack_STD']
#df_merged['Crack_Lower'] = df_merged['Crack_MA'] - width* df_merged['Crack_STD']
#df_merged['Crack_Ret'] = df_merged['Crack_Mid'].pct_change().shift(-1)

# Zero out the returns on the roll days
df_merged.loc[df_merged['month_symbol_GO'] !=  df_merged['month_symbol_GO'].shift(-1) ,'Crack_Ret'] =0

# Calculate the ratios
df_merged['ratio'] = (df_merged['Mid_GO'] /7.45) / df_merged['Mid_CO']
df_merged['ratio_MA'] = df_merged['ratio'].rolling(MA_length).mean().shift(1)
df_merged['ratio_STD'] =  df_merged['ratio'].rolling(STD_length).std().shift(1)
df_merged['ratio_upper'] = df_merged['ratio_MA']  + width * df_merged['ratio_STD']
df_merged['ratio_lower'] =  df_merged['ratio_MA']  - width* df_merged['ratio_STD']


#df_merged['ratio_MA_5'] = df_merged['ratio'].rolling(5).mean().shift(1)
#df_merged['ratio_MA_10'] = df_merged['ratio'].rolling(10).mean().shift(1)
#df_merged['ratio_MA_15'] = df_merged['ratio'].rolling(15).mean().shift(1)
#df_merged['ratio_MA_20'] = df_merged['ratio'].rolling(20).mean().shift(1)




# Build the signal
df_merged['Signal'] =0
df_merged.loc[df_merged['ratio'] > df_merged['ratio_upper'],'Signal'] =-1
df_merged.loc[df_merged['ratio'] < df_merged['ratio_lower'],'Signal'] =1

#calculate the return
df_merged['Return'] = df_merged['Return_GO'] - df_merged['Return_CO']
df_merged['Return'] = df_merged['Return'] * df_merged['Signal']
df_merged['CUM_Return'] = df_merged['Return'].cumsum()

import matplotlib.pyplot as plt
plt.plot(df_merged['Return'],color='red')
plt.plot(df_merged['CUM_Return'],color='blue')
plt.show()



#Draw the basic data
plt.plot(df_merged['Crack_Mid'], color='red')
plt.plot(df_merged['Crack_MA'], color='blue')
plt.show()

#Draw the ratio data
plt.plot(df_merged['ratio'], color='red')
plt.plot(df_merged['ratio_MA'], color='blue')
plt.show()



