import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq


df_trades = pd.read_csv(
    "C:\\TickData\\TickWrite7\\DATA\\fivedays\\trades_CO10-14.09.2018.csv",
    dtype={
        "Date": str,
        "Time": str,
        "Price": np.float64,
        "Volume": np.float64,
        "Market Flag": str,
    },
)
# Set the index
df_trades["timestamp"] = df_trades["Date"] + " " + df_trades["Time"]
df_trades["timestamp"] = pd.to_datetime(
    df_trades["timestamp"], format="%m/%d/%Y %H:%M:%S.%f"
)
df_trades.set_index("timestamp", inplace=True)
df_trades.drop(["Date", "Time"], axis=1, inplace=True)
df_trades.sort_index(inplace=True)
df_trades.drop("Market Flag", inplace=True, axis=1)
df_trades.columns = ["price", "volume"]

# Save final result as compressed parquet
df_trades.to_parquet("C:\\TickData\\TickWrite7\\DATA\\fivedays\\trades_CO10-14.09.2018.parq", compression="snappy")


#############################
df_quotes = pd.read_csv(
    "C:\\TickData\\TickWrite7\\DATA\\fivedays\\quotes_CO10-14.09.2018.csv",
    dtype={
        "Date": str,
        "Time": str,
        "Price": np.float64,
        "Volume": np.float64,
        "Market Flag": str,
    },
)
# Set the index
df_quotes["timestamp"] = df_quotes["Date"] + " " + df_quotes["Time"]
df_quotes["timestamp"] = pd.to_datetime(
    df_quotes["timestamp"], format="%m/%d/%Y %H:%M:%S.%f"
)
df_quotes.set_index("timestamp", inplace=True)
df_quotes.drop(["Date", "Time"], axis=1, inplace=True)
df_quotes.sort_index(inplace=True)
df_quotes.drop("Market Flag", inplace=True, axis=1)
df_quotes.drop("Quote Condition", inplace=True, axis=1)
df_quotes.columns = ["bid_price", "bid_size", "ask_price", "ask_size"]
# Save final result as compressed parquet
df_quotes.to_parquet("C:\\TickData\\TickWrite7\\DATA\\fivedays\\quotes_CO10-14.09.2018.parq", compression="snappy")
