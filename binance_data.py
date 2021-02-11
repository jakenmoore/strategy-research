from binance.client import Client
from binance.websockets import BinanceSocketManager
from datetime import datetime
import time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

load_dotenv(find_dotenv())

PUBLIC = os.getenv("BINANCE_API_KEY")
SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(api_key=PUBLIC, api_secret=SECRET)
bm = BinanceSocketManager(client)

data = []
max_len = 1000

def store_and_clear(lst):
    """
    When max_len of the tick data list is reached, creates csv and/or parquet files.
    """

    print(colored('SAVING DATA TO DISK', 'red'))
    csv_file = '/Volumes/GoogleDrive/Shared drives/Data/crypto/binance_trades_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv'
    df = pd.DataFrame(np.array(lst))
    df.columns = ["now", "timestamp", "buysell", "instrument", "price", "qty"]
    df = df.set_index("timestamp")
    # df.to_csv(csv_file)
    parquet_file = '/Volumes/GoogleDrive/Shared drives/Data/crypto/binance_trades_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.parquet'
    df.to_parquet(parquet_file)
    lst.clear()

def handle_message(msg):
    '''
    Generates trades tick stream from Binance and stores in memory as a list.
    '''

    if msg["e"] == "error":
        print(msg["m"])

    else:
        coins_exchanged = float(msg["p"]) * float(msg["q"])
        timestamp = msg["T"] / 1000
        timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S:%f")

        # Add in timestamp for now
        now = datetime.now()

        # Buy or sell?
        if msg["m"] == True:
            event_side = int(-1)
        else:
            event_side = int(1)

        # Print the data
        print(
            "{} - {} - {} - {} - Price: {} - Qty: {} {} Qty: {}".format(
                now, timestamp, event_side, msg["s"], msg["p"], msg["s"][:3], msg["q"], coins_exchanged
            )
        )
        if len(data) >= max_len:
            store_and_clear(data)
        else:
            data.append([now, timestamp, event_side, msg["s"], msg["p"], msg["q"]])

print(colored(("Collecting trades data from Binance"), "blue"))


# connections for each instrument
conn_key = bm.start_trade_socket("BTCUSDT", handle_message)
conn_key_eth = bm.start_trade_socket("ETHUSDT", handle_message)
conn_key_usdt = bm.start_trade_socket("USDTUSD", handle_message)

# end with keyboard interrupt
try:
    bm.start()
except KeyboardInterrupt:
    bm.stop_socket(conn_key)
    bm.stop_socket(conn_key_eth)
    bm.stop_socket(conn_key_usdt)
