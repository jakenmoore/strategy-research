
import pandas as pd

# list the instruments we want to analyse
instrument_list = ["EURUSD",
                   "USDJPY",
                   "GBPUSD",
                   "USDCAD",
                   "AUDUSD",
                   "USDCHF",
                   "NZDUSD",
                    "EURJPY",
                   "EURGBP",
                   "EURCHF",
                   "XAUUSD"]

def get_df(instrument_list, df_all):
    dict = {}
    for x in instrument_list:
        df_instrument = df_all.loc[df_all["instrument"] == x]
        dict[x] = df_instrument
    return dict

def calculate_mm_stats(df):
    df["spread_pct"] = df["usdPnl-PT0S"] / df["baseQuantityUsd"]
    df["mm_yield"] = (df["usdPnl-PT10S"] + df["usdPnl-PT30S"] + df["usdPnl-PT2M"] + df["usdPnl-PT4M"]) / 4
    df["mm_yield"] = df["mm_yield"] / df["baseQuantityUsd"]
    df["mm_pnl"] = df["mm_yield"] + 0.00001

def calculate_mm_stats_usd(df):
    df["spread_pct"] = df["usdPnl-PT0S"]
    df["mm_yield"] = (df["usdPnl-PT10S"] + df["usdPnl-PT30S"] + df["usdPnl-PT2M"] + df["usdPnl-PT4M"]) / 4
    df["mm_pnl"] = df["mm_yield"] + df["usdPnl-PT0S"]

def data_wrangle(df):
    df_df = pd.read_csv(df_file, index_col='t', parse_dates=['t'])
    df = df_df[["Bid0", "Offer0"]]
    del df_df
    df = df.rename(columns={"Bid0": "df_bid0", "Offer0": "df_offer0"})
    df["df_mid"] = (df["df_bid0"] + df["df_offer0"]) / 2
    df["df_mid_change"] = df["df_mid"].pct_change()
    df = df.resample(resample_period).last().ffill()

data_wrangle()