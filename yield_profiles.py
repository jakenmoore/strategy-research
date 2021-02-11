windows = [10,100,500]
for window in windows:
        result["yield_offer_{}".format(window)] = np.where(result.side == "OFFER", (result.mid.shift(window) / result.trade_price) - 1,0)
        result["yield_bid_{}".format(window)] = np.where(result.side == "BID", (result.trade_price.shift(window) / result.mid) - 1,0)
        result["yield_{}".format(window)] = result["yield_offer_{}".format(window)] + result["yield_bid_{}".format(window)]
        cols = ["yield_offer_{}".format(window),"yield_bid_{}".format(window),"yield_{}".format(window)]
        result[cols] = result[cols].replace({0:np.nan})

yield_60_seconds = result.yield_bid_10.mean()


def tester_df(df):
    "{}_frame".format(df)["numbers"] = df
