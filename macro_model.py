import numpy as np
import pandas as pd

# config risk target
annual_risk_target = 0.1  # 10%
daily_risk_target = annual_risk_target / np.sqrt(252)
nav = 5000000

# config lists
usd_yields = ['usgg2yr_index', 'usgg10yr_index']
usd_prices = ['tu1_comdty', 'ty1_comdty', 'spx_index']
eur_yields = ['gecu2yr_index', 'gecu10yr_index']
eur_prices = ['du1_comdty', 'rx1_comdty', 'dax_index']
jpy_yields = ["gjgb2_index", "gjgb10_index"]
jpy_prices = ["jb1_comdty", "nky_index"]
gbp_yields = ["gtgbp2y_govt", "gtgbp10y_govt"]
gbp_prices = ["wb1_comdty", "g_1_comdty", "ukx_index"]
aud_yields = ["gacgb2_index", "gacgb10_index"]
aud_prices = ["ym1_comdty", "xm1_comdty", "as51_index"]
cad_yields = ["gcan2yr_index", "gcan10yr_index"]
cad_prices = ["cv1_comdty", "cn1_comdty", "sptsx_index"]
com_prices = ["cl1_comdty", "xau_curncy", "hg1_comdty"]


all_yields = usd_yields + eur_yields + jpy_yields + gbp_yields + aud_yields + cad_yields
all_prices = usd_prices + eur_prices + jpy_prices + gbp_prices + aud_prices + cad_prices + com_prices


rates_prices_list = ["ty1_comdty", "rx1_comdty", "jb1_comdty", "g_1_comdty", "cn1_comdty", "xm1_comdty", ]
risk_prices_list = ["spx_index", "dax_index", "nky_index", "ukx_index", "as51_index", "sptsx_index", "xau_curncy",
                    "hg1_comdty"]

all_prices_list = ['ty1_comdty_weight',
                   'spx_index_weight',
                   'rx1_comdty_weight',
                   'dax_index_weight',
                   'jb1_comdty_weight',
                   'nky_index_weight',
                   'wb1_comdty_weight',
                   'g_1_comdty_weight',
                   'ukx_index_weight',
                   'xm1_comdty_weight',
                   'as51_index_weight',
                   'cn1_comdty_weight',
                   'sptsx_index_weight',
                   'xau_curncy_weight',
                   'hg1_comdty_weight']


def load_data(data_start_date, end_date, daily_risk_target):
    # read bloomberg data
    usd = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/USD.xlsx").loc[data_start_date:end_date]
    eur = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/EUR.xlsx").loc[data_start_date:end_date]
    jpy = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/JPY.xlsx").loc[data_start_date:end_date]
    gbp = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/GBP.xlsx").loc[data_start_date:end_date]
    aud = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/AUD.xlsx").loc[data_start_date:end_date]
    cad = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/CAD.xlsx").loc[data_start_date:end_date]
    com = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/commodities.xlsx").loc[data_start_date:end_date]
    frames = [usd, eur, jpy, gbp, aud, cad, com]
    all_data = pd.concat(frames, axis=1)
    add_data_columns(all_data, all_yields, all_prices, daily_risk_target)
    return all_data


def read_bloomberg(file_name):
    """ reads an excel spreadsheet with data from bloomberg"""
    data = pd.read_excel(file_name,
                         header=0, skiprows=5, index_col='Dates', parse_dates=True)
    data.columns = pd.read_excel(file_name, skiprows=2).iloc[0, 1:]
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.lower()
    data = data.apply(pd.to_numeric, errors='ignore')
    return data


def create_indices(all_data):
    # create indices and set up data for models
    all_data["weighted_10y_rates_index_return"] = (all_data[["usgg10yr_index_weighted_return",
                                                             "gecu10yr_index_weighted_return",
                                                             "gjgb10_index_weighted_return",
                                                             "gtgbp10y_govt_weighted_return",
                                                             "gacgb10_index_weighted_return",
                                                             "gcan10yr_index_weighted_return", ]]).sum(axis=1)

    all_data["weighted_risk_index_return"] = (all_data[["spx_index_weighted_return",
                                                        "dax_index_weighted_return",
                                                        "nky_index_weighted_return",
                                                        "ukx_index_weighted_return",
                                                        "as51_index_weighted_return",
                                                        "sptsx_index_weighted_return",
                                                        "xau_curncy_weighted_return",
                                                        "hg1_comdty_weighted_return"]]).sum(axis=1)

    # turn rates index the other way round
    all_data["weighted_10y_rates_index_return"] = all_data["weighted_10y_rates_index_return"] * (-1)

    # average of both indices
    all_data["weighted_index_return"] = all_data[["weighted_risk_index_return",
                                                  "weighted_10y_rates_index_return"]].sum(axis=1)

    all_data["weighted_index_return_20d_vol"] = all_data["weighted_index_return"].rolling(20).std() * np.sqrt(252)
    all_data["scalar"] = annual_risk_target / all_data["weighted_index_return_20d_vol"]
    # cumulative returns
    all_data["weighted_risk_cum_sum"] = all_data["weighted_risk_index_return"].cumsum()
    all_data["weighted_rates_cum_sum"] = all_data["weighted_risk_index_return"].cumsum()
    return


def value_at_risk(df):
    """This calculates value at risk for a series"""
    return


def return_pct(df, is_yield=False):
    """calculates the percentage change for a price or yield series. Default is price"""
    if not is_yield:
        return np.log(df).diff()
    else:
        return df.diff()


# TODO scalar for every position = weighted return actual / target
# TODO make a max weight - so that when vol falls position never gets too big
# TODO weight - make the changes in size discrete, every few percent
# TODO trading logic - if vol > percentile and price < ma, cut risk to zero.
# TODO short system
# TODO drawdown - if drawdown > x, cut position

def add_data_columns(df, yields, prices, daily_risk_target):
    """Adds in a number of computed columns for a given DF of data with lists of column names prices and yields. /
    Adds a risk target, set to 10% annualised as default"""
    data = yields + prices
    for column_name in yields:
        df[column_name + "_change"] = return_pct(df[column_name], True)
    for column_name in prices:
        df[column_name + "_change"] = return_pct(df[column_name], False)
    for column_name in data:
        df[column_name + "_120d_vol"] = df[column_name + "_change"].rolling(120).std() * np.sqrt(252)
        df[column_name + "_21d_vol"] = df[column_name + "_change"].rolling(21).std() * np.sqrt(252)
        df[column_name + "_5d_vol"] = df[column_name + "_change"].rolling(5).std() * np.sqrt(252)
        df[column_name + "_21d_ma"] = df[column_name + "_change"].rolling(21).mean()
        df[column_name + "_weight"] = daily_risk_target / df[column_name + "_21d_vol"].shift(1)
        df[column_name + "_weighted_return"] = df[column_name + "_weight"] * df[column_name + "_change"]
        df[column_name + "_60d_percentile"] = df[column_name + "_change"].rolling(120).mean().rank(pct=True)
        df[column_name + "_60d_percentile_detrended"] = (
                df[column_name + "_change"] - df[column_name + "_21d_ma"]).rolling(120).mean().rank(pct=True)


def add_vol(df):
    for column_name in df:
        df[column_name + "_120d_vol"] = df[column_name + "_change"].rolling(120).std() * np.sqrt(252)


def risk_scalar(df):
    """(.0031/@movstdev(risk(-1),120))*6 not sure why *6"""
    return


def annual_return(returns):
    annual_return = returns.mean() * 252
    return annual_return


def annual_risk(returns):
    annual_risk = returns.std() * np.sqrt(252)
    return annual_risk


def sharpe_ratio(returns):
    """returns Sharpe of annualised risk and return"""
    sharpe_ratio = annual_return(returns) / annual_risk(returns)
    return sharpe_ratio


# list_of_files = [usd, eur, jpy, gbp, aud, cad, commodities]


# TODO Thy doesnt this work?
# def load_all_data(list_of_files):
#     for filename in list_of_files:
#         filename = read_bloomberg("/Volumes/GoogleDrive/Shared drives/data/macro/" + filename + ".xlsx").loc[
#                    data_start_date:end_date]
#     return filename
# load_all_data(list_of_files)


def run_model(all_data):
    # relative performance of risk and return
    all_data["risk_30d_ma"] = all_data["weighted_risk_index_return"].rolling(21).mean()
    all_data["rates_30d_ma"] = all_data["weighted_10y_rates_index_return"].rolling(21).mean()
    all_data["relative_performance_30d_ma"] = (all_data["risk_30d_ma"] - all_data["rates_30d_ma"])
    # create relative performance indicator
    all_data["relative_performance"] = (
            all_data["weighted_risk_index_return"] - all_data["weighted_10y_rates_index_return"])
    # set up signals on the weighted indices - don't shift yet
    all_data["rp_signal"] = np.where(all_data["relative_performance_30d_ma"].diff() >= 0, 1, -1)
    all_data["weighted_risk_trend_signal"] = np.where(
        (all_data["weighted_risk_cum_sum"] - all_data["weighted_risk_cum_sum"].rolling(21).mean()) >= 0, 1, -1)
    all_data["weighted_rates_trend_signal"] = np.where(
        (all_data["weighted_rates_cum_sum"] - all_data["weighted_rates_cum_sum"].rolling(21).mean()) >= 0, 1, -1)
    # create risk signal
    all_data["risk_signal"] = 0
    all_data.loc[(all_data["rp_signal"] == 1), "risk_signal"] = 1
    all_data.loc[(all_data["rp_signal"] == -1) & (all_data["weighted_risk_trend_signal"] == -1), "risk_signal"] = -1
    all_data.loc[(all_data["rp_signal"] == -1) & (all_data["weighted_risk_trend_signal"] == 1), "risk_signal"] = 0
    # shift the signal
    all_data["risk_signal"] = all_data["risk_signal"].shift(1)
    # create rates signal
    all_data["rates_signal"] = 0
    all_data.loc[(all_data["rp_signal"] == -1), "rates_signal"] = 1
    all_data.loc[(all_data["rp_signal"] == 1) & (all_data["weighted_rates_trend_signal"] == -1), "rates_signal"] = -1
    all_data.loc[(all_data["rp_signal"] == 1) & (all_data["weighted_rates_trend_signal"] == 1), "risk_signal"] = 1

    # shift the signal
    all_data["rates_signal"] = all_data["rates_signal"].shift(1)
    # create the risk and rates moodel pnls
    all_data["risk_pnl"] = all_data["risk_signal"] * all_data["weighted_risk_index_return"]
    all_data["rates_pnl"] = all_data["rates_signal"] * all_data["weighted_10y_rates_index_return"]
    all_data["model_pnl"] = (all_data["risk_pnl"] + all_data["rates_pnl"])
    all_data["model_pnl_sum"] = all_data["model_pnl"].cumsum()
    return


def run_valuation_models(all_data):
    # create valuation indicator
    all_data['risk_pct_rank'] = all_data['weighted_risk_index_return'].rolling(120).mean().rank(pct=True)
    all_data['rates_pct_rank'] = all_data['weighted_10y_rates_index_return'].rolling(120).mean().rank(pct=True)
    all_data["rich_cheap"] = (all_data["spx_index_60d_percentile"] +
                              all_data["spx_index_60d_percentile_detrended"]) / 2
    # create trading signal based on value
    all_data['value_signal'] = 0
    all_data.loc[all_data["rich_cheap"] >= .8, 'value_signal'] = -1
    all_data.loc[(all_data["rich_cheap"] <= .2), 'value_signal'] = 1
    # shift the signal
    all_data["value_signal"] = all_data["value_signal"].shift(1)
    # TODO create vol signal
    # TODO create momentum signal
    return


def show_performance_stats(all_data):
    ar = annual_return(all_data["model_pnl"])
    risk = annual_risk(all_data["model_pnl"])
    sr = sharpe_ratio(all_data["model_pnl"])
    print(f"Annual return: {ar:.1%}")
    print(f"Annual risk: {risk:.1%}")
    print(f"Sharpe Ratio: {sr:.1f}")


weights_futures_list = ['ty1_comdty_weight_signal',
                        'spx_index_weight_signal',
                        'rx1_comdty_weight_signal',
                        'dax_index_weight_signal',
                        'jb1_comdty_weight_signal',
                        'nky_index_weight_signal',
                        'wb1_comdty_weight_signal',
                        'g_1_comdty_weight_signal',
                        'ukx_index_weight_signal',
                        'xm1_comdty_weight_signal',
                        'as51_index_weight_signal',
                        'cn1_comdty_weight_signal',
                        'sptsx_index_weight_signal',
                        'xau_curncy_weight_signal',
                        'hg1_comdty_weight_signal']

weights_pct_list = ['ty1_comdty_weight_signal_pct',
                    'spx_index_weight_signal_pct',
                    'rx1_comdty_weight_signal_pct',
                    'dax_index_weight_signal_pct',
                    'jb1_comdty_weight_signal_pct',
                    'nky_index_weight_signal_pct',
                    'wb1_comdty_weight_signal_pct',
                    'g_1_comdty_weight_signal_pct',
                    'ukx_index_weight_signal_pct',
                    'xm1_comdty_weight_signal_pct',
                    'as51_index_weight_signal_pct',
                    'cn1_comdty_weight_signal_pct',
                    'sptsx_index_weight_signal_pct',
                    'xau_curncy_weight_signal_pct',
                    'hg1_comdty_weight_signal_pct']

returns_list = ['ty1_comdty_change',
                'spx_index_change',
                'rx1_comdty_change',
                'dax_index_change',
                'jb1_comdty_change',
                'nky_index_change',
                'wb1_comdty_change',
                'g_1_comdty_change',
                'ukx_index_change',
                'xm1_comdty_change',
                'as51_index_change',
                'cn1_comdty_change',
                'sptsx_index_change',
                'xau_curncy_change',
                'hg1_comdty_change']


def output_weights_futures(all_data):
    all_data['ty1_comdty_weight_signal'] = all_data["ty1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["rx1_comdty_weight_signal"] = all_data["rx1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["jb1_comdty_weight_signal"] = all_data["jb1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["wb1_comdty_weight_signal"] = all_data["wb1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["g_1_comdty_weight_signal"] = all_data["g_1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["xm1_comdty_weight_signal"] = all_data["xm1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data["cn1_comdty_weight_signal"] = all_data["cn1_comdty_weight"] * all_data["rates_signal"] * nav / 1000000
    all_data['spx_index_weight_signal'] = all_data["spx_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["dax_index_weight_signal"] = all_data["dax_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["nky_index_weight_signal"] = all_data["nky_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["ukx_index_weight_signal"] = all_data["ukx_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["as51_index_weight_signal"] = all_data["as51_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["sptsx_index_weight_signal"] = all_data["sptsx_index_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["xau_curncy_weight_signal"] = all_data["xau_curncy_weight"] * all_data["risk_signal"] * nav / 1000000
    all_data["hg1_comdty_weight_signal"] = all_data["hg1_comdty_weight"] * all_data["risk_signal"] * nav / 1000000
    return


def output_weights_pct(all_data):
    all_data['ty1_comdty_weight_signal_pct'] = all_data["ty1_comdty_weight"] * all_data["rates_signal"]
    all_data["rx1_comdty_weight_signal_pct"] = all_data["rx1_comdty_weight"] * all_data["rates_signal"]
    all_data["jb1_comdty_weight_signal_pct"] = all_data["jb1_comdty_weight"] * all_data["rates_signal"]
    all_data["wb1_comdty_weight_signal_pct"] = all_data["wb1_comdty_weight"] * all_data["rates_signal"]
    all_data["g_1_comdty_weight_signal_pct"] = all_data["g_1_comdty_weight"] * all_data["rates_signal"]
    all_data["xm1_comdty_weight_signal_pct"] = all_data["xm1_comdty_weight"] * all_data["rates_signal"]
    all_data["cn1_comdty_weight_signal_pct"] = all_data["cn1_comdty_weight"] * all_data["rates_signal"]
    all_data['spx_index_weight_signal_pct'] = all_data["spx_index_weight"] * all_data["risk_signal"]
    all_data["dax_index_weight_signal_pct"] = all_data["dax_index_weight"] * all_data["risk_signal"]
    all_data["nky_index_weight_signal_pct"] = all_data["nky_index_weight"] * all_data["risk_signal"]
    all_data["ukx_index_weight_signal_pct"] = all_data["ukx_index_weight"] * all_data["risk_signal"]
    all_data["as51_index_weight_signal_pct"] = all_data["as51_index_weight"] * all_data["risk_signal"]
    all_data["sptsx_index_weight_signal_pct"] = all_data["sptsx_index_weight"] * all_data["risk_signal"]
    all_data["xau_curncy_weight_signal_pct"] = all_data["xau_curncy_weight"] * all_data["risk_signal"]
    all_data["hg1_comdty_weight_signal_pct"] = all_data["hg1_comdty_weight"] * all_data["risk_signal"]
    return

def individual_returns(all_data):
    all_data["spx_model_return"] = all_data["spx_index_weight_signal_pct"] * all_data["spx_index_change"]
    all_data["dax_model_return"] = all_data["dax_index_weight_signal_pct"] * all_data["dax_index_change"]
    all_data["xau_model_return"] = all_data["xau_curncy_weight_signal_pct"] * all_data["xau_curncy_change"]
    all_data["hg1_model_return"] = all_data["hg1_comdty_weight_signal_pct"] * all_data["hg1_comdty_change"]
    all_data["ty1_model_return"] = all_data["ty1_comdty_weight_signal_pct"] * all_data["ty1_comdty_change"]
    all_data["g_1_model_return"] = all_data["g_1_comdty_weight_signal_pct"] * all_data["g_1_comdty_change"]
    all_data["rx1_model_return"] = all_data["rx1_comdty_weight_signal_pct"] * all_data["rx1_comdty_change"]
    return



def hist_value_at_risk(returns, weights, alpha, lookback):
    sim_returns = returns[-lookback:] * weights
    sim_returns_sum = sim_returns.sum(axis=1)
    return np.percentile(sim_returns_sum, 100 * (1 - alpha))


def conditional_value_at_risk(returns, weights, alpha, lookback):
    # Call out to our existing function
    hvar = hist_value_at_risk(returns, weights, alpha, lookback=lookback)
    returns = returns.fillna(0.0)
    sim_returns = returns[-lookback:] * weights
    sim_returns_sum = sim_returns.sum(axis=1)
    return np.nanmean(sim_returns_sum[sim_returns_sum < hvar])


def brexit_stress(returns, weights):
    brexit_rets = returns['2016-6-24':'2016-6-24']
    brexit_stress_sim = brexit_rets * weights
    brexit_stress_sim = int(brexit_stress_sim.sum(axis=1))
    return brexit_stress_sim



def show_risk_metrics(returns, pct_positions):
    HVaR_95_1y = hist_value_at_risk(returns, pct_positions, 0.95, 252)
    CVaR_95_1y = conditional_value_at_risk(returns, pct_positions, 0.95, 252)
    HVaR_99_2y = hist_value_at_risk(returns, pct_positions, 0.99, 504)
    CVaR_99_2y = conditional_value_at_risk(returns, pct_positions, 0.99, 504)
    notional_positions = np.sum(np.absolute(pct_positions)) * nav
    leverage = (np.sum(np.absolute(pct_positions)) * nav) / nav
    stress_brexit = brexit_stress(returns, pct_positions)
    print(f"HVaR 95% 1 year: {HVaR_95_1y:.2%}")
    print(f"CVaR 95% 1 year: {CVaR_95_1y:.2%}")
    print(f"HVaR 99% 2 year: {HVaR_99_2y:.2%}")
    print(f"CVaR 99% 2 year: {CVaR_99_2y:.2%}")
    print(f"Notional positions: {notional_positions:.1f}")
    print(f"Leverage: {leverage:.1%}")
    print(f"Brexit stress: {stress_brexit: .2%}")

