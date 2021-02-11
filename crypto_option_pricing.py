import json
import os
import pprint
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.stats as si
from dotenv import find_dotenv, load_dotenv
import ccxt
import sympy as sy
from sympy.stats import Normal, cdf


def black_scholes_currency(s, k, r, rf, sigma, t, option="call", exact=False):
    if option not in ("call", "put"):
        raise ValueError('option parameter must be one of "call" (default) or "put".')

    if exact:
        d1 = (sympy.ln(s / k) + t * (r - rf + sigma ** 2 / 2)) / (sigma * sympy.sqrt(t))
        d2 = d1 - sigma * sympy.sqrt(t)

        n = Normal("x", 0.0, 1.0)

        if option == "call":
            price = s * sympy.exp(-rf * t) * cdf(n)(d1) - k * sympy.exp(-r * t) * cdf(
                n
            )(d2)
        elif option == "put":
            price = k * sympy.exp(-r * t) * cdf(n)(-d2) - s * sympy.exp(-rf * t) * cdf(
                n
            )(-d1)

    else:
        d1 = (np.log(s / k) + t * (r - rf + sigma ** 2 / 2)) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        if option == "call":
            price = s * np.exp(-rf * t) * si.norm.cdf(d1) - k * np.exp(
                -r * t
            ) * stats.norm.cdf(d2)
        elif option == "put":
            price = k * np.exp(-r * t) * si.norm.cdf(-d2) - s * np.exp(
                -rf * t
            ) * stats.norm.cdf(-d1)

    return price


def generalized_black_scholes(s, k, sigma, t, r=0, b=0, option="call", exact=False):
    """
    s: spot price
    k: strike price
    t: time to maturity
    r: interest / funding rate
    sigma: volatility of underlying asset
    b = cost of funding
        If currency option, b = r - rf
        If option on future b = 0
    """
    if option not in ("call", "put"):
        raise ValueError('option parameter must be one of "call" (default) or "put".')

    if exact:
        d1 = (sympy.ln(s / k) + t * (b + sigma ** 2 / 2)) / (sigma * sympy.sqrt(t))
        d2 = d1 - sigma * sympy.sqrt(t)

        n = Normal("x", 0.0, 1.0)

        if option == "call":
            price = s * sympy.exp((b - r) * t) * cdf(n)(d1) - k * sympy.exp(
                -r * t
            ) * cdf(n)(d2)
        elif option == "put":
            price = k * sympy.exp(-r * t) * cdf(n)(-d2) - s * sympy.exp(
                (b - r) * t
            ) * cdf(n)(-d1)

    else:
        d1 = (np.log(s / k) + t * (b + sigma ** 2 / 2)) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        if option == "call":
            price = s * np.exp((b - r) * t) * si.norm.cdf(d1) - k * np.exp(
                -r * t
            ) * si.norm.cdf(d2)
        elif option == "put":
            price = k * np.exp(-r * t) * si.norm.cdf(-d2) - s * np.exp(
                (b - r) * t
            ) * si.norm.cdf(-d1)

    return np.round(price, 2)


def generalized_black_scholes_pct(price, s):
    pct_price = np.round(price / s, 4)
    return pct_price


def delta(s, k, t, r, sigma, option="call"):

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    if option == "call":
        result = si.norm.cdf(d1, 0.0, 1.0)
    if option == "put":
        result = -si.norm.cdf(-d1, 0.0, 1.0)

    return np.round(result, 3)


def theta(s, k, t, r, sigma, option="call"):

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)

    if option == "call":
        theta = (-sigma * s * prob_density) / (2 * np.sqrt(t)) - r * k * np.exp(
            -r * t
        ) * si.norm.cdf(d2, 0.0, 1.0)
    if option == "put":
        theta = (-sigma * s * prob_density) / (2 * np.sqrt(t)) + r * k * np.exp(
            -r * t
        ) * si.norm.cdf(-d2, 0.0, 1.0)

    return theta


def gamma(s, k, t, r, sigma):

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)

    gamma = prob_density / (s * sigma * np.sqrt(t))

    return gamma


def vega(s, s0, k, t, r, sigma):

    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)

    vega = s0 * prob_density * np.sqrt(t)

    return vega


def rho(s, k, t, r, sigma, option="call"):

    d2 = (np.log(s / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    if option == "call":
        rho = t * k * np.exp(-r * t) * si.norm.cdf(d2, 0.0, 1.0)
    if option == "put":
        rho = -t * k * np.exp(-r * t) * si.norm.cdf(-d2, 0.0, 1.0)

    return rho


"""
Connect to Darebit
"""

# Unused for now, but kept in for posterity
api_key = os.getenv("DERIBIT_API_KEY")
api_secret = os.getenv("DERIBIT_API_SECRET")
api_passphrase = os.getenv("DERIBIT_API_PASSPHRASE")

load_dotenv(find_dotenv())


def get_deribit_option_data(instrument_name):
    exchange_class = getattr(ccxt, "deribit")
    exchange_class_params = {
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_passphrase,
        "timeout": 30000,
        "enableRateLimit": True,
    }
    exchange = exchange_class(exchange_class_params)
    deribit_option_data = exchange.publicGetGetOrderBook(
        {"instrument_name": instrument_name}
    )
    return deribit_option_data


def get_deribit_strike_price(instrument, currency):
    exchange_class = getattr(ccxt, "deribit")
    exchange_class_params = {
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_passphrase,
        "timeout": 30000,
        "enableRateLimit": True,
    }
    exchange = exchange_class(exchange_class_params)
    response = exchange.publicGetGetInstruments(
        {"currency": currency, "kind": "option"}
    )
    instruments = response["result"]
    for instr in instruments:
        if instr["instrument_name"] == instrument:
            strike_price = instr["strike"]
            return float(strike_price)
    return False


def get_deribit_expiry_date(instrument, currency):
    exchange_class = getattr(ccxt, "deribit")
    exchange_class_params = {
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_passphrase,
        "timeout": 30000,
        "enableRateLimit": True,
    }
    exchange = exchange_class(exchange_class_params)
    response = exchange.publicGetGetInstruments(
        {"currency": currency, "kind": "option"}
    )
    instruments = response["result"]
    for instr in instruments:
        if instr["instrument_name"] == instrument:
            expiry = instr["expiration_timestamp"] / 1000
            expiry = datetime.fromtimestamp(expiry)
            return expiry
    return False

def get_deribit_option_type(instrument, currency):
    exchange_class = getattr(ccxt, "deribit")
    exchange_class_params = {
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_passphrase,
        "timeout": 30000,
        "enableRateLimit": True,
    }
    exchange = exchange_class(exchange_class_params)
    response = exchange.publicGetGetInstruments(
        {"currency": currency, "kind": "option"}
    )
    instruments = response["result"]
    for instr in instruments:
        if instr["instrument_name"] == instrument:
            option_type = instr["option_type"]
            return option_type
    return False


def get_days_between(date_past, date_future):
    difference = date_future - date_past
    return difference.total_seconds() / timedelta(days=1).total_seconds()


def get_days_to_expiry(deribit_expiry):
    days_to_expiry = get_days_between(datetime.now(), deribit_expiry)
    return days_to_expiry


def get_deribit_bid_iv(option):
    deribit_bid_iv = float(option["bid_iv"]) / 100
    return deribit_bid_iv


def get_deribit_ask_iv(option):
    deribit_ask_iv = float(option["ask_iv"]) / 100
    return deribit_ask_iv


def get_deribit_mid_iv(deribit_bid_iv, deribit_ask_iv):
    deribit_mid_iv = (deribit_bid_iv + deribit_ask_iv) / 2
    return deribit_mid_iv


def get_deribit_index_price(option):
    deribit_index_price = float(option["index_price"])
    return deribit_index_price


def get_deribit_underlying_price(option):
    deribit_underlying_price = float(option["underlying_price"])
    return deribit_underlying_price


def get_deribit_best_bid_price(option):
    deribit_best_bid_price = float(option["best_bid_price"])
    return deribit_best_bid_price


def get_deribit_best_ask_price(option):
    deribit_best_ask_price = float(option["best_ask_price"])
    return deribit_best_ask_price


def get_deribit_delta(option):
    deribit_delta = float(option["greeks"]["delta"])
    return np.round(deribit_delta, 3)
