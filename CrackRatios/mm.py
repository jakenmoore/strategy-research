"""
The simple inventory algorithm.
"""
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd

from strategy.constant import ADD, AMEND, CANCEL, LIMIT, MARKET
from strategy.log import get_logger

from .algorithm import Algorithm


class SimpleInventory(Algorithm):
    """
    The simple inventory algorithm.
    """

    def __init__(self, max_size: int, sigma_freq: float, gamma: float, n: float) -> None:
        super().__init__()
        self.max_size = max_size
        self.sigma_freq = sigma_freq
        self.gamma = gamma
        self.n = n
        self._log = get_logger("Algorithm::" + self.__class__.__name__)

    def execute(
        self,
        signals: pd.DataFrame,
        position: dict,
        balance: dict,
        nav: float,
        orders: List[dict],
        L1: dict,
        tick: dict,
    ) -> List[dict]:
        actions = []
        signal = signals.iloc[-1].to_dict()
        sigma, tfi = signal["sigma"], signal["tfi"]
        add_bid, add_ask = True, True
        bid_price, ask_price = L1["bid_price"], L1["ask_price"]
        mid_price = (bid_price + ask_price) / 2
        live_position = (position["amount"] / nav)

        # Calculate USD sigma
        # TODO: Review sigma freq
        sigma_freq = sigma * np.sqrt(self.sigma_freq)
        sigma_usd = sigma_freq * mid_price

        # TODO: Figure out scaling for live_position
        # TODO: Figure out inv_adj_usd
        inv_adj = sigma_usd  self.gamma  live_position
        inv_adj_usd = inv_adj * mid_price

        # sensitivity to inventory - could have this set as -gamma? also needs to be scaled to size of lot
        # TODO: Check the calculation, esp. the * and / that were removed from skype
        bid_pct = np.abs(np.round(np.where(live_position < 0, self.max_size, self.max_size / np.exp(-self.n * live_position)), 0))
        ask_pct = -1.0  np.abs(np.round(np.where(live_position > 0, self.max_size, self.max_size  np.exp(-self.n * live_position)), 0))
        self._log.info(f"LIVE_POSITION: {live_position:.2f}, SIGMA_USD: {sigma_usd:.6f}, MAX_SIZE: {self.max_size}, N: {self.n:.6f}")
        self._log.info(f"BID_PCT: {bid_pct:.2%}, ASK_PCT: {ask_pct:.2%}")

        # TODO: Add support for bid or ask size of 0, do not add order, cancel existing order!
        # Calculate size in terms of dollars, minimum unit of $1 for bitmex
        bid_size = np.floor(bid_pct * nav)
        ask_size = np.floor(ask_pct * nav)

        # Amend existing orders
        for order in orders:
            # Don't add new order if existing order can be amended
            if add_bid and np.sign(order["amount"]) == 1:
                actions.append(
                    {
                        "order_id": order["order_id"],
                        "operation": AMEND,
                        "amount": bid_size,
                        "price": bid_price,
                    }
                )
                add_bid = False
                continue
            elif add_ask and np.sign(order["amount"]) == -1:
                actions.append(
                    {
                        "order_id": order["order_id"],
                        "operation": AMEND,
                        "amount": ask_size,
                        "price": ask_price,
                    }
                )
                add_ask = False
                continue
            actions.append({"operation": CANCEL, "order_id": order["order_id"]})

        # Create order based on inventory target
        # TODO: Add support to calculate bid, ask price based on inventory rather than L1
        # TODO: Add support to round price to $0.5 increments and not cross the book
        if add_bid:
            actions.append(
                {
                    "operation": ADD,
                    "cl_order_id": str(uuid4()),
                    "type": LIMIT,
                    "amount": bid_size,
                    "price": bid_price,
                }
            )
        if add_ask:
            actions.append(
                {
                    "operation": ADD,
                    "cl_order_id": str(uuid4()),
                    "type": LIMIT,
                    "amount": ask_size,
                    "price": ask_price,
                }
            )
        return actions