import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

#cur_dir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(cur_dir)
plt.style.use('ggplot')
np.random.seed(123)

"""
Parameters of the backtesting
"""

s0 = 100 # initial price
T = 1# time
sigma = 2 # volatility daily
dt = 0.005 # number of time steps - this creates 200 price points
q0 = 0 # time steps
gamma = 0.1 # risk aversion
k = 1.5
A = 140 # frequency of buy or sell orders
sim_length = 1000

class Market():
    def __init__(self):
        self.bid_price = np.nan()  
        self.bid_size = np.nan()
        self.offer_size = np.nan()
        self.offer_price = np.nan()
    def mid(self):
        return (self.bid_price + self.offer_price) /2

def calculate_value(s,inventory,gamma,sigma,T,step,dt):
    return s - inventory * gamma * sigma ** 2 * (T - step * dt)

def calculate_spread(gamma, sigma,T,step, dt,k,):
    _spread = gamma * sigma ** 2 * (T - step * dt) + (2 / gamma) * np.log(1 + (gamma / k))
    _spread /= 2
    return _spread

def calculate_Bid_offer(value, spread):
    return value - spread , value + spread

def generate_orders(value, spread,max_quantity,n,inventory):
    return_value = {}
    bid_size = np.round(np.where(inventory < 0, max_quantity, max_quantity / np.exp(-n * inventory)), 0)
    ask_size = np.round(np.where(inventory > 0, max_quantity, max_quantity * np.exp(-n * inventory)), 0)
    return_value.append({"Direction":"BUY", "Price": value - spread,"Quantity":np.min(bid_size,max_quantity)})
    return_value.append({"Direction": "SELL", "Price": value + spread,"Quantity":np.min(ask_size,max_quantity)})
    return return_value

if __name__ == '__main__':
    """
    Variables holding inventory and P&l
    """
    inventory_s1 = [q0] * sim_length
    pnl_s1 = [0] * sim_length

    inventory_s2 = [q0] * sim_length
    pnl_s2 = [0] * sim_length

    """
    Variables holding the price properties
    that will be used in the price plot
    """
    price_a = [0] * (int(T / dt) + 1)
    price_b = [0] * (int(T / dt) + 1)
    midprice = [0] * (int(T / dt) + 1)

    """
    Computation of spread for a symmetric strategy
    """
    sym_spread = 0
    for i in np.arange(0, T, dt):
        sym_spread += gamma * sigma**2 * (T - i) + \
                      (2/gamma) * np.log(1 + (gamma / k))

    av_sym_spread = (sym_spread / (T / dt))

    prob = A * np.exp(- k * av_sym_spread / 2) * dt

    """
    Simulation
    """
    for i in range(sim_length):

        white_noise = sigma * np.sqrt(dt) * np.random.choice([1, -1], int(T / dt))
        price_process = s0 + np.cumsum(white_noise)
        price_process = np.insert(price_process, 0, s0)

        for step, s in enumerate(price_process):

            """
            Inventory strategy
            """

            reservation_price = s - inventory_s1[i] * gamma * \
                                    sigma**2 * (T - step * dt)
            spread = gamma * sigma**2 * (T - step * dt) + \
                     (2 / gamma) * np.log(1 + (gamma / k))
            spread /= 2

            print(spread)
            if reservation_price >= s:
                ask_spread = spread + (reservation_price - s)
                bid_spread = spread - (reservation_price - s)
            else:
                ask_spread = spread - (s - reservation_price)
                bid_spread = spread + (s - reservation_price)

            ask_prob = A * np.exp(- k * ask_spread) * dt
            bid_prob = A * np.exp(- k * bid_spread) * dt
            ask_prob = max(0, min(ask_prob, 1))
            bid_prob = max(0, min(bid_prob, 1))
            ask_action_s1 = np.random.choice([1, 0],
                                             p=[ask_prob, 1 - ask_prob])
            bid_action_s1 = np.random.choice([1, 0],
                                             p=[bid_prob, 1 - bid_prob])

            inventory_s1[i] -= ask_action_s1
            pnl_s1[i] += ask_action_s1 * (s + ask_spread)
            inventory_s1[i] += bid_action_s1
            pnl_s1[i] -= bid_action_s1 * (s - bid_spread)

            if i == 0:
                price_a[step] = s + ask_spread
                price_b[step] = s - bid_spread
                midprice[step] = s

            """
            Symmetric strategy
            """

            ask_action_s2 = np.random.choice([1, 0], p=[prob, 1 - prob])
            bid_action_s2 = np.random.choice([1, 0], p=[prob, 1 - prob])
            inventory_s2[i] -= ask_action_s2
            pnl_s2[i] += ask_action_s2 * (s + av_sym_spread / 2)
            inventory_s2[i] += bid_action_s2
            pnl_s2[i] -= bid_action_s2 * (s - av_sym_spread / 2)

        pnl_s1[i] += inventory_s1[i] * s
        pnl_s2[i] += inventory_s2[i] * s

    x_range = [-50, 150]
    y_range = [0, 250]
    plt.figure(figsize=(16, 12), dpi=100)
    bins = np.arange(x_range[0], x_range[1] + 1, 4)
    plt.hist(pnl_s1, bins=bins, alpha=0.25,
             label="Inventory strategy")
    plt.hist(pnl_s2, bins=bins, alpha=0.25,
             label="Symmetric strategy")
    plt.ylabel('P&l')
    plt.legend()
    plt.axis(x_range + y_range)
    plt.title("The P&L histogram of the two strategies")
    plt.savefig('pnl.pdf', bbox_inches='tight', dpi=100,
                format='pdf')

    x = np.arange(0, T + dt, dt)
    plt.figure(figsize=(16, 12), dpi=100)
    plt.plot(x, price_a, linewidth=1.0, linestyle="-",
             label="ASK")
    plt.plot(x, price_b, linewidth=1.0, linestyle="-",
             label="BID")
    plt.plot(x, midprice, linewidth=1.0, linestyle="-",
             label="MID-PRICE")
    plt.legend()
    plt.title("The mid-price and the optimal bid and ask quotes")
    plt.savefig('prices.pdf', bbox_inches='tight', dpi=100,
                format='pdf')
    plt.show()

    print("P&L - Mean of the inventory strategy: "
          "{}".format(np.array(pnl_s1).mean()))
    print("P&L - Mean of the symmetric strategy: "
          "{}".format(np.array(pnl_s2).mean()))
    print("P&L - Standard deviation of the inventory strategy: "
          "{}".format(np.sqrt(np.array(pnl_s1).var())))
    print("P&L - Standard deviation of the symmetric strategy: "
          "{}".format(np.sqrt(np.array(pnl_s2).var())))
    print("INV - Mean of the inventory strategy: "
          "{}".format(np.array(inventory_s1).mean()))
    print("INV - Mean of the symmetric strategy: "
          "{}".format(np.array(inventory_s2).mean()))
    print("INV - Standard deviation of the inventory strategy: "
          "{}".format(np.sqrt(np.array(inventory_s1).var())))
    print("INV - Standard deviation of the symmetric strategy: "
          "{}".format(np.sqrt(np.array(inventory_s2).var())))
