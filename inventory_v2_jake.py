
## Montecarlo backtesting with numpy ##
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

test = pd.read_csv("/home/jake/Code/kaiseki/local_data/testdata.csv")
test = test.tail(200)
test = test.to_numpy()
# test = test[:,1]

# Parameters for mid price backtesting:

# S0 = 100.  # initial price
T = 1.0  # time
sigma = 0.04  # volatility = 2
# M = 200  # number of time steps
M = len(test)
dt = T / M  # time step
Sim = 1000  # number of simulations
gamma = 0.1  # risk aversion = 0.1
k = 1.5 # set to 1.5
A = 140
I = 1


# Results:
AverageSpread = []
Profit = []
Std = []

for i in range(1, Sim + 1):

    ###reservation price:
    ### r(s,t) = s - q * gamma * sigma**2 * (T-t)

    S = test[:,1]
    Bid = np.zeros((M + 1, I))
    Ask = np.zeros((M + 1, I))
    ReservPrice = np.zeros((M + 1, I))
    spread = np.zeros((M + 1, I))
    deltaB = np.zeros((M + 1, I))
    deltaA = np.zeros((M + 1, I))
    q = np.zeros((M + 1, I))
    w = np.zeros((M + 1, I))
    equity = np.zeros((M + 1, I))


    ReservPrice[0] = S[0]
    Bid[0] = S[0]
    Ask[0] = S[0]
    spread[0] = 0
    deltaB[0] = 0
    deltaA[0] = 0
    q[0] = 0  # position
    w[0] = 0  # wealth
    equity[0] = 0

    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        # S[t] = S[t - 1] + sigma * math.sqrt(dt) * z
        ReservPrice[t] = S[t] - q[t - 1] * gamma * (sigma ** 2) * (T - t / float(M))
        spread[t] = gamma * (sigma ** 2) * (T - t / float(M)) + (2 / gamma) * math.log(1 + (gamma / k))
        Bid[t] = ReservPrice[t] - spread[t] / 2.
        Ask[t] = ReservPrice[t] + spread[t] / 2.

        deltaB[t] = S[t] - Bid[t]
        deltaA[t] = Ask[t] - S[t]

        lambdaA = A * np.exp(-k * deltaA[t])
        ProbA = 1 - np.exp(-lambdaA * dt)
        fa = random.random()

        lambdaB = A * np.exp(-k * deltaB[t])
        ProbB = 1 - np.exp(-lambdaB * dt)
        fb = random.random()

        if ProbB > fb and ProbA < fa:
            q[t] = q[t - 1] + 1
            w[t] = w[t - 1] - Bid[t]

        if ProbB < fb and ProbA > fa:
            q[t] = q[t - 1] - 1
            w[t] = w[t - 1] + Ask[t]
        if ProbB < fb and ProbA < fa:
            q[t] = q[t - 1]
            w[t] = w[t - 1]
        if ProbB > fb and ProbA > fa:
            q[t] = q[t - 1]
            w[t] = w[t - 1] - Bid[t]
            w[t] = w[t] + Ask[t]

        equity[t] = w[t] + q[t] * S[t]

    AverageSpread.append(spread.mean())
    Profit.append(equity[-1])
    Std.append(np.std(equity))

print
"                   Results              "
print
"----------------------------------------"

print
"%14s %21s" % ('statistic', 'value')
print
40 * "-"
print
"%14s %20.5f" % ("Average spread :", np.array(AverageSpread).mean())
print
"%16s %20.5f" % ("Profit :", np.array(Profit).mean())
print
"%16s %20.5f" % ("Std(Profit) :", np.array(Std).mean())

# Plots:

x = np.linspace(0., T, num=(M + 1))

fig = plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)  # number of rows, number of  columns, number of the subplot
plt.plot(x, S[:], lw=1., label='S')
plt.plot(x, Ask[:], lw=1., label='Ask')
plt.plot(x, Bid[:], lw=1., label='Bid')
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('P')
plt.title('Prices')
plt.subplot(2, 1, 2)
plt.plot(x, q[:], 'g', lw=1., label='q')  # plot 2 lines
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()
# Histogram of profit:

plt.figure(figsize=(7, 5))
plt.hist(np.array(Profit), label=['Inventory strategy'], bins=100)
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('pnl')
plt.ylabel('number of values')
plt.title('Histogram')


pnl = pd.DataFrame(equity)
pnl.plot()