"""
A Simulation of the Q,r Inventory Policy
==========================================


we simulate a periodic-review inventory process subjected to the
Q,r policy.

Model
--------------

Orders arrive according to a Poisson process with rate \mu at
an inventory.  We assume that orders are accepted in accordance to the
partial acceptance rule: orders are satisfied to the extent that
inventory level is available, for the rest they are rejected. When the
inventory position hits the reorder level r a replenishment
order of size Q is issued to replenish the inventory a
constant lead time L later.

We use the following notation

* D(t) is the demand during period t
* d(t) is the satisfied demand in period t
* I(t)` is the on hand inventory level at the end of  period t
* IP(t) is the inventory position at the end of  period t
* Q(t) is the ordered amount demand in period t.

We assume that replenishments arrive at the beginning of a period, and
that demand is to be met at the end of a period. This leads to the recursions

d(t) = \min\{D(t), I(t-1) + Q(t-L)\}
I(t) = I(t-1) + Q(t-L) - d(t)
Q(t) = Q\cdot\mathbf{1}\{IP(t-1) \leq r\}
IP(t) = IP(t-1) + Q(t) - d(t)

"""

import numpy as np
from scipy.stats import poisson
from pylab import plot, show, legend

np.random.seed(1)

mu = 2
L = 3
Q = 4
r = 7
N = 20

QQ = np.zeros(N) # is Q(t)
I = np.zeros(N)
IP = np.zeros(N)
d = np.zeros(N)

IP[0] = I[0] = r+Q
D = poisson(mu).rvs(N)


for t in range(1,L):
    d[t] = min(D[t], I[t-1])
    I[t] = I[t-1] - d[t]
    QQ[t] = Q * (IP[t-1] <= r)
    IP[t] = IP[t-1] + QQ[t] - d[t]

for t in range(L, N):
    d[t] = min(D[t], I[t-1] + QQ[t-L])
    I[t] = I[t-1] + QQ[t-L] - d[t]
    QQ[t] = Q * (IP[t-1] <= r)
    IP[t] = IP[t-1] + QQ[t] - d[t]

plot(I, label = "I", drawstyle="steps-post")
plot(IP, label = "IP", drawstyle = "steps-post")
plot(QQ, "8", label = "Q")
plot(d, "D", label = "d")
plot(D, "o", label = "D")
legend()
show()
