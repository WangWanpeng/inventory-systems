"""This code is an implementation of the algorithm of Federgruen and
Zheng, OR, 1991, to compute the optimal (Q,r)-policy for
single-item periodic review systems.

"""

from functools import lru_cache
import numpy as np
from scipy.stats import poisson

import unittest

class Federgruen_Zheng_Qr:
    def __init__(self, X, f, K, labda):
        self.f = f
        self.X = X
        self.K = K
        self.labda = labda

    @lru_cache(maxsize=None)
    def G(self, y):
        return self.X.expect(lambda j: self.f(y-j), 0, np.inf)

    @lru_cache(maxsize=None)
    def c(self, r, Q):  # long-run average cost
        S = sum(self.G(y) for y in range(r+1, r+Q+1))/Q
        S += self.K*self.labda/Q
        return S

    def optimize(self):
        # obtain first estimate for r
        y = 0
        while self.G(y) > self.G(y+1):
            y += 1

        r = y-1
        Q = 1

        # now optimize over r and Q
        while True:
            if self.c(r-1, Q) < self.c(r, Q):
                r -= 1
            elif self.c(r, Q+1) < self.c(r, Q):
                Q += 1
            elif self.c(r-1, Q+1) < self.c(r, Q):
                Q += 1
                r -= 1
            else:
                break

        return r, Q


class Test(unittest.TestCase):

    # Values of Hadley and Whitin, example 4.10, page 195.
    # The results of HW are not entirely correct due to a typo in their text,
    # see my implementation of Hadley and Whitin's formulae for the Q,r model

    def test_holding_cost(self):
        labda = 400  # arrival rate
        tau = 0.25   # lead time
        A = K = 0    # ordering cost, = 0 if only considering inventory cost
        C = 10.0     # buying cost per item
        I = 0.2      # interest per unit time
        pi = 0.1     # stockout cost per item
        pihat = 0.3  # backlog cost per item per unit time

        h = I * C  # holding cost
        b = pihat  # backlogging cost per item per unit time

        X = poisson(labda*tau)  # demand during lead time
        f = holding_cost = lambda j: h*np.maximum(j, 0)

        r, Q = 96, 19
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        self.assertAlmostEqual(fz.c(r=r, Q=Q), 16.38452936, places=6)

    def test_backlogging_cost(self):
        labda = 400  # arrival rate
        tau = 0.25   # lead time
        A = K = 0    # ordering cost
        C = 10.0     # buying cost per item
        I = 0.2      # interest per unit time
        pi = 0.1     # stockout cost per item
        pihat = 0.3  # backlog cost per item per unit time

        h = I * C  # holding cost
        b = pihat  # backlogging cost per item per unit time

        X = poisson(labda*tau)  # demand during lead time
        f = backlogging_cost = lambda j: b*np.maximum(-j, 0)

        r, Q = 96, 19
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        self.assertAlmostEqual(fz.c(r=r, Q=Q), 0.65767940414, places=6)

    def test_stockout_cost(self):
        labda = 400  # arrival rate
        tau = 0.25   # lead time
        A = K = 0    # ordering cost
        C = 10.0     # buying cost per item
        I = 0.2      # interest per unit time
        pi = 0.1     # stockout cost per item
        pihat = 0.3  # backlog cost per item per unit time

        h = I * C  # holding cost
        b = pihat  # backlogging cost per item per unit time

        X = poisson(labda*tau)  # demand during lead time
        f = stockout_cost = lambda j: pi*labda*(j <= 0)

        r, Q = 96, 19
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        self.assertAlmostEqual(fz.c(r=r, Q=Q), 12.5313312246, places=6)

    def test_ordering_cost(self):
        labda = 400  # arrival rate
        tau = 0.25   # lead time
        A = K = 0.16  # ordering cost
        C = 10.0     # buying cost per item
        I = 0.2      # interest per unit time
        pi = 0.1     # stockout cost per item
        pihat = 0.3  # backlog cost per item per unit time

        h = I * C  # holding cost
        b = pihat  # backlogging cost per item per unit time

        X = poisson(labda*tau)  # demand during lead time
        f = lambda j: 0

        r, Q = 96, 19
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        self.assertAlmostEqual(fz.c(r=r, Q=Q), 3.3684210, places=6)

    def test_total_cost(self):
        labda = 400  # arrival rate
        tau = 0.25   # lead time
        A = K = 0.16 # ordering cost
        C = 10.0     # buying cost per item
        I = 0.2      # interest per unit time
        pi = 0.1     # stockout cost per item
        pihat = 0.3  # backlog cost per item per unit time

        h = I * C  # holding cost
        b = pihat  # backlogging cost per item per unit time

        X = poisson(labda*tau)  # demand during lead time

        holding_cost = lambda j: h*np.maximum(j, 0)
        backlogging_cost = lambda j: b*np.maximum(-j, 0)
        stockout_cost = lambda j: pi*labda*(j <= 0)

        f = lambda j: holding_cost(j)+backlogging_cost(j)+\
            stockout_cost(j)
        r, Q = 96, 19
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        self.assertAlmostEqual(fz.c(r=r, Q=Q), 32.9419610, places=6)

if __name__ == '__main__':
    unittest.main()

quit()

if __name__ == "__main__":
    h = G.I * G.C
    b = G.pihat
    f = lambda j: h*np.maximum(j,0) + b*np.maximum(-j,0) + G.pi*G.labda*(j<=0)
    qrfz = QrFZ(G.X,
                f= f, 
                K=G.A,
                labda=G.labda)
    
    r, Q = qrfz.optimize()
    r
    Q
    qrfz.c(r,Q)
