"""This code is an implementation of the algorithm of Federgruen and
Zheng, OR, 1991, to compute the optimal (Q,r)-policy for
single-item periodic review systems.


To find the minimum of G below we first find an estimate for r. We
assume that the cost of backlogging is higher than the cost of
inventory. Hence, the minimum of G must be to the right of 0. Next, we
search for the optimal values of r and Q. Since we use memoization,
the computations of c(r,Q) are replaced by look-ups.  Hence, there is
no overlap in the computations.

"""

import sys
sys.path.append('../')

from functools import lru_cache
import numpy as np
from scipy.stats import poisson

import unittest

from common_functions import find_argmin, find_left_right_roots, argmin

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

    def test_find_optimal_qr(self):
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
        fz = Federgruen_Zheng_Qr(X, f, K, labda)
        r, Q = fz.optimize()
        self.assertEqual((r, Q), (96, 18))

    def test_compare_against_basestock(self):
        h = 1
        b = 9
        K = 32

        def holding_cost(j): return h*np.maximum(j, 0)

        def backlogging_cost(j): return b*np.maximum(-j, 0)

        def f(j): return holding_cost(j) + backlogging_cost(j)

        def basestock_value(qr, D):
            y = find_argmin(qr.G)
            cost = qr.G(y) + K*D.sf(0)
            return y, cost

        cases = [
            [0, 5, 3, 20, 18.14],
            [0, 10, 7, 28, 25.64],
            [0, 25, 21, 44, 40.55],
            [0, 26, 32, 1, 41.31],
            [1, 28, 52, 49, 45.05],
            [1, 29, 67, 1, 45.72],
            [1, 35, 80, 1, 47.04],
            [3, 5, 19, 21, 20.56],
            [3, 20, 78, 43, 41.04],
            [5, 14, 84, 36, 36.44],
            ]

        for case in cases:
            L, mu, r, Q, C = case
            D = poisson(mu)
            D_L = poisson(mu*(L+1))

            qr = Federgruen_Zheng_Qr(D_L, f, K, mu)
            qr.r, qr.Q = qr.optimize()

            s, cost = basestock_value(qr, D)

            print(Q, qr.r, qr.Q, qr.c(qr.r, qr.Q), s, cost, qr.c(s, 1))

            # self.assertLessEqual(qr.r, nqr.r_star)
            # self.assertLessEqual(nqr.Q_star, qr.Q)
            # self.assertLessEqual(nqr.C_star, qr.c(qr.r, qr.Q))


if __name__ == '__main__':
    unittest.main()

