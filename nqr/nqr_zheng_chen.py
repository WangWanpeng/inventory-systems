"""
This is the nqr model of Zheng and Chen, naval research
"""

from functools import lru_cache
import numpy as np
from scipy.stats import poisson

import unittest


@lru_cache(maxsize=None)
def find_argmin(f):
    # Assume that f is quasi convex
    min_f = f(0)
    y = 0
    # search to the right
    while f(y+1) <= min_f:
        min_f = f(y+1)
        y += 1
    # search to the left
    while f(y-1) <= min_f:
        min_f = f(y-1)
        y -= 1
    return y


class Zheng_Chen:
    def __init__(self, D, D_L, f, K, mu):
        self.D = D
        self.D_L = D_L
        self.f = f
        self.K = K
        self.mu = mu

    @lru_cache(maxsize=None)
    def G(self, y):
        return self.D_L.expect(lambda j: self.f(y-j), 0, np.inf)

    @lru_cache(maxsize=None)
    def P(self, y):
        return self.D.sf(y-1)

    @lru_cache(maxsize=None)
    def c(self, r, Q):
        S = self.K*sum(self.P(j) for j in range(1, Q+1))
        S += sum(self.G(y) for y in range(r+1, r+Q+1))
        return S/Q

    def C(self, Q):
        pass

    def optimize(self):
        y1 = find_argmin(self.G)
        G1 = self.G(y1)
        Q = 1
        a = b = y1
        C = self.K*self.P(1) + G1
        C_bar = self.K*self.mu + G1
        self.r_star = y1-1
        self.Q_star = 1
        self.C_star = C
        while C_bar >= min(self.G(a-1), self.G(b+1)):
            Q += 1
            if self.G(a-1) <= self.G(b+1):
                a -= 1
                C_bar = ((Q-1)*C_bar + self.G(a))/Q
                C = ((Q-1)*C + self.K*self.P(Q) + self.G(a))/Q
            else:
                b += 1
                C_bar = ((Q-1)*C_bar + self.G(b))/Q
                C = ((Q-1)*C + self.K*self.P(Q) + self.G(b))/Q
            if C < self.C_star:  # typo in the algo, C instead of C(Q)?
                self.r_star = a-1
                self.Q_star = Q
                self.C_star = C


class Test(unittest.TestCase):
    # values of table 2 of Zheng and Chen

    def test_one(self):
        h = 1
        b = 9
        K = 32

        L, mu = 0, 26
        D = poisson(mu)
        D_L = poisson(mu*(L+1))

        def holding_cost(j): return h*np.maximum(j, 0)

        def backlogging_cost(j): return b*np.maximum(-j, 0)

        def f(j): return holding_cost(j) + backlogging_cost(j)

        zc = Zheng_Chen(D, D_L, f, K, mu)
        zc.optimize()
        self.assertEqual((zc.r_star, zc.Q_star), (32, 1))
        self.assertAlmostEqual(zc.C_star, 41.31, places=2)

    def test_many(self):
        h = 1
        b = 9
        K = 32

        def holding_cost(j): return h*np.maximum(j, 0)

        def backlogging_cost(j): return b*np.maximum(-j, 0)

        def f(j): return holding_cost(j) + backlogging_cost(j)

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

            zc = Zheng_Chen(D, D_L, f, K, mu)
            zc.optimize()
            self.assertEqual((zc.r_star, zc.Q_star), (r, Q))
            self.assertAlmostEqual(zc.C_star, C, places=2)

if __name__ == '__main__':
    unittest.main()

