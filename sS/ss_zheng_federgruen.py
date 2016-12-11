"""This code is an implementation of the algorithm of Zheng and
Federgruen Zheng, OR, 1991, to compute the optimal (s,S)-policy for
single-item periodic review systems.  Gerlach van der Heijde provided
the implemented  the method findOptimalPolicy().

"""

from functools import lru_cache

import numpy as np
from scipy.stats import poisson
import unittest


class ZhengFedergruen:
    def __init__(self, X, f, K, mu):
        self.f = f
        self.X = X
        self.p = self.X.pmf
        self.K = K
        self.mu = mu

    @lru_cache(maxsize=None)
    def G(self, y):
        return self.X.expect(lambda j: self.f(y-j), 0, np.inf)

    @lru_cache(maxsize=None)
    def m(self, j):  # 2a
        if j == 0:
            return 1./(1. - self.p(0))
        else:  # 2b
            res = sum(self.p(l)*self.m(j-l) for l in range(1, j+1))
            res /= (1. - self.p(0))
            return res

    @lru_cache(maxsize=None)
    def M(self, j):
        if j == 0:
            return 0.
        else:
            return self.M(j-1) + self.m(j-1)

    def k(self, s, y):
        res = self.K
        res += sum(self.m(j)*self.G(y-j) for j in range(y-s))
        return res

    def c(self, s, S):
        return self.k(s, S)/self.M(S-s)

    def findOptimalPolicy(self, ystar):
        s = ystar - 1  # upper bound for s
        S_0 = ystar + 0  # lower bound for S_0
        # calculate the optimal s for S fixed at its lower bound S0
        while self.c(s, S_0) > self.G(s):
            s -= 1
        s_0 = s  # + 0 #optimal value of s for S0
        c0 = self.c(s_0, S_0)  # costs for this starting value
        S0 = S_0  # + 0  # S0 = S^0 of the paper
        S = S0+1
        while self.G(S) <= c0:
            if self.c(s, S) < c0:
                S0 = S+0
                while self.c(s, S0) <= self.G(s+1):
                    s += 1
                c0 = self.c(s, S0)
            S += 1
        self.s_star = s
        self.S_star = S0
        return s, S0

class Test(unittest.TestCase):

    # values of Feng and Xiao, IEE Transactions 2000

    def test_1(self):
        mu = 10
        K = 64
        b = 9.
        h = 1.
        X = poisson(mu)

        def f(y): return b*np.maximum(0, -y) + h*np.maximum(0, y)

        ystar = X.ppf(b/(b+h)).astype(int)

        zf = ZhengFedergruen(X, f, K, mu)
        s, S = zf.findOptimalPolicy(ystar)
        self.assertEqual((s, S), (6, 40))
        self.assertAlmostEqual(zf.c(s, S), 35.0215552, places=6)

    def test_2(self):
        mu = 20
        K = 64
        b = 9.
        h = 1.
        X = poisson(mu)

        def f(y): return b*np.maximum(0, -y) + h*np.maximum(0, y)

        ystar = X.ppf(b/(b+h)).astype(int)

        zf = ZhengFedergruen(X, f, K, mu)
        s, S = zf.findOptimalPolicy(ystar)
        self.assertEqual((s, S), (14, 62))
        self.assertAlmostEqual(zf.c(s, S), 49.173036, places=6)

    def test_3(self):
        mu = 64
        K = 64
        b = 9.
        h = 1.
        X = poisson(mu)

        def f(y): return b*np.maximum(0, -y) + h*np.maximum(0, y)

        ystar = X.ppf(b/(b+h)).astype(int)

        zf = ZhengFedergruen(X, f, K, mu)
        s, S = zf.findOptimalPolicy(ystar)
        self.assertEqual((s, S), (55, 74))
        self.assertAlmostEqual(zf.c(s, S), 78.402321, places=6)


if __name__ == '__main__':
    unittest.main()
