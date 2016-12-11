"""We implement the formulas of Hadley and Whitin, Section 4.7, to
compute the various cost components for a given single-item inventory
system under a (Q,r) policy. We assume the Q and r  given.

Note: G contains a set of global parameters.

"""

from scipy.stats import poisson


class Qr_HW:

    def alpha(self, v):  # eq 4.45
        # return G.mu*G.P(v) - v*G.P(v+1)
        return G.mu*G.X.sf(v-1) - v*G.X.sf(v)

    def Pout(self, r, Q):  # eq 4.44
        return (self.alpha(r) - self.alpha(r+Q))/Q

    def E(self, r, Q):  # eq. 4.46
        return G.labda*self.Pout(r, Q)

    def beta(self, v):  # eq 4.51
        # see def 4.51 in HW
        # res = G.mu*G.mu/2.*G.P(v-1)
        # res -= G.mu*v*G.P(v)
        # res += v*(v+1)/2*G.P(v+1)
        res = G.mu*G.mu/2.*G.X.sf(v-2)
        res -= G.mu*v*G.X.sf(v-1)
        res += v*(v+1)/2*G.X.sf(v)
        return res

    def B(self, r, Q):  # eq. 4.52
        return (self.beta(r) - self.beta(r+Q))/Q

    def D(self, r, Q):  # eq. 4.53
        return (Q+1.)/2. + r - G.mu + self.B(r, Q)

    def orderingCost(self, r, Q):
        return G.labda*G.A/Q

    def holdingCost(self, r, Q):
        return G.I*G.C*self.D(r, Q)

    def backloggingCost(self, r, Q):
        return G.pihat*self.B(r, Q)

    def stockoutCost(self, r, Q):
        return G.pi*self.E(r, Q)

    def totalCost(self, r, Q):  # eq 4.61
        tot = self.orderingCost(r, Q)
        tot += self.holdingCost(r, Q)
        tot += self.stockoutCost(r, Q)
        tot += self.backloggingCost(r, Q)
        return tot


class HW_data:
    # We use the values of the example of Section 4.10 of Hadley and
    # Whitin
    def __init__(self):
        self.labda = 400
        self.tau = 0.25
        self.A = 0.16
        self.C = 10.0
        self.I = 0.2
        self.pi = 0.1
        self.pihat = 0.3
        self.X = poisson(self.labda*self.tau)
        self.mu = self.X.mean()
        self.P = lambda i: self.X.sf(i-1)  # loss probability


if __name__ == "__main__":
    # And now we check the values of the example of Section 4.10 of
    # Hadley and Whitin. Our results are not completely the same due
    # to a tiny error in one of their formulas. See the accompyning
    # tex file for the correct result.

    G = HW_data()
    r, Q = 96, 19
    qr = Qr_HW()
    print("holding cost: ", qr.holdingCost(r, Q))
    print("backloggin: ", qr.backloggingCost(r, Q))
    print("ordering cost", qr.orderingCost(r, Q))
    print("stockout cost: ", qr.stockoutCost(r, Q))
    print("stockout probability: ", qr.Pout(r, Q))
    print("total cost: ", qr.totalCost(r, Q))
