from functools import lru_cache
import numpy as np

@lru_cache(maxsize=None)
def find_argmin(f):
    # Assuming that f is quasi convex, find the minimizer.
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


def argmin(f, left, right):
    # Find the minimizer between the left and right bounds inclusive
    return min(range(left, right+1), key=lambda j: f(j))


def find_left_right_roots(f, g):
    # Find left and right roots of f(x)-g, assuming f is quasiconvex
    left = find_argmin(f)
    while f(left-1) < g:
        left -= 1
    right = find_argmin(f)
    while f(right+1) < g:
        right += 1
    return left, right


def distribution(pmf, eps):
    """Compute the entire distribution of probability mass function given
    by pmf. It is assumed that pmf(i) = 0 for i < 0.

    """
    n, tot, Z = 0, 0., {}
    while tot < 1-eps:
        p = pmf(n)
        Z[n] = p
        tot += p
        n += 1
    return {j: p/tot for j, p in Z.items()}
    


# compound Poisson probabilities
@lru_cache(maxsize=None)
def r(j, labda, t, a= (0., 1.)):
    if j == 0:
        return np.exp(-labda*t*(1.-a[0]))
    m = max(0,j-len(a)+1)
    return float(labda)*t/j * np.sum((j-k)*a[j-k]*r(k, labda, t, a) for k in range(m,j))

