from functools import lru_cache


# THIS FUNCTION DOES NOT RESTART, DUE TO THE MEMOIZATION

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
