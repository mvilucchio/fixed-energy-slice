import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p, exp, pi


@vectorize()
def beta_q_e(q, m, e, p):
    return 2*exp( log(-e * (1 - m ** p)) - log1p(-q**p) )

@njit()
def compute_q_FP(m, q, p, e):
    beta = -2 * e * (1 - m ** p) / (1 - q ** p)
    return (
        m ** 2 +
        beta ** 2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1) )
    )

# @njit()
def fixed_points_q(m, e, p, blend=0.01, tol=1e-9, q_init=0.9):
    err = 1e10
    q = q_init
    iter = 0
    while err > 1e1 * tol:
        iter +=1
        q_new = compute_q_FP(m, q, p, e)
        if (q_new >= 1):
            print(f"q_new = {q_new}")

        err = abs(q_new - q)
        q = blend * q + (1 - blend) * q_new
        if (iter > 10_000):
            raise ValueError('Fixed point iteration did not converge')
            #h = np.NaN
            #q = np.NaN

    return q

@njit()
def f_FP(q, m, p, e):
    beta = -2 * e * (1 - m ** p) / (1 - q ** p)
    J0 = -e
    return (
        0.5 * beta**2 * (1 - q**p) #switch to 1
        #+ 2 * beta * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * beta * J0 * m**p
        + 1 + log(2*pi)
    )/ (-2 * beta)

def s_FP(q, m, p, e):
    return beta_q_e(q, m, e, p)*( e - f_FP(q, m, p, e) )

@vectorize()
def annealed_entropy(m, e, p):
    H = (1 + log(-2*(-1 + m**2)*pi))/2.
    return H - e**2 * (1 - m**p) ** 2

