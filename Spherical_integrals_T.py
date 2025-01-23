import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p, exp, pi


@vectorize()
def beta_q_e(q, m, e, p, h):
    return 2*exp( log(-e * (1 - m ** p) + 0 * m) - log1p(-q**p) )

@vectorize()
def e_q_beta(q, m, beta, p):
    return -exp( log(beta/2) + log1p(-q**p) ) - (0.5/beta) * m ** p

@njit()
def compute_m_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
    return (
        beta * (1 - q) * (J0 * p * m ** (p - 1) - h)
    )

@njit()
def compute_q_FP(m, q, p, T):
    beta = 1/T
    return (
        m ** 2 +
        beta ** 2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1) )
    )

def compute_h(h, m, q, p, e):
    return compute_m_FP(m, q, h, p, e) - m

# @njit()
def fixed_points_q(m, T, p, blend=0.01, tol=1e-9, h_init=-0.1, q_init=0.9):
    err = 1e10
    q = q_init
    iter = 0
    while err > 1e1 * tol:
        iter +=1
        q_new = compute_q_FP(m, q, p, T)
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
def f_FP(q, m, h, p, T):
    beta = 1/T
    J0 = 1/(2*T)
    return (
        0.5 * beta**2 * (0 - q**p) #switch to 1
        #+ 2 * beta * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * beta * J0 * m**p
        #+ 1 + log(2*pi)
    )/ 1#(-2 * beta)

def s_FP(q, m, h, p, T):
    beta = 1/T
    en = e_q_beta(q, m, beta, p)
    return beta*(en - f_FP(q, m, h, p, T) )

