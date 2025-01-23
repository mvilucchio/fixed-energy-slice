import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p, exp, pi


@vectorize()
def beta_q_e(q, m, e, p, h):
    return 2*exp( log(-e * (1 - m ** p) + 0 * m) - log1p(-q**p) )


@njit()
def compute_m_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
    return (
        beta * (1 - q) * (J0 * p * m ** (p - 1) - h)
    )

@njit()
def compute_q_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    return (
        m ** 2 +
        beta ** 2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1) )
    )

def compute_h(h, m, q, p, e):
    return compute_m_FP(m, q, h, p, e) - m

# @njit()
def fixed_points_h_q(m, e, p, blend=0.25, tol=1e-9, h_init=-0.1, q_init=0.01):
    err = 1e10
    q = q_init
    h = h_init
    iter = 0
    while err > 1e1 * tol:
        iter += 1
        h_new = root_scalar(
            compute_h,
            bracket=[-1e4, 1e4],
            args=(m, q, p, e),
            method = "bisect",
            xtol=tol,
            rtol=tol,
        ).root
        q_new = compute_q_FP(m, q, h_new, p, e)
        
        if (q_new >= 1):
            print(f"q_new = {q_new}")

        err = max(abs(h_new - h), abs(q_new - q))
        h = blend * h + (1 - blend) * h_new
        q = blend * q + (1 - blend) * q_new
        if (iter > 10_000):
            raise ValueError('Fixed point iteration did not converge')
            #h = np.NaN
            #q = np.NaN

    return h, q

@njit()
def f_FP(q, m, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
    return (
        0.5 * beta**2 * (1 - q**p)
        + 2 * beta * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * beta * J0 * m**p
        + 1 + log(2*pi)
    )/ (-2 * beta)

@njit()
def s_FP(q, m, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    return beta*(e - f_FP(q, m, h, p, e) )



@vectorize()
def annealed_entropy(m, e, p):
    H = (1 + log(-2*(-1 + m**2)*pi))/2.
    return H - e**2 * (1 - m**p) ** 2