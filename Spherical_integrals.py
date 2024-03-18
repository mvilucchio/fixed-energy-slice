import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p

# gaussian integration
r, w = np.polynomial.hermite.hermgauss(99)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


@njit()
def beta_q_e_spherical(q, m, e, p):
    return -2 * e / (1 + m**p - q**p)


@njit()
def compute_q_FP_spherical(m, q, h, p, e):
    return


def free_energy_FP(q, m, beta, h, p, e):
    return (
        0.5 * beta**2 * (1 - q**p)
        + beta**2 * h**2 * (1 - q)
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 0.5 * beta**2 * m**p
    )


def test_fun_2(h, q, m, beta, p):
    return 0.5 * beta**2 * p * q ** (p - 1) + beta**2 * h**2 - (q - m**2) / (1 - q) ** 2


def compute_overlaps_spherical(m, p, beta):
    q = 1 - (0.25 * p * beta**2 * m ** (p - 2)) ** 2

    return q, h
