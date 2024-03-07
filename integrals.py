import numpy as np
from numba import njit

r, w = np.polynomial.hermite.hermgauss(99)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


@njit()
def beta_q_e(q, e):
    return -2 * e / (1 - q**3)


@njit()
def compute_q(m, J0, q, e):
    return np.sum(
        weights
        * (
            np.tanh(-2 * e * (3 * J0 * m**2 + roots * np.sqrt(3 / 2) * q) / (1 - q**3))
            ** 2
        )
    )


@njit()
def compute_q_FP(m, J0, q, h, p, beta):
    return np.sum(
        weights
        * (
            np.tanh(
                beta
                * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
            )
            ** 2
        )
    )


@njit()
def compute_m(m, J0, q, e):
    return np.sum(
        weights
        * np.tanh(-2 * e * (3 * J0 * m**2 + roots * np.sqrt(3 / 2) * q) / (1 - q**3))
    )


@njit()
def compute_m_FP(m, J0, q, h, p, beta):
    return np.sum(
        weights
        * np.tanh(
            beta * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
        )
    )


@njit()
def compute_m_q_fixedpoint_FP(J0, h, p, beta, blend=0.8, tol=1e-7):
    err = 1.0
    m = 0.5
    q = 0.5

    while err > tol:
        q_new = compute_q_FP(m, J0, q, h, p, beta)
        m_new = compute_m_FP(m, J0, m, h, p, beta)
        err = max([abs(q_new - q), abs(m_new - m)])
        m = blend * m_new + (1 - blend) * m
        q = blend * q_new + (1 - blend) * q

    return m, q


@njit()
def f_FP(m, J0, q, h, p, beta):
    integral = np.sum(
        weights
        * (
            np.log(
                2
                * np.cosh(
                    beta
                    * (
                        p * J0 * m ** (p - 1)
                        + roots * np.sqrt(p * q ** (p - 1) / 2)
                        + h
                    )
                )
            )
        )
    )
    return (
        0.25 * beta**2 * (p - 1) * q**p
        - (p - 1) * beta * J0 * m**p
        - 0.25 * beta**2 * p * q ** (p - 1)
        + integral
    )


@njit()
def compute_s(m, J0, q, e):
    beta = beta_q_e(q, e)
    i = np.sum(
        weights
        * (
            np.log(
                2
                * np.cosh(
                    -2 * e * (3 * J0 * m**2 + roots * np.sqrt(3 / 2) * q) / (1 - q**3)
                )
            )
        )
    )
    return beta**2 * q**3 - 3 * beta * J0 * m**3 - 0.25 * beta**2 * (1 + 3 * q**2) + i
