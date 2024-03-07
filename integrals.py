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
def compute_m(m, J0, q, e):
    return np.sum(
        weights
        * np.tanh(-2 * e * (3 * J0 * m**2 + roots * np.sqrt(3 / 2) * q) / (1 - q**3))
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
