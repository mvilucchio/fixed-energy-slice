import numpy as np
from numba import njit

r, w = np.polynomial.hermite.hermgauss(99)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)

# -------------------------
r_small, w_small = np.polynomial.hermite.hermgauss(30)

roots_small = np.sqrt(2) * np.array(r_small)
weights_small = np.array(w_small) / np.sqrt(np.pi)


@njit()
def compute_q_standard(m, q, p, β, J0):
    return np.sum(
        weights
        * (
            np.tanh(
                β
                * (
                    p * J0 * m ** (p - 1)
                    + roots * np.sqrt(p * q ** (p - 1) / 2)
                )
            )
            ** 2
        )
    )


@njit()
def compute_m_standard(m, q, p, β, J0):
    return np.sum(
        weights
        * np.tanh(
            β
            * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2))
        )
    )


@njit()
def compute_free_energy_standard(m, q, p, β, J0):
    integral = np.sum(
        weights
        * (
            np.log(
                2
                * np.cosh(
                    β
                    * (
                        p * (0.5 * β) * m ** (p - 1)
                        + roots * np.sqrt(p * q ** (p - 1) / 2)
                    )
                )
            )
        )
    )
    return -(
        0.25 * β * (p - 1) * q**p
        - J0 * (p - 1) * m**p
        + 0.25 * β
        - 0.25 * β * p * q ** (p - 1)
        + integral / β
    )


# def free_energy_1RSB(x, q0, q1, T, J0, p):
#     integral = np.sum(weights_small * ())

#     return (
#         J0 * p * x**p
#         - 0.25 * (1 - x) * (p - 1) * q1**p / T
#         - 0.25 * x * (p - 1) * q0**p / T
#         - 0.25 * x / T
#         + 0.25 * p * q1**p / T
#         - np.log(2) * T
#         + integral
#     )
