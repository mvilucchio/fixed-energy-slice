import numpy as np
from numba import njit
from math import sqrt, log, pi, log1p


@njit()
def e_J0(q, m, p, β, J0):
    return J0 * m**p - 0.5 * β * (1 - q**p)


@njit()
def compute_m_FP_J0(m, q, p, β, J0):
    return β * (1 - q) * J0 * p * m ** (p - 1)


@njit()
def compute_q_FP_J0(m, q, p, β):
    return m**2 + β**2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1))


@njit()
def newm(thism, thisq, p, β):
    return sqrt(
        thisq - (1 - thisq) * (1 - thisq) * (0.5 * β**2 * p * pow(thisq, p - 1))
    )


@njit()
def newq(thism, thisq, p, β, J0):
    return 1 - thism / (β * p * J0 * pow(thism, p - 1))


def fixed_points_q_J0(m, β, p, blend=0.01, tol=1e-9, q_init=0.9):
    err = 1e10
    q = q_init
    iter = 0
    while err > 1e1 * tol:
        iter += 1
        q_new = compute_q_FP_J0(m, q, p, β)
        if q_new >= 1:
            print(f"q_new = {q_new}")

        err = abs(q_new - q)
        q = blend * q + (1 - blend) * q_new
        if iter > 10_000:
            raise ValueError("Fixed point iteration did not converge")
            # h = np.NaN
            # q = np.NaN

    return q


@njit()
def f_FP_J0(q, m, p, β, J0):
    # β = -2 * e * (1 - m ** p) / (1 - q ** p)
    return (
        0.5 * β**2 * (1 - q**p)  # switch to 1
        # + 2 * β * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * β * J0 * m**p
        + 1
        + log(2 * pi)
    ) / (-2 * β)


def s_FP_J0(q, m, p, β, J0):
    return β * (e_J0(q, m, p, β, J0) - f_FP_J0(q, m, p, β, J0))


def Hes_eig_J0(q, m, p, β, J0):
    return (
        -0.125
        * (
            4 * m**4 * q**2
            + 4 * β * J0 * m**p * (-1 + p) * p * (-1 + q) ** 3 * q**2
            - m**2
            * (
                β**2 * (-1 + p) * p * (-1 + q) ** 3 * q**p
                - 2 * q**2 * (1 + q * (-5 + 2 * q))
            )
            + np.sqrt(
                16
                * m**2
                * (-1 + q) ** 2
                * q**2
                * (
                    m**2
                    * (β**2 * (-1 + p) * p * (-1 + q) ** 3 * q**p + 2 * q**2 * (1 + q))
                    + β
                    * J0
                    * m**p
                    * (-1 + p)
                    * p
                    * (-1 + q)
                    * (
                        β**2 * (-1 + p) * p * (-1 + q) ** 3 * q**p
                        + 2 * q**2 * (1 - 2 * m**2 + q)
                    )
                )
                + (
                    4 * β * J0 * m**p * (-1 + p) * p * (-1 + q) ** 3 * q**2
                    + m**2
                    * (
                        -(β**2 * (-1 + p) * p * (-1 + q) ** 3 * q**p)
                        + 2 * q**2 * (1 + 2 * m**2 + q * (-5 + 2 * q))
                    )
                )
                ** 2
            )
        )
        / (β * m**2 * (-1 + q) ** 3 * q**2)
    )
