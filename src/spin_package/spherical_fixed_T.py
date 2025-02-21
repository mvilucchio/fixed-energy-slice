import numpy as np
from numba import njit
from math import sqrt, log, pi, log1p, exp


@njit()
def e_J0(q, m, p, β, J0):
    # return J0 * m**p - 0.5 * β * (1 - q**p)
    # return exp(log(J0) + p * log(m)) - exp(log(0.5) + log(β) + log1p(-(q**p)))
    A = log(J0) + p * log(m)
    B = log(0.5) + log(β) + log1p(-(q**p))
    M = max(A, B)
    return exp(M) * (exp(A - M) - exp(B - M))


@njit()
def compute_m_FP_J0(m, q, p, β, J0):
    return β * (1 - q) * J0 * p * m ** (p - 1)


@njit()
def compute_q_FP_J0(m, q, p, β):
    return m**2 + β**2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1))


@njit()
def newm(m: float, q: float, p: int, β: float):
    # return sqrt(q - (1 - q) * (1 - q) * (0.5 * β**2 * p * pow(q, p - 1)))
    if q == 0:
        return 0.0

    if q == 1:
        return 1.0

    # For q < 1 and very large p, q^(p-1) approaches 0
    # When q is small but p is large, approximation to sqrt(q) is valid
    one_minus_q = 1.0 - q
    one_minus_q_squared = one_minus_q * one_minus_q
    β_squared = β * β

    if q < 1.0 and (p - 1.0) * log(q) < -700.0:
        return sqrt(q)

    log_q_pow_p_minus_1 = (p - 1.0) * log1p(q - 1)
    log_scaling_term = log(0.5) + 2.0 * log(β) + log(p) + log_q_pow_p_minus_1

    if log_scaling_term < -700.0:
        return sqrt(q)

    scaling_term = one_minus_q_squared * exp(log_scaling_term)

    # Make sure we don't get negative values inside the sqrt due to numerical errors
    # arg = max(0.0, q - scaling_term)

    # return sqrt(arg)
    return sqrt(q - scaling_term)


@njit()
def newq(m: float, q: float, p: int, beta: float, J0: float):
    # return 1 - m / (beta * p * J0 * pow(m, p - 1))
    if m == 0:
        return 0.0

    if m == 1:
        return 1.0 - 1.0 / (beta * p * J0)

    if m < 1.0:
        threshold = -700
        if (p - 1) * log(m) < threshold:
            return 1.0

        log_term = log(beta) + log(p) + log(J0) + (p - 2) * log(m)

        if log_term > 300:
            return 1.0

        return 1.0 - exp(-log_term)


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
