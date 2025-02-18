import numpy as np
from numba import njit, vectorize
from scipy.optimize import root_scalar
from math import log, log1p, exp, atanh, sqrt

# gaussian integration
r, w = np.polynomial.hermite.hermgauss(199)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


# ---
@vectorize()
def second_moment_H(m, q):
    return 0.25 * (
        log(256)
        - 2 * (1 - q) * log1p(-q)
        - (1 - 2 * m + q) * log1p(-2 * m + q)
        - (1 + 2 * m + q) * log1p(2 * m + q)
    )


@vectorize()
def second_moment_bound(m, q, e, p):
    H = 0.25 * (
        log(256)
        - 2 * (1 - q) * log1p(-q)
        - (1 - 2 * m + q) * log1p(-2 * m + q)
        - (1 + 2 * m + q) * log1p(2 * m + q)
    )
    return H - 2 * e**2 * (1 - m**p) ** 2 / (1 + q**p)


@vectorize()
def first_moment_H(m):
    return -0.5 * (1 + m) * log1p(m) - 0.5 * (1 - m) * log1p(-m) + log(2)


@vectorize()
def annealed_entropy(m, e, p):
    H = -0.5 * (1 + m) * log1p(m) - 0.5 * (1 - m) * log1p(-m) + log(2)
    return H - e**2 * (1 - m**p) ** 2


@njit()
def deriv_ann_entropy(m, e, p):
    return 2 * e**2 * m ** (p - 1) * (-1 + m**p) * p + np.arctanh(m)


@njit()
def int_high_p(betat, p):
    betatp = betat * log(p)
    return 1 - np.sum(weights * (np.tanh(betatp * roots + betatp**2)))


def int_first_order(a):
    return 2 * np.exp(-2 * a * a) * np.sum(weights * (np.exp(-2 * (a * roots))))


@njit()
def int_high_p2(beta, p):
    betat = beta * sqrt(0.5 * p)
    return np.sum(
        weights * ((1 + 0.5 * roots) / np.cosh(betat * roots + betat**2) ** 2)
    ) + 1 / ((p - 1) * betat)


# ------------
# Fixed points equations
# ------------


@vectorize()
def beta_q_e(q, m, e, p, h):
    return 2 * exp(log(-e) + log1p(-(m**p)) - log1p(-(q**p)))


@vectorize()
def e_q_beta(q, m, beta, p):
    return -exp(log(beta / 2) + log1p(-(q**p)) - log1p(-(m**p)))


@njit()
def compute_q_FP(m, q, h, p, e):
    return np.sum(
        weights
        * (
            np.tanh(
                beta_q_e(q, m, e, p, h)
                * (p * (-e) * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
            )
            ** 2
        )
    )


@njit()
def compute_m_FP(m : float, q: float, h: float, p: int, e: float):
    return np.sum(
        weights
        * np.tanh(
            beta_q_e(q, m, e, p, h)
            * (p * (-e) * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
        )
    )


@njit()
def f_FP(m: float, q: float, h: float, p: int, e: float):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
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
        + 0.25 * beta**2
        - 0.25 * beta**2 * p * q ** (p - 1)
        + integral
    ) / (-beta) + h * m


@njit()
def deltaf_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
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
        (
            0.25 * beta**2 * (p - 1) * q**p
            - (p - 1) * beta * J0 * m**p
            + 0.25 * beta**2
            - 0.25 * beta**2 * p * q ** (p - 1)
            + integral
        )
        / (-beta)
        + h * m
        - (0.25 * beta_q_e(0, 0, e, p, 0) ** 2 + np.log(2)) / (-beta_q_e(0, 0, e, p, 0))
    )


@njit()
def s_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p, h)
    J0 = -e
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
        (
            0.25 * beta**2 * (p - 1) * q**p
            - (p - 1) * beta * J0 * m**p
            + 0.25 * beta**2
            - 0.25 * beta**2 * p * q ** (p - 1)
            + integral
        )
        - beta * h * m
        + beta * (-J0 * m**p - 0.5 * beta * (1 - q**p))
    )


def dAT_condition(q, m, h, beta, J0, p):
    integral = np.sum(
        weights
        * (
            np.cosh(
                np.sqrt(0.5 * p * beta**2 * q ** (p - 1)) * roots
                + beta * h
                + beta * J0 * p * m ** (p - 1)
            )
            ** (-4)
        )
    )
    return 1 - 0.5 * p * (p - 1) * beta**2 * q ** (p - 2) * integral


# @njit()
def compute_h(h, m, q, p, e):
    return compute_m_FP(m, q, h, p, e) - m


# ---
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
            bracket=[-1e3, 1e3],
            args=(m, q, p, e),
            method="bisect",
            xtol=tol,
            rtol=tol,
        ).root
        q_new = compute_q_FP(m, q, h_new, p, e)
        if q_new >= 1:
            print(f"q_new = {q_new}")

        err = max(abs(h_new - h), abs(q_new - q))
        h = blend * h + (1 - blend) * h_new
        q = blend * q + (1 - blend) * q_new
        if iter > 10_000:
            raise ValueError("Fixed point iteration did not converge")

    return h, q


@njit()
def compute_m_FP_T(m, q, h, p, T, J0):
    return np.sum(
        weights
        * np.tanh(
            (1 / T)
            * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
        )
    )

@njit()
def compute_q_FP_T(m, q, h, p, T, J0):
    return np.sum(
        weights
        * np.tanh(
            (1 / T)
            * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2) + h)
        )
        ** 2
    )


@njit()
def compute_h_T(h, m, q, p, T, J0):
    return compute_m_FP_T(m, q, h, p, T, J0) - m


def fixed_points_h_q_T(m, T, p, J0, blend=0.25, tol=1e-9, h_init=-0.1, q_init=0.01):
    err = 1e10
    q = q_init
    h = h_init
    iter = 0
    while err > 1e1 * tol:
        iter += 1
        h_new = root_scalar(
            compute_h_T,
            bracket=[-1e3, 1e3],
            args=(m, q, p, T, J0),
            method="bisect",
            xtol=tol,
            rtol=tol,
        ).root
        q_new = compute_q_FP_T(m, q, h_new, p, T, J0)
        if q_new >= 1:
            print(f"q_new = {q_new}")

        err = max(abs(h_new - h), abs(q_new - q))
        h = blend * h + (1 - blend) * h_new
        q = blend * q + (1 - blend) * q_new
        if iter > 10_000:
            raise ValueError("Fixed point iteration did not converge")

    return h, q


@njit()
def f_FP_T(m, q, h, p, T, J0):
    beta = 1/T
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
        + 0.25 * beta**2
        - 0.25 * beta**2 * p * q ** (p - 1)
        + integral
    ) / (-beta) + h * m

