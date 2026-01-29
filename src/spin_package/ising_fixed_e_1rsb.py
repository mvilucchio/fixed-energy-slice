import numpy as np
from numba import njit, vectorize
from scipy.optimize import root_scalar
from math import log, log1p, exp, atanh, sqrt
from numpy import log1p

# from scipy.special import logsumexp

# gaussian integration
r, w = np.polynomial.hermite.hermgauss(71)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


@njit
def logsumexp(a):
    a_max = np.max(a)
    if np.isinf(a_max) and a_max < 0:  # All -inf
        return -np.inf
    tmp = np.exp(a - a_max)
    s = np.sum(tmp)
    return a_max + np.log(s)


@njit()
def logcosh(x):
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + log1p(p) - np.log(2)


@njit()
def compute_G(m, q0, q1, h, p, e, root2):
    return (
        root2 * np.sqrt(0.5 * p * q0 ** (p - 1))
        - e * p * m ** (p - 1)
        + roots * np.sqrt(0.5 * p * (q1 ** (p - 1) - q0 ** (p - 1)))
        + h
    )


# ------------
# Fixed points equations
# ------------


@njit()
def compute_denom(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    # return np.sum(
    #    weights
    #    * (
    #        np.cosh(
    #            beta
    #            * G
    #        )
    #        ** x
    #    )
    # )
    return np.exp(logsumexp(np.log(weights) + x * logcosh(beta * G)))


@njit()
def compute_num1(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    print("compute_num1 G:", G)
    return np.sum(weights * (np.cosh(beta * G) ** x * np.tanh(beta * G)))


@njit()
def compute_num2(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    return np.sum(weights * (np.cosh(beta * G) ** x * np.tanh(beta * G) ** 2))


@njit()
def compute_num_f(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    return np.sum(weights * (np.cosh(beta * G) ** x * np.log(np.cosh(beta * G))))


@njit()
def compute_num_e1(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    return np.sum(weights * G * (np.cosh(beta * G) ** (x - 1) * np.sinh(beta * G)))


@njit()
def compute_num_e2(beta, x, m, q0, q1, h, p, e, root2):
    G = compute_G(m, q0, q1, h, p, e, root2)
    return np.sum(
        weights
        * G
        * (np.cosh(beta * G) ** (x - 1) * np.sinh(beta * G))
        * (1 / x + np.log(np.cosh(beta * G)))
    )


@njit()
def compute_m_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    # beta = beta_q_e(q0, q1, m, e, p, h, x)
    return np.sum(
        weights
        * np.array(
            [
                compute_num1(beta, x, m, q0, q1, h, p, e, root2)
                / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                for root2 in roots
            ]
        )
    )


@njit()
def compute_q0_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    # beta = beta_q_e(q0, q1, m, e, p, h, x)
    return np.sum(
        weights
        * np.array(
            [
                compute_num1(beta, x, m, q0, q1, h, p, e, root2)
                / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                for root2 in roots
            ]
        )
        ** 2
    )


@njit()
def compute_q1_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    # beta = beta_q_e(q0, q1, m, e, p, h, x)
    return np.sum(
        weights
        * np.array(
            [
                compute_num2(beta, x, m, q0, q1, h, p, e, root2)
                / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                for root2 in roots
            ]
        )
    )


@njit()
def compute_e_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    return (
        -e * p * m**p
        + 0.5 * beta * (1 + (p - 1) * q1**p - p * q1 ** (p - 1))
        + beta * x * (p - 1) * (q0**p - q1**p)
        + np.sum(
            weights
            * (
                -np.array(
                    [
                        (
                            compute_num_e1(beta, x, m, q0, q1, h, p, e, root2)
                            + x * compute_num_e2(beta, x, m, q0, q1, h, p, e, root2)
                        )
                        / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                        for root2 in roots
                    ]
                )
                + x
                * np.array(
                    [
                        (
                            compute_num_f(beta, x, m, q0, q1, h, p, e, root2)
                            * compute_num_e1(beta, x, m, q0, q1, h, p, e, root2)
                        )
                        / compute_denom(beta, x, m, q0, q1, h, p, e, root2) ** 2
                        for root2 in roots
                    ]
                )
            )
        )
    )


# @njit()
def compute_beta(beta, m, q0, q1, h, p, e, x):
    return compute_e_FP(beta, m, q0, q1, h, p, e, x) - e


# @vectorize()
# @njit()
def beta_q_e(q0, q1, m, e, p, h, x, tol=1e-9):
    return root_scalar(
        compute_beta,
        bracket=[1e-5, 1e5],
        args=(m, q0, q1, h, p, e, x),
        # method="bisect",
        xtol=tol,
        rtol=tol,
    ).root


@njit()
def compute_f_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    # beta = beta_q_e(q0, q1, m, e, p, h, x)
    integral = np.sum(
        weights
        * (
            np.array(
                [
                    compute_num_f(beta, x, m, q0, q1, h, p, e, root2)
                    / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                    for root2 in roots
                ]
            )
        )
    )
    return (
        beta * e * p * m**p
        + 0.25 * (1 - 2 * x) * beta**2 * (p - 1) * q1**p
        + 0.5 * x * (p - 1) * beta**2 * q0**p
        + 0.25 * beta**2 * (1 - p * q1 ** (p - 1))
        + integral
        + np.log(2)
    ) / (-beta) + h * m


# Complexity
@njit()
def compute_Sigma_FP(
    beta: float, m: float, q0: float, q1: float, h: float, p: int, e: float, x: float
):
    # beta = beta_q_e(q0, q1, m, e, p, h, x)
    integral = np.sum(
        weights
        * (
            np.array(
                [
                    np.log(compute_denom(beta, x, m, q0, q1, h, p, e, root2))
                    for root2 in roots
                ]
            )
            - x
            * np.array(
                [
                    compute_num_f(beta, x, m, q0, q1, h, p, e, root2)
                    / compute_denom(beta, x, m, q0, q1, h, p, e, root2)
                    for root2 in roots
                ]
            )
        )
    )
    return 0.25 * beta**2 * x**2 * (p - 1) * (q1**p - q0**p) + integral


# @njit()
def compute_h(h, m, q0, q1, p, e, x):
    beta = beta_q_e(q0, q1, m, e, p, h, x)
    print(
        f"compute_h: beta = {beta}, h = {h}, m = {compute_m_FP(beta, m , q0, q1, h, p, e, x)}"
    )
    return compute_m_FP(beta, m, q0, q1, h, p, e, x) - m


# ---
# @njit()
def fixed_points_h_q(
    m, e, p, x, blend=0.25, tol=1e-9, h_init=-0.1, q0_init=0.01, q1_init=0.01
):
    err = 1e10
    q0 = q0_init
    q1 = q1_init
    h = h_init
    iter = 0
    while err > 1e1 * tol:
        iter += 1
        h_new = root_scalar(
            compute_h,
            bracket=[-1e3, 1e3],
            args=(m, q0, q1, p, e, x),
            method="bisect",
            xtol=tol,
            rtol=tol,
        ).root
        beta = beta_q_e(q0, q1, m, e, p, h_new, x)
        q0_new = compute_q0_FP(beta, m, q0, q1, h, p, e, x)
        q1_new = compute_q1_FP(beta, m, q0, q1, h, p, e, x)
        if q0_new >= 1 or q1_new >= 1:
            print(f"q0_new = {q0_new}, q1_new = {q1_new}, h_new = {h_new}")

        err = max(abs(h_new - h), abs(q0_new - q0), abs(q1_new - q1))
        h = blend * h + (1 - blend) * h_new
        q0 = blend * q0 + (1 - blend) * q0_new
        q1 = blend * q1 + (1 - blend) * q1_new
        if iter > 10_000:
            raise ValueError("Fixed point iteration did not converge")

    return h, q0, q1, beta
