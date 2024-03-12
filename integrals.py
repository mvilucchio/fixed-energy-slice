import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p

# gaussian integration
r, w = np.polynomial.hermite.hermgauss(99)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


# ---
@vectorize()
def second_moment_bound(m, q, e, p):
    H = 0.25*(log(256) - 2*(1-q)*log1p(-q) - (1-2*m+q) * log1p(-2*m+q) - (1+2*m+q) * log1p(2*m+q))
    return H - 2*e**2 * (1 - m**p) ** 2/(1 + q**p)

@vectorize()
def annealed_entropy(m, e, p):
    H = -0.5 * (1 + m) * log1p(m) - 0.5 * (1 - m) * log1p(-m) + log(2)
    return H - e**2 * (1 - m**p) ** 2


# ---
@njit()
def beta_q_e(q, m, e, p):
    return -2 * e / (1 + m**p - q**p)


@njit()
def compute_q_FP(m, q, h, p, e):
    return np.sum(
        weights
        * (
            np.tanh(
                beta_q_e(q, m, e, p)
                * (
                    p * (0.5 * beta_q_e(q, m, e, p)) * m ** (p - 1)
                    + roots * np.sqrt(p * q ** (p - 1) / 2)
                    + h
                )
            )
            ** 2
        )
    )


@njit()
def compute_m_FP(m, q, h, p, e):
    return np.sum(
        weights
        * np.tanh(
            beta_q_e(q, m, e, p)
            * (
                p * (0.5 * beta_q_e(q, m, e, p)) * m ** (p - 1)
                + roots * np.sqrt(p * q ** (p - 1) / 2)
                + h
            )
        )
    )


@njit()
def f_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p)
    J0 = beta / 2
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
    beta = beta_q_e(q, m, e, p)
    J0 = beta / 2
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
        - (0.25 * beta_q_e(0, 0, e, p) ** 2 + np.log(2)) / (-beta_q_e(0, 0, e, p))
    )


@njit()
def s_FP(m, q, h, p, e):
    beta = beta_q_e(q, m, e, p)
    J0 = beta / 2
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
def fixed_points_h_q(m, e, p, blend=0.1, tol=1e-9, h_init=-0.1, q_init=0.5):
    err = 1e10
    q = q_init
    h = h_init

    while err > 1e1 * tol:
        # h_new = brent_root_finder(
        #     compute_h,
        #     -1_000,
        #     1_000,
        #     tol,
        #     tol,
        #     2_000,
        #     (m, q, p, e),
        # )
        h_new = root_scalar(
            compute_h,
            bracket=[-1000, 1000],
            args=(m, q, p, e),
            xtol=tol,
            rtol=tol,
        ).root
        q_new = compute_q_FP(m, q, h_new, p, e)

        err = max(abs(h_new - h), abs(q_new - q))
        h = blend * h + (1 - blend) * h_new
        q = blend * q + (1 - blend) * q_new

    return h, q


def multiple_empty_arrays(shape, n):
    return [np.empty(shape) for _ in range(n)]


def observables(shape):
    return {
        "delta_f": np.empty(shape),
        "T": np.empty(shape),
        "dAT": np.empty(shape),
        "q": np.empty(shape),
        "h": np.empty(shape),
        "s": np.empty(shape),
    }


# @dataclass
# class Observables:
#     T: np.ndarray
#     q: np.ndarray
#     h: np.ndarray
#     delta_f: np.ndarray
#     s: np.ndarray
#     dAT: np.ndarray

# def observables(shape):
#     return Observables(
#         T=np.empty(shape),
#         q=np.empty(shape),
#         h=np.empty(shape),
#         delta_f=np.empty(shape),
#         s=np.empty(shape),
#         dAT=np.empty(shape)
#     )
