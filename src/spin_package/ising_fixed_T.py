import numpy as np
from numba import njit
from scipy.optimize import root_scalar

r, w = np.polynomial.hermite.hermgauss(299)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


@njit()
def compute_q_standard(m: float, q: float, p: int, β: float, J0: float):
    return np.sum(
        weights
        * (
            np.tanh(β * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2)))
            ** 2
        )
    )


@njit()
def compute_m_standard(m: float, q: float, p: int, β: float, J0: float):
    return np.sum(
        weights
        * np.tanh(β * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2)))
    )


@njit()
def compute_free_energy_standard(m: float, q: float, p: int, β: float, J0: float):
    integral = np.sum(
        weights
        * (
            np.log(
                2
                * np.cosh(
                    β
                    * (
                        p * J0 * m ** (p - 1)
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


@njit()
def compute_energy_standard(m, q, p, β, J0):
    return -J0 * m**p - 0.5 * β * (1 - q**p)


def compute_Td_standard(p, blend=0.25, verbose=False, deltaT=0.01):
    T_init = 0.6
    m_init = 1.0
    q_init = 1.0

    m = m_init
    q = q_init
    T = T_init

    while deltaT > 1e-8:
        J0 = 1 / (2 * T)
        err = 1
        m_old = m
        q_old = q
        while err > 1e-8:
            m_new = compute_m_standard(m, q, p, 1 / T, J0)
            q_new = compute_q_standard(m, q, p, 1 / T, J0)

            err = max(abs(q_new - q), abs(m_new - m))
            m = blend * m + (1 - blend) * m_new
            q = blend * q + (1 - blend) * q_new

        # Ts.append(T)
        # ms.append(m)

        if verbose:
            print(f"T = {T:.9f}, m = {m:.9f}, q = {q:.9f}")
        if m < 0.01:
            m = m_old
            q = q_old
            T -= deltaT / 2
            deltaT /= 2
        else:
            T += deltaT
    return T


def compute_m_atTd_standard(p, blend=0.25, verbose=False):
    T_init = 0.6
    m_init = 1.0
    q_init = 0.9
    deltaT = 0.01

    m = m_init
    q = q_init
    T = T_init

    # Ts = []
    # ms = []

    while deltaT > 1e-9:
        J0 = 1 / (2 * T)
        err = 1
        m_old = m
        q_old = q
        while err > 1e-9:
            m_new = compute_m_standard(m, q, p, 1 / T, J0)
            q_new = compute_q_standard(m, q, p, 1 / T, J0)

            err = max(abs(q_new - q), abs(m_new - m))
            m = blend * m + (1 - blend) * m_new
            q = blend * q + (1 - blend) * q_new

        if verbose:
            print(f"T = {T:.9f}, m = {m:.9f}, q = {q:.9f}")
        if m < 0.01:
            m = m_old
            q = q_old
            T -= deltaT / 2
            deltaT /= 2
        else:
            T += deltaT
            m_save = m
    return m_save


# with the field

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
