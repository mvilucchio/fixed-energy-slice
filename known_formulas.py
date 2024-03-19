import numpy as np
from numba import njit
from math import sqrt, log


def get_Tk_Td(p, model="ising"):
    if model == "ising":
        if p == 3:
            T_kauz, T_dyn = 0.651385, 0.6815
        elif p == 4:
            T_kauz, T_dyn = 0.61688, 0.6784
        elif p == 5:
            T_kauz, T_dyn = 0.60695, 0.7001
        elif p == 10:
            T_kauz, T_dyn = 0.6005, 0.838
        elif p == 20:
            T_kauz, T_dyn = 0.5/np.sqrt(np.log(2)), 1.0615
        else:
            raise ValueError("p must be 3, 4, 5 or 10")
    elif model == "spherical":
        if p == 3:
            T_kauz, T_dyn = 0.586, 0.611
        elif p == 4:
            T_kauz, T_dyn = 0.502, 0.544
        elif p == 5:
            T_kauz, T_dyn = 0.461, 0.511
        elif p == 10:
            T_kauz, T_dyn = 0.382, 0.462
        else:
            raise ValueError("p must be 3, 4, 5 or 10")
    else:
        raise ValueError("model must be 'ising' or 'spherical'")

    return T_kauz, T_dyn


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
            np.tanh(β * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2)))
            ** 2
        )
    )


@njit()
def compute_m_standard(m, q, p, β, J0):
    return np.sum(
        weights
        * np.tanh(β * (p * J0 * m ** (p - 1) + roots * np.sqrt(p * q ** (p - 1) / 2)))
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

@njit()
def compute_energy_standard(m, q, p, β, J0):
    return -J0 * m**p - 0.5 * β * (1 - q**p)

def compute_Td_standard(p,blend=0.25,verbose=False):
    T_init = 0.6
    m_init = 1.
    q_init = 0.9
    deltaT = 0.01

    m = m_init
    q = q_init
    T = T_init

    #Ts = []
    #ms = []

    while (deltaT > 1e-8):
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

        #Ts.append(T)
        #ms.append(m)

        if verbose: print(f"T = {T:.9f}, m = {m:.9f}, q = {q:.9f}")
        if (m < 0.01):
            m = m_old
            q = q_old
            T -= deltaT/2
            deltaT /= 2
        else:  T += deltaT
    return T

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


@njit()
def g(q, beta, p):
    return 0.5 * beta**2 * q**p


@njit()
def D_g(q, beta, p):
    return 0.5 * beta**2 * p * q ** (p - 1)


@njit()
def h(q, beta, p):
    return beta**2 * q**p


@njit()
def D_h(q, beta, p):
    return beta**2 * p * q ** (p - 1)


def iterate_single_spherical(beta, p, q_init=0.8, m_init=0.8, tol=1e-6, blend=0.8):
    q = q_init
    m = m_init
    err = tol + 1.0
    iter = 0

    while (tol < err and iter < 10_000) or iter < 30:
        A_new = D_g(q, beta, p)
        B_new = 0.5 * D_h(m, beta, p)

        C_new = A_new + B_new**2

        q_new = 0.5 * (2 + 1 / C_new - sqrt(1 + 4 * C_new) / C_new)
        m_new = B_new * (1 - q_new)

        err = max([abs(q - q_new), abs(m - m_new)])
        q = blend * q_new + (1 - blend) * q
        m = blend * m_new + (1 - blend) * m
        iter += 1

    return q, m


@njit()
def free_energy_FP_spherical(m, q, p, beta):
    return -(
        1
        + log(2 * np.pi)
        + g(1, beta, p)
        - g(q, beta, p)
        + log(1 - q)
        + (q - m**2) / (1 - q)
        + h(m, beta, p)
    ) / (2 * beta)

