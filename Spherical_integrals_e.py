import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p, exp, pi, sqrt


@vectorize()
def beta_q_e(q, m, e, p):
    return 2*exp( log(-e * (1 - m ** p)) - log1p(-q**p) )

@njit()
def compute_q_FP(m, q, p, e):
    beta = -2 * e * (1 - m ** p) / (1 - q ** p)
    return (
        m ** 2 +
        beta ** 2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1) )
    )

# @njit()
def fixed_points_q(m, e, p, blend=0.01, tol=1e-9, q_init=0.9):
    err = 1e10
    q = q_init
    iter = 0
    while err > 1e1 * tol:
        iter +=1
        q_new = compute_q_FP(m, q, p, e)
        if (q_new >= 1):
            print(f"q_new = {q_new}")

        err = abs(q_new - q)
        q = blend * q + (1 - blend) * q_new
        if (iter > 10_000):
            raise ValueError('Fixed point iteration did not converge')
            #h = np.NaN
            #q = np.NaN

    return q

@njit()
def f_FP(q, m, p, e):
    beta = -2 * e * (1 - m ** p) / (1 - q ** p)
    J0 = -e
    return (
        0.5 * beta**2 * (1 - q**p) #switch to 1
        #+ 2 * beta * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * beta * J0 * m**p
        + 1 + log(2*pi)
    )/ (-2 * beta)

def s_FP(q, m, p, e):
    return beta_q_e(q, m, e, p)*( e - f_FP(q, m, p, e) )

@vectorize()
def annealed_entropy(m, e, p):
    H = (1 + log(-2*(-1 + m**2)*pi))/2.
    return H - e**2 * (1 - m**p) ** 2

################################# J0 = beta_pl / 2 #################################

@njit()
def e_J0(q, m, p, beta, J0):
    return J0 * m ** p - 0.5 * beta * (1 - q ** p)

@njit()
def compute_m_FP_J0(m, q, p, beta, J0):
    return (
        beta * (1 - q) * J0 * p * m ** (p - 1)
    )

@njit()
def compute_q_FP_J0(m, q, p, beta):
    return (
        m ** 2 +
        beta ** 2 * (1 - q) ** 2 * (0.5 * p * q ** (p - 1) )
    )

@njit()
def newm(thism, thisq, p, beta):
    return sqrt(thisq - (1 - thisq) * (1 - thisq) * (0.5 * beta**2 * p * pow(thisq, p - 1)))


@njit()
def newq(thism, thisq, p, beta, J0):
    return 1 - thism / (beta * p * J0 * pow(thism, p - 1))


def fixed_points_q_J0(m, beta, p, blend=0.01, tol=1e-9, q_init=0.9):
    err = 1e10
    q = q_init
    iter = 0
    while err > 1e1 * tol:
        iter +=1
        q_new = compute_q_FP_J0(m, q, p, beta)
        if (q_new >= 1):
            print(f"q_new = {q_new}")

        err = abs(q_new - q)
        q = blend * q + (1 - blend) * q_new
        if (iter > 10_000):
            raise ValueError('Fixed point iteration did not converge')
            #h = np.NaN
            #q = np.NaN

    return q

@njit()
def f_FP_J0(q, m, p, beta, J0):
    #beta = -2 * e * (1 - m ** p) / (1 - q ** p)
    return (
        0.5 * beta**2 * (1 - q**p) #switch to 1
        #+ 2 * beta * h * m
        + log1p(-q)
        + (q - m**2) / (1 - q)
        + 2 * beta * J0 * m**p
        + 1 + log(2*pi)
    )/ (-2 * beta)

def s_FP_J0(q, m, p, beta, J0):
    return beta *( e_J0(q, m, p, beta, J0) - f_FP_J0(q, m, p, beta, J0) )

def Hes_eig_J0(q, m, p, beta, J0):
    return (
        -0.125*(4*m**4*q**2 + 4*beta*J0*m**p*(-1 + p)*p*(-1 + q)**3*q**2 - 
        m**2*(beta**2*(-1 + p)*p*(-1 + q)**3*q**p - 2*q**2*(1 + q*(-5 + 2*q))) + 
        np.sqrt(16*m**2*(-1 + q)**2*q**2*
           (m**2*(beta**2*(-1 + p)*p*(-1 + q)**3*q**p + 2*q**2*(1 + q)) + 
             beta*J0*m**p*(-1 + p)*p*(-1 + q)*
              (beta**2*(-1 + p)*p*(-1 + q)**3*q**p + 2*q**2*(1 - 2*m**2 + q))) + 
          (4*beta*J0*m**p*(-1 + p)*p*(-1 + q)**3*q**2 + 
             m**2*(-(beta**2*(-1 + p)*p*(-1 + q)**3*q**p) + 
                2*q**2*(1 + 2*m**2 + q*(-5 + 2*q))))**2))/(beta*m**2*(-1 + q)**3*q**2))