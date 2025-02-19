import numpy as np
from numba import njit
from math import isclose, sqrt, log
from scipy.optimize import root
from typing import Iterable


def Td_spherical(p):
    return np.exp(
        0.5
        * (np.log(p) + (p - 2) * np.log(p - 2) - (p - 1) * np.log(p - 1) - np.log(2))
    )


def Tkauz_spherical(p):
    y = root(
        lambda x: 2 / p + 2 * x * (1 - x + np.log(x)) / ((1 - x) ** 2), 1e-9
    ).x[0]
    return y * (1 - y) ** (0.5 * p - 1) * np.sqrt(p / (2 * y))


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
            T_kauz, T_dyn = 0.5 / np.sqrt(np.log(2)), 1.0615
        else:
            raise ValueError("p must be 3, 4, 5 or 10")
    elif model == "spherical":
        return Tkauz_spherical(p), Td_spherical(p)
    else:
        raise ValueError("model must be 'ising' or 'spherical'")

    return T_kauz, T_dyn


def legend_name_Tk_Td_e(e, T_kauz, T_dyn):
    legend_name = f"{e:.3f}"
    if isclose(e, -1 / (2 * T_kauz), rel_tol=1e-6):
        legend_name = r"$- 1/2T_k$"
    if isclose(e, -1 / (2 * T_dyn), rel_tol=1e-6):
        legend_name = r"$ - 1/2T_d$"

    return legend_name


def legend_name_Tk_Td_T(T, T_kauz, T_dyn):
    legend_name = f"{T:.3f}"
    if isclose(T, T_kauz, rel_tol=1e-6):
        legend_name = r"$T_k$"
    if isclose(T, T_dyn, rel_tol=1e-6):
        legend_name = r"$T_d$"

    return legend_name


def multiple_empty_arrays(shape, n):
    return [np.empty(shape) for _ in range(n)]


def observables(list_names: Iterable[str], shape) -> dict:
    return {name: np.empty(shape) for name in list_names}


def get_data_folder_name(type: str):
    return "../data/" + type + "/"


def get_file_name_T_sweep_m(
    type: str, p: int, T_planting: float, T: float, m_range: Iterable[float]
) -> str:
    return f"{type}_sweep_m_p{p}_T{T:.5f}_Tplanting{T_planting:.5f}_m{m_range[0]:.5f}_{m_range[-1]:.5f}.pkl"


def get_file_name_sweep_T(
    type: str, p: int, T_planting: float, T_range: Iterable[float], m0: float, q0: float
) -> str:
    return f"{type}_sweep_T_p{p}_Tplanting{T_planting:.5f}_T{T_range[0]:.5f}_{T_range[-1]:.5f}_m0{m0:.5f}_q0{q0:.5f}.pkl"

def get_file_name_until_fail(
    type: str, p: int, T_planting: float, T_init:float, deltaT: float, m0: float, q0: float
) -> str:
    return f"{type}_until_fail_p{p}_Tplanting{T_planting:.5f}_Tinit{T_init:.5f}_deltaT{deltaT:.5f}_m0{m0:.5f}_q0{q0:.5f}.pkl"

def get_file_name_Tdplus_Tk(
    type: str, deltaT: float, m0: float, q0: float, p_init: int, p_end: int, n_p: int
) -> str:
    return f"{type}_Tdplus_Tk_deltaT{deltaT:.5f}_m0{m0:.5f}_q0{q0:.5f}_p{p_init}_{p_end}_{n_p}.pkl"