import matplotlib.pyplot as plt
import numpy as np
from math import isclose


def plot_dashed_instable(ax, xs, ys, stable_idxs, color="black", legend_name=""):
    if np.all(stable_idxs):
        ax.plot(xs, ys[:], "-", label=legend_name, color=color)
    else:
        i_start, i_end = np.where(np.diff(np.r_[True, stable_idxs, True]) == True)[0]

        ax.plot(xs[:i_start], ys[:i_start], "-", label=legend_name, color=color)
        ax.plot(xs[i_end:], ys[i_end:], "-", color=color)

        ax.plot(xs[i_start:i_end], ys[i_start:i_end], "--", color=color, alpha=0.75)


def legend_name_Tk_Td(e, T_kauz, T_dyn):
    legend_name = f"{e:.3f}"
    if isclose(e, -1 / (2 * T_kauz),rel_tol=1e-6):
        legend_name = r"$- 1/2T_k$"
    if isclose(e, -1 / (2 * T_dyn),rel_tol=1e-6):
        legend_name = r"$ - 1/2T_d$"

    return legend_name


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
