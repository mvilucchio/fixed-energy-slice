import numpy as np
from numba import njit, vectorize
from root_finding import brent_root_finder
from scipy.optimize import root_scalar
from math import log, log1p

# gaussian integration
r, w = np.polynomial.hermite.hermgauss(99)

roots = np.sqrt(2) * np.array(r)
weights = np.array(w) / np.sqrt(np.pi)


