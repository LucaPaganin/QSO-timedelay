from scipy import stats
import numpy as np
from typing import Tuple


def compute_lags_matrix(t) -> np.ndarray:
    N = len(t)
    t_row_repeated = np.repeat(t[np.newaxis, :], N, axis=0)
    t_col_repeated = np.repeat(t[:, np.newaxis], N, axis=1)
    tau = np.abs(t_row_repeated - t_col_repeated)
    return tau


def estimate_structure_func_from_data(t, y, err_y) -> Tuple[np.ndarray]:
    N = len(t)
    y_row_repeated = np.repeat(y[np.newaxis, :], N, axis=0)
    y_col_repeated = np.repeat(y[:, np.newaxis], N, axis=1)
    err_y_row_repeated = np.repeat(err_y[np.newaxis, :], N, axis=0)
    err_y_col_repeated = np.repeat(err_y[:, np.newaxis], N, axis=1)
    v = (y_row_repeated - y_col_repeated)**2 - (err_y_row_repeated**2 + err_y_col_repeated**2)
    tau = compute_lags_matrix(t)
    tau_v_sorted = np.array([[mytau, myv] for mytau, myv in sorted(zip(tau.ravel(), v.ravel()))])
    tau_vals = tau_v_sorted[:, 0]
    v_vals = tau_v_sorted[:, 1]
    tau_binned = stats.binned_statistic(tau_vals, tau_vals, bins=100)[0]
    v_binned = stats.binned_statistic(tau_vals, v_vals, bins=100)[0]
    return tau_binned, v_binned
