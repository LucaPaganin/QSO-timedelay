import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


def fit_GP_to_lightcurve(t, y, err_y, kernel):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=err_y ** 2,
                                  n_restarts_optimizer=10, optimizer='fmin_l_bfgs_b', normalize_y=True)
    gp.fit(np.expand_dims(t, 1), y)
    return gp


def numeric_derivative(f: np.ndarray, step: float) -> np.ndarray:
    return (f[1:] - f[:-1]) / step


def WAV(f, sigma, step):
    f_prime = numeric_derivative(f, step)
    weights = 2 / (sigma[:-1] + sigma[1:])
    WAV = np.dot(np.abs(f_prime), weights) / weights.sum()
    return WAV


def time_delay_grid_search(y_pred_1: np.ndarray, y_pred_2: np.ndarray,
                           sigma1: np.ndarray, sigma2: np.ndarray,
                           gp_time_step: float,
                           dt_min: float = 0, dt_max: float = 100) -> float:
    shift = np.arange(int(dt_min/gp_time_step), int(dt_max/gp_time_step), 1)
    win = len(shift)
    WAV_values = []
    for i in shift:
        diff = y_pred_1[win:-win] - y_pred_2[win + i:-win + i]
        sigma_diff = sigma1[win:-win] + sigma2[win + i:-win + i]
        WAV_values.append(WAV(diff, sigma_diff, gp_time_step))
    return np.argmin(WAV_values) * gp_time_step

