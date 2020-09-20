import numpy as np
from scipy import stats


def power_law_sf(tau, slope, intercept):
    return 10 ** intercept * tau ** slope


def exp_sf(tau, V0, dt0):
    return V0 * (1 - np.exp(-tau / dt0))


def mag_to_flux(mag):
    return 10 ** (-mag / 2.5)


def flux_to_mag(flux):
    return -2.5 * np.log10(flux)


def mag_flux_sum(mag1, mag2):
    return flux_to_mag(mag_to_flux(mag1) + mag_to_flux(mag2))


def flux_sum_err(mag1, mag2, magerr1, magerr2):
    flux1 = mag_to_flux(mag1)
    flux2 = mag_to_flux(mag2)
    return (flux1*magerr1 + flux2*magerr2)/(flux1+flux2)


def compute_lags_matrix(t):
    N = len(t)
    t_row_repeated = np.repeat(t[np.newaxis, :], N, axis=0)
    t_col_repeated = np.repeat(t[:, np.newaxis], N, axis=1)
    tau = np.abs(t_row_repeated - t_col_repeated)
    return tau


def estimate_structure_func_from_data(t, y, err_y, n_bins=100):
    N = len(t)
    y_row_repeated = np.repeat(y[np.newaxis, :], N, axis=0)
    y_col_repeated = np.repeat(y[:, np.newaxis], N, axis=1)
    err_y_row_repeated = np.repeat(err_y[np.newaxis, :], N, axis=0)
    err_y_col_repeated = np.repeat(err_y[:, np.newaxis], N, axis=1)
    v = (y_row_repeated - y_col_repeated) ** 2 - (err_y_row_repeated ** 2 + err_y_col_repeated ** 2)
    tau = compute_lags_matrix(t)

    tau = tau.ravel()
    v = v.ravel()[np.argsort(tau)]
    tau.sort()
    tau_binned_means = stats.binned_statistic(tau, tau, bins=n_bins)[0]
    v_binned_means = stats.binned_statistic(tau, v, bins=n_bins)[0]
    return tau_binned_means, v_binned_means


def generate_PRH_light_curves(support, y, sigma, slope, intercept, delay, mag_shift):
    N = len(support)
    t_doubled = np.concatenate([support, support - delay])
    err_doubled = np.concatenate([sigma, sigma])
    tau_doubled = compute_lags_matrix(t_doubled)
    s2 = ((y-sigma)**2).mean()
    C = s2 - power_law_sf(tau_doubled, slope, intercept)
    C += 1e-10 * np.eye(2 * N)
    L = np.linalg.cholesky(C)
    y = L @ np.random.normal(0, 1, 2 * N) + err_doubled @ np.random.normal(0, 1, 2 * N)

    yA = y[:N]
    yB = y[N:]
    yA -= yA.mean()
    yB -= yB.mean()
    
    yB += mag_shift

    return yA, yB
