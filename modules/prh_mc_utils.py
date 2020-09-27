import numpy as np
from scipy import stats
import pandas as pd
import h5py
from typing import Dict
from pathlib import Path
from .utils import hdf5_opts


def power_law_sf(tau, slope, intercept):
    return 10 ** intercept * tau ** slope


def exp_sf(tau, V0, dt0):
    return V0 * (1 - np.exp(-tau / dt0))


def mag_to_flux(mag):
    return 10 ** (-mag / 2.5)


def flux_to_mag(flux):
    return -2.5 * np.log10(flux)


def mags_to_fluxsum_mag(mag1, mag2, magerr1, magerr2):
    y = flux_to_mag(mag_to_flux(mag1) + mag_to_flux(mag2))
    err_y = fluxsum_mag_error(mag1, mag2, magerr1, magerr2)
    return y, err_y


def mags_to_fluxsum(mag1, mag2, magerr1, magerr2):
    f1 = mag_to_flux(mag1) 
    f2 = mag_to_flux(mag2)
    y = f1 + f2
    err_y = np.log(10)/2.5 * f1 * magerr1 + np.log(10)/2.5 * f2 * magerr2
    return y, err_y


def fluxsum_mag_error(mag1, mag2, magerr1, magerr2):
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


def generate_PRH_light_curves(support, y, sigma, slope, intercept, delay):
    N = len(support)
    t_doubled = np.concatenate([support, support - delay])
    err_doubled = np.concatenate([sigma, sigma])
    tau_doubled = compute_lags_matrix(t_doubled)
    s2 = ((y-sigma)**2).mean()
    C = s2 - power_law_sf(tau_doubled, slope, intercept)
    C += 1e-10 * np.eye(2 * N)
    L = np.linalg.cholesky(C)
    y_out = L @ np.random.normal(0, 1, 2 * N) + err_doubled @ np.random.normal(0, 1, 2 * N)

    yA = y_out[:N]
    yB = y_out[N:]
    yA -= yA.mean()
    yB -= yB.mean()

    return yA, yB


def create_qso_base_file(qso_dict: Dict[str, np.ndarray] = None,
                         gp_dict: Dict[str, np.ndarray] = None,
                         sf_dict: Dict[str, np.ndarray] = None,
                         outfile: Path = None):
    hf = h5py.File(outfile, 'w')
    base_grp = hf.create_group('qso_base_data')
    orig_grp = base_grp.create_group('original_data')
    for key in qso_dict:
        if key == 't':
            orig_grp.create_dataset(name='t', data=qso_dict['t'], **hdf5_opts)
        else:
            grp = orig_grp.create_group(key)
            for subkey in qso_dict[key]:
                grp.create_dataset(name=subkey, data=qso_dict[key][subkey], **hdf5_opts)
    gp_grp = base_grp.create_group('fluxsum_gp_interpolation')
    for key in gp_dict:
        gp_grp.create_dataset(name=key, data=gp_dict[key], **hdf5_opts)
    sf_grp = base_grp.create_group('structure_function')
    for key in sf_dict:
        if isinstance(sf_dict[key], np.ndarray):
            sf_grp.create_dataset(name=key, data=sf_dict[key], **hdf5_opts)
        else:
            sf_grp.create_dataset(name=key, data=sf_dict[key])
    hf.close()


