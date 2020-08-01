#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
import time
from pathlib import Path
import os
import sys
import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import Tuple
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import logging

logger = logging.getLogger()

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['text.usetex'] = True


def configure_logger(logger, logfile):
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    # create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)
    # create formatters and add them to the handlers
    logfile_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(logfile_formatter)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    stderr_handler.setFormatter(logging.Formatter('%(message)s'))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


def power_law_sf(tau, slope, intercept):
    return 10**intercept * tau**slope


def exp_sf(tau, V0, dt0):
    return V0*(1-np.exp(-tau/dt0))


def spline_sf(tau, v):
    spline = interpolate.UnivariateSpline(tau, v, s=1e-6, k=3)
    return spline


def compute_lags_matrix(t):
    N = len(t)
    t_row_repeated = np.repeat(t[np.newaxis, :], N, axis=0)
    t_col_repeated = np.repeat(t[:, np.newaxis], N, axis=1)
    tau = np.abs(t_row_repeated - t_col_repeated)
    return tau


def estimate_structure_func_from_data(t, y, err_y):
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


def generate_PRH_light_curves(support, y, sigma, slope, intercept, delay):
    N = len(support)
    t_doubled = np.concatenate([support, support - delay])
    err_doubled = np.concatenate([sigma, sigma])
    tau_doubled = compute_lags_matrix(t_doubled)
    s2 = (y**2).mean()
    C = s2 - power_law_sf(tau_doubled, slope, intercept)
    C += 1e-10*np.eye(2*N)
    L = np.linalg.cholesky(C)
    y = np.dot(L, np.random.normal(0, 1, 2*N)) + np.dot(err_doubled, np.random.normal(0, 1, 2*N))
    
    yA = y[:N] - y[:N].mean()
    yB = y[N:] - y[N:].mean()
    
    return yA, yB


def main(*args):
    N_MC = 1000
    file_path = Path(args[0])
    workdir = Path(args[1])

    os.chdir(workdir)

    logfile = str(workdir / 'logfile_curvesum.log')
    
    configure_logger(logger, logfile)
    logger.info('Reading data')
    data = pd.read_table(file_path)
    t = data['mhjd'].to_numpy(dtype=np.float64)
    A = data['mag_A'].to_numpy(dtype=np.float64)
    errA = data['magerr_A'].to_numpy(dtype=np.float64)

    kernel = ConstantKernel(2, (1e-3, 1e2)) * Matern(length_scale=200.0, length_scale_bounds=(1, 300), nu=1.5)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=errA ** 2, n_restarts_optimizer=10,
                                  optimizer='fmin_l_bfgs_b', normalize_y=True)

    logger.info('Fitting GP to data')
    gp.fit(np.expand_dims(t, 1), A)

    N = 2000
    support = np.linspace(t[0], t[-1], N)

    logger.info('Predicting with GP')
    A_pred, sigmaA = gp.predict(np.expand_dims(support, 1), return_std=True)

    logger.info('Estimating SF')
    tau, v = estimate_structure_func_from_data(support, A_pred, sigmaA)

    beg_off = int(0.10 * len(tau))
    cut_off = int(0.60 * len(tau))

    pars = stats.linregress(np.log10(tau[beg_off:cut_off]), np.log10(v[beg_off:cut_off]))
    slope = pars.slope
    intercept = pars.intercept

    y = A_pred
    sigma = sigmaA

    logger.info('Starting MC')
    t0 = time.time()
    Xdata = []
    true_delays = np.random.random(N_MC) * 100

    for i, delay in enumerate(true_delays):
        logger.info(f'Realization n° {i + 1}')
        yA, yB = generate_PRH_light_curves(support, y, sigma, slope, intercept, delay)
        Xdata.append(yA + yB)

    logger.info('Done')

    Xdata = np.stack(Xdata)
    tf = time.time()
    logger.info(f'Total time: {tf-t0} seconds')

    ydata = true_delays

    logger.info('Writing output to file')
    file_name = f'HE0435_NMC_{N_MC}_curvesum'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset(name='X', data=Xdata, compression='gzip', compression_opts=9)
    hf.create_dataset(name='y', data=ydata, compression='gzip', compression_opts=9)
    hf.close()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)

