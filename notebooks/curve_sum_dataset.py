#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import itertools
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special
from scipy import integrate
from scipy import interpolate
from scipy import linalg
from scipy import signal
from scipy.optimize import curve_fit
import time
from pathlib import Path
import os
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['text.usetex'] = True

import sys
import mpld3

from typing import Tuple

def power_law_sf(tau, slope, intercept):
    return 10**intercept * tau**slope


def exp_sf(tau, V0, dt0):
    return V0*(1-np.exp(-tau/dt0))


def spline_sf(tau, v):
    spline = interpolate.UnivariateSpline(tau, v, s=1e-6, k=3)
    return spline


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

def generate_PRH_light_curves(support, y, sigma, slope, intercept, delay):
    N = len(support)
    t_doubled = np.concatenate([support, support - delay])
    err_doubled = np.concatenate([sigma, sigma])
    tau_doubled = compute_lags_matrix(t_doubled)
    s2 = (y**2).mean()
    C = s2 - power_law_sf(tau_doubled, slope, intercept)
    C += 1e-10*np.eye(2*N)
    L = np.linalg.cholesky(C)
    y = L @ np.random.normal(0, 1, 2*N) + err_doubled @ np.random.normal(0, 1, 2*N)
    
    yA = y[:N] - y[:N].mean()
    yB = y[N:] - y[N:].mean()
    
    return yA, yB


file_path = Path('../data/cosmograil/HE0435_Bonvin2016.rdb_.txt')

qso_id = file_path.name.split('_')[0]
data = pd.read_table(file_path)

t = data['mhjd'].to_numpy(dtype=np.float64)
A = data['mag_A'].to_numpy(dtype=np.float64)
errA = data['magerr_A'].to_numpy(dtype=np.float64)


kernel = ConstantKernel(2, (1e-3, 1e2)) * Matern(length_scale=200.0, length_scale_bounds=(1, 300), nu=1.5)

gp = GaussianProcessRegressor(kernel=kernel, alpha=errA**2, n_restarts_optimizer=10, 
                              optimizer='fmin_l_bfgs_b', normalize_y=True)

gp.fit(np.expand_dims(t,1), A)

N = 2000
support = np.linspace(t[0], t[-1], N)

A_pred, sigmaA = gp.predict(np.expand_dims(support, 1), return_std=True)

tau, v = estimate_structure_func_from_data(support, A_pred, sigmaA)

beg_off = int(0.10*len(tau))
cut_off = int(0.60*len(tau))

x = tau[beg_off:cut_off]
y = v[beg_off:cut_off]

pars = stats.linregress(np.log10(x), np.log10(y))
slope = pars.slope
intercept = pars.intercept

with open('logfile_curvesum.log', 'w') as logfile:
    t0 = time.time()
    N_MC = int(1e5)
    Xdata = []
    true_delays = np.random.random(N_MC)*100

    for i, delay in enumerate(true_delays):
        logfile.write(f'Realization n° {i+1}\n')
        yA, yB = generate_PRH_light_curves(support, y, sigma, slope, intercept, delay)
        Xdata.append(yA+yB)

    logfile.write('Done\n')

    Xdata = np.stack(Xdata)
    tf = time.time()
    
    hf = h5py.File(f'HE0435_NMC_{N_MC}_curvesum.h5', 'w')
    hf.create_dataset(name='X', data=Xdata, compression='gzip', compression_opts=9)
    hf.create_dataset(name='y', data=true_delays, compression='gzip', compression_opts=9)
    hf.close()

