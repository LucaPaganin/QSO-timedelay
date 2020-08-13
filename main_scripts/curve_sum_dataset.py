#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import time
from pathlib import Path
import os
import sys
import h5py
import logging

logger = logging.getLogger()


def main(*args):
    N_MC = 1000
    file_path = Path(args[0])
    workdir = Path(args[1])
    modules_path = Path(args[2])

    sys.path.insert(0, str(modules_path))
    from modules import utils
    from modules import prh_mc_utils

    os.chdir(workdir)

    logfile = str(workdir / 'logfile_curvesum.log')
    
    utils.configure_logger(logger, logfile)
    logger.info('Reading data')
    data = h5py.File(file_path, 'r')
    t = data['t'][()]
    y_data = data['y'][()]
    sigma_data = data['err_y'][()]
    slope = data['slope'][()]
    intercept = data['intercept'][()]

    logger.info('Starting MC')
    t0 = time.time()
    min_delay = -30
    max_delay = +30
    max_mag_shift = 2
    true_delays = np.random.random(N_MC) * (max_delay - min_delay) + min_delay
    mag_shifts = np.random.random(N_MC) * max_mag_shift
    shrink_factors = np.random.random(N_MC) * (1 - 0.8) + 0.8

    Xdata = []
    sf_mc_data = []

    for i, (mag_shift, delay, shf) in enumerate(zip(mag_shifts, true_delays, shrink_factors)):
        logger.info(f'Realization n° {i + 1}')
        logger.info('Generating curve combo')
        yA, yB = prh_mc_utils.generate_PRH_light_curves(support=t, y=y_data, sigma=sigma_data,
                                                        slope=slope, intercept=intercept,
                                                        delay=delay, mag_shift=mag_shift, shrink_factor=shf)
        y_combo = prh_mc_utils.mag_flux_sum(yA, yB)
        logger.info('Estimating structure function')
        _, v_mc = prh_mc_utils.estimate_structure_func_from_data(t, y_combo, sigma_data)
        Xdata.append(y_combo)
        sf_mc_data.append(v_mc)

    logger.info('Done')

    Xdata = np.asarray(Xdata)
    sf_mc_data = np.asarray(sf_mc_data)
    sf_mc_mean = np.mean(sf_mc_data, axis=0)
    tf = time.time()
    logger.info(f'Total time: {tf - t0} seconds')

    ydata = true_delays

    logger.info('Writing output to file')
    file_name = f'HE0435_NMC_{N_MC}_curvesum.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset(name='X', data=Xdata, compression='gzip', compression_opts=9)
    hf.create_dataset(name='y', data=ydata, compression='gzip', compression_opts=9)
    hf.create_dataset(name='sf_mc_mean', data=sf_mc_mean, compression='gzip', compression_opts=9)
    hf.close()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)
