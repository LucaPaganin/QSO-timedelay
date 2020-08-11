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
    Xdata = []
    max_delay = 20
    max_mag_shift = 2
    true_delays = np.random.random(N_MC) * max_delay
    mag_shifts = np.random.random(N_MC) * max_mag_shift

    for i, (mag_shift, delay) in enumerate(zip(mag_shifts, true_delays)):
        logger.info(f'Realization n° {i + 1}')
        yA, yB = prh_mc_utils.generate_PRH_light_curves(support=t, y=y_data, sigma=sigma_data,
                                                        slope=slope, intercept=intercept,
                                                        delay=delay, mag_shift=mag_shift)
        y_combo = prh_mc_utils.flux_to_mag(prh_mc_utils.mag_to_flux(yA) + prh_mc_utils.mag_to_flux(yB))
        Xdata.append(y_combo)

    logger.info('Done')

    Xdata = np.stack(Xdata)
    tf = time.time()
    logger.info(f'Total time: {tf - t0} seconds')

    ydata = true_delays

    logger.info('Writing output to file')
    file_name = f'HE0435_NMC_{N_MC}_curvesum.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset(name='X', data=Xdata, compression='gzip', compression_opts=9)
    hf.create_dataset(name='y', data=ydata, compression='gzip', compression_opts=9)
    hf.close()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)
