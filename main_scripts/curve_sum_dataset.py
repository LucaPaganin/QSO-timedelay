#!/usr/bin/env python3
# coding: utf-8
import json
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
    config_file = Path(args[3])
    with open(config_file, 'r') as jsf:
        config = json.load(jsf)

    qso_id = file_path.stem

    sys.path.insert(0, str(modules_path))
    from modules import utils
    from modules import prh_mc_utils

    os.chdir(workdir)

    logfile = str(workdir / 'logfile_curvesum.log')
    
    utils.configure_logger(logger, logfile)
    logger.info('Reading data')
    data = h5py.File(file_path, 'r')
    gp_grp = data['qso_base_data']['fluxsum_gp_interpolation']
    t = gp_grp['t'][()]
    y_data = gp_grp['y_pred'][()]
    sigma_data = gp_grp['sigma_pred'][()]
    sf_grp = data['qso_base_data']['structure_function']
    slope = sf_grp['slope'][()]
    intercept = sf_grp['intercept'][()]

    logger.info('Starting MC')
    t0 = time.time()
    min_delay = 0
    max_delay = float(config['max_delay'])
    max_mag_shift = float(config['max_mag_shift'])
    true_delays = np.random.random(N_MC) * (max_delay - min_delay) + min_delay
    mag_shifts = np.random.random(N_MC) * max_mag_shift

    Xdata = []

    for i, (mag_shift, delay) in enumerate(zip(mag_shifts, true_delays)):
        logger.info(f'Realization n° {i + 1}')
        logger.info('Generating curve combo')
        yA, yB = prh_mc_utils.generate_PRH_light_curves(support=t, y=y_data, sigma=sigma_data,
                                                        slope=slope, intercept=intercept,
                                                        delay=delay, mag_shift=mag_shift)
        y_combo = prh_mc_utils.mag_flux_sum(yA, yB)
        Xdata.append(y_combo)

    logger.info('Done')

    Xdata = np.asarray(Xdata)
    tf = time.time()
    logger.info(f'Total time: {tf - t0} seconds')

    ydata = true_delays

    logger.info('Writing output to file')
    file_name = f'{qso_id}_NMC_{N_MC}.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset(name='X', data=Xdata, **utils.hdf5_opts)
    hf.create_dataset(name='y', data=ydata, **utils.hdf5_opts)
    hf.close()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)
