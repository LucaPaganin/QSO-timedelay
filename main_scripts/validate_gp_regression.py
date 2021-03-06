import h5py
import numpy as np
import pandas as pd
import time
import sys
import logging

sys.path.insert(0, '..')

from modules import regression as rg
from modules.utils import configure_logger

logger = logging.getLogger()


def main():
    logfile = 'validate_GP_regression.log'
    configure_logger(logger, logfile)
    data = h5py.File('../aux/RXJ1131_random_delays_0_30.h5', 'r')
    t_dom = data['t_domain'][()]
    # A = data['original_data']['A'][()]
    errA = data['original_data']['errA'][()]
    realization_keys = [k for k in data.keys() if k.startswith('realization')]
    true_delays = []
    estimated_delays = []
    for k in realization_keys:
        y1 = data[k]['y'][()]
        y2 = data[k]['y_delayed'][()]
        true_delay = data[k]['delay'][()]
        logger.info(f'True delay: {true_delay}')

        t0 = time.time()
        estimated_delay = rg.estimate_delay(t_dom, y1, y2, errA, errA)
        tf = time.time()
        logger.info(f'Estimated delay: {estimated_delay}')
        logger.info(f'Elapsed time {tf-t0} s')

        true_delays.append(true_delay)
        estimated_delays.append(estimated_delay)

    with open('../aux/results.txt', 'w') as f:
        f.write('true_delay \t estimated_delay\n')
        for td, ed in zip(true_delays, estimated_delays):
            f.write(f'{td:.2f} \t {ed:.2f}\n')


if __name__ == '__main__':
    main()
