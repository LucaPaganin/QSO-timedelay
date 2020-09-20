import h5py
import numpy as np
from pathlib import Path
import sys
import logging

logger = logging.getLogger()

try:
    from modules import utils
except ImportError:
    sys.path.insert(0, '..')
    from modules import utils


def main(*args):
    utils.configure_logger(logger, 'log_build_dataset.log')
    input_dir = Path(args[0])
    output_file = Path(args[1])
    files = list(input_dir.glob("*/*.h5"))
    X_data = []
    y_data = []

    for file in files:
        with h5py.File(file, 'r') as hf:
            X_data.append(hf['X'][()])
            y_data.append(hf['y'][()])

    X_data = np.stack(X_data)
    y_data = np.stack(y_data)

    with h5py.File(output_file, 'a') as out_hf:
        tr_grp = out_hf.create_group('training_data')
        tr_grp.create_dataset('X', X_data, **utils.hdf5_opts)
        tr_grp.create_dataset('y', y_data, **utils.hdf5_opts)


if __name__ == '__main__':
    main(*sys.argv[1:])
