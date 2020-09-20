import h5py
import numpy as np
from pathlib import Path
import sys
import argparse
import logging

logger = logging.getLogger()

try:
    from modules import utils
except ImportError:
    sys.path.insert(0, '..')
    from modules import utils


def program_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, help="Path to input directory")
    parser.add_argument('--output_file', '-o', required=True, help="Path to output file. If it already exists, "
                                                                   "it will be opened in append mode, in order "
                                                                   "to not overwrite it")
    return parser


def main(args_dict):
    utils.configure_logger(logger, 'log_build_dataset.log')
    input_dir = Path(args_dict['input_dir'])
    output_file = Path(args_dict['output_file'])
    files = list(input_dir.glob("*/*.h5"))
    X_data = []
    y_data = []

    for file in files:
        with h5py.File(file, 'r') as hf:
            X_data.append(hf['X'][()])
            y_data.append(hf['y'][()])

    X_data = np.stack(X_data)
    y_data = np.stack(y_data)

    out_file_mode = 'a' if output_file.exists() else 'w'

    with h5py.File(output_file, out_file_mode) as out_hf:
        tr_grp = out_hf.create_group('training_data')
        tr_grp.create_dataset('X', X_data, **utils.hdf5_opts)
        tr_grp.create_dataset('y', y_data, **utils.hdf5_opts)


if __name__ == '__main__':
    arguments_dict = vars(program_options().parse_args())
    main(arguments_dict)
