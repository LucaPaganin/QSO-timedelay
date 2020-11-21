import subprocess
from pathlib import Path
import datetime
import shutil
import sys
import h5py
import time
import argparse
import numpy as np
import logging

sys.path.insert(0, '..')

from modules import utils

logger = logging.getLogger()


def program_opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to input file")
    return parser


def main(args_dict):
    N_jobs = 100
    workspace = Path('.').absolute()
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    venv_path = workspace.parent / 'qso_timedelay_venv'
    main_script_path = workspace / 'curve_sum_dataset.py'
    modules_path = Path('..').resolve()
    config_file = workspace / 'dataset_config.json'

    input_file = Path(args_dict['input_file']).resolve()
    qso_id = input_file.stem
    outdir = workspace.parent.parent / f'outputs/{qso_id}_dataset_{now}'
    workdirs = [outdir / f'{i+1:03}' for i in range(N_jobs)]

    logfile = f'log_dataset_{qso_id}_{now}.log'
    utils.configure_logger(logger, logfile)

    outdir.mkdir(exist_ok=True)
    for wd in workdirs:
        wd.mkdir(exist_ok=True)

        sh_file = wd / 'job.sh'
        with open(sh_file, 'w') as sh:
            sh.write('#!/bin/bash\n\n')
            sh.write(f'source {venv_path}/bin/activate\n')
            sh.write(f'python3 {main_script_path} {input_file} {wd} {modules_path} {config_file}\n')
        sh_file.chmod(0o755)

        err_file = wd / 'err_file.err'
        out_file = wd / 'out_file.out'
        bsub_cmd = f'bsub -P c7 -q long -n 4 -R"span[hosts=1]" -e {err_file} -o {out_file} {sh_file}'
        proc = subprocess.run(bsub_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info(proc.stdout.decode('utf-8'))

    check_time = 100
    n_jobs = len(workdirs)
    n_outputs = 0
    logger.info('Waiting for jobs execution')
    t0 = time.time()
    while n_outputs != n_jobs:
        logger.info(f'Waiting for {check_time} s...')
        time.sleep(check_time)
        t1 = time.time()
        silent = int(t1-t0) % 1800 != 0 
        n_outputs = len(list(outdir.glob('*/*.h5')))
        if not silent:
            logger.info(f'{n_outputs}/{n_jobs} completed')
    
    tf = time.time()
    logger.info(f'Total elapsed time: {tf-t0:.2f} s')

    output_files = list(outdir.glob('*/*.h5'))
    out_file = outdir / input_file.name
    shutil.copy(input_file, out_file)

    X_data = []
    y_data = []

    for file in output_files:
        logger.info(f'Reading file {file}')
        with h5py.File(file, 'r') as hf:
            X_data.append(hf['X'][()])
            y_data.append(hf['y'][()])

    X_data = np.concatenate(X_data)
    y_data = np.concatenate(y_data)

    with h5py.File(out_file, 'a') as out_hf:
        if 'training_data' in out_hf:
            raise Exception(f'File already populated with training data')
        else:
            tr_grp = out_hf.create_group('training_data')
            tr_grp.create_dataset(name='X', data=X_data, **utils.hdf5_opts)
            tr_grp.create_dataset(name='y', data=y_data, **utils.hdf5_opts)

    logger.info('Deleting partial output files')
    for file in output_files:
        logger.info(f'Deleting file {file}')
        file.unlink()


if __name__ == '__main__':
    arguments_dict = vars(program_opts().parse_args())
    main(arguments_dict)
