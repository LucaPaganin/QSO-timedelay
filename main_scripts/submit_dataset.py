import subprocess
import shutil
from pathlib import Path
import datetime
import sys


def main(*args):
    N_jobs = 100
    workspace = Path('.').absolute()
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    venv_path = workspace.parent / 'qso_timedelay_venv'
    main_script_path = workspace / 'curve_sum_dataset.py'
    modules_path = Path('..').resolve()
    config_file = workspace / 'dataset_config.json'

    input_file = Path(args[0])
    qso_id = input_file.stem
    outdir = workspace.parent.parent / f'{qso_id}_dataset_{now}'
    workdirs = [outdir / f'{i+1:03}' for i in range(N_jobs)]

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
        print(proc.stdout)


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)
