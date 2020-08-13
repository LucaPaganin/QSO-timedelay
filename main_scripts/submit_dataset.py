import subprocess
import shutil
from pathlib import Path
import datetime
import os
import sys
from typing import Union


def main():
    N_jobs = 100
    workspace = Path('.').absolute()
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    venv_path = workspace.parent / 'qso_timedelay_venv'
    input_file = workspace.parent / 'data/cnn_base_data/HE0435_Aflux_plus_Bflux.h5'
    main_script_path = workspace / 'curve_sum_dataset.py'
    modules_path = Path('..')
    qso_id = input_file.name.split('_')[0]
    outdir = workspace.parent / f'aux/{qso_id}_outdir_{now}'
    workdirs = [outdir / f'{i+1:03}' for i in range(N_jobs)]

    outdir.mkdir(exist_ok=True)
    for wd in workdirs:
        wd.mkdir(exist_ok=True)
        dst_main_path = wd / main_script_path.name
        dst_input_path = wd / input_file.name
        shutil.copy(src=main_script_path, dst=dst_main_path)
        dst_main_path.chmod(0o755)
        shutil.copy(src=input_file, dst=dst_input_path)

        sh_file = wd / 'job.sh'
        with open(sh_file, 'w') as sh:
            sh.write('#!/bin/bash\n\n')
            sh.write(f'source {venv_path}/bin/activate\n')
            sh.write(f'{dst_main_path} {dst_input_path} {wd} {modules_path}\n')
        sh_file.chmod(0o755)

        err_file = wd / 'err_file.err'
        out_file = wd / 'out_file.out'
        bsub_cmd = f'bsub -P c7 -q long -n 8 -R"span[hosts=1]" -e {err_file} -o {out_file} {sh_file}'
        cmd = subprocess.Popen(bsub_cmd, shell=True, stdout=subprocess.PIPE)
        cmd_stdout, cmd_stderr = cmd.communicate()
        cmd_stdout = cmd_stdout.decode('utf-8') if cmd_stdout is not None else ''
        cmd_stderr = cmd_stderr.decode('utf-8') if cmd_stderr is not None else ''
        if len(cmd_stderr) > 0:
            raise Exception(cmd_stderr)
        else:
            print(cmd_stdout)


if __name__ == '__main__':
    main()
