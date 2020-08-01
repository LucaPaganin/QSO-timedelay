import subprocess
import shutil
from pathlib import Path
import os
import sys


def main():
    N_jobs = 100
    workspace = Path('.').absolute()
    outdir = workspace.parent / 'aux/outdir'
    input_file = workspace.parent / 'data/cosmograil/HE0435_Bonvin2016.rdb_.txt'
    main_script_path = workspace / 'curve_sum_dataset.py'
    workdirs = [outdir / f'{i+1:03}' for i in range(N_jobs)]

    outdir.mkdir(exist_ok=True)
    for wd in workdirs:
        wd.mkdir(exist_ok=True)
        dst_main_path = wd / main_script_path.name
        dst_input_path = wd / input_file.name
        shutil.copy(src=main_script_path, dst=dst_main_path)
        shutil.copy(src=input_file, dst=dst_input_path)
        err_file = wd / 'err_file.err'
        out_file = wd / 'out_file.out'

        dst_main_path.chmod(0o755)
        bsub_cmd = f'bsub -P c7 -q long -e {err_file} -o {out_file} {dst_main_path} ' \
                   f'{dst_input_path} {wd}'

        cmd = subprocess.Popen(bsub_cmd, shell=True, stdout=subprocess.PIPE)
        cmd_stdout, cmd_stderr = cmd.communicate()
        cmd_stdout = cmd_stdout.decode('utf-8')
        cmd_stderr = cmd_stderr.decode('utf-8')
        if len(cmd_stderr) > 0:
            raise Exception(cmd_stderr)
        else:
            print(cmd_stdout)







