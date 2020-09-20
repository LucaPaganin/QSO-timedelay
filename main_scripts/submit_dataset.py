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
    qso_id = input_file.name.split('_')[0]
    outdir = workspace.parent / f'aux/{qso_id}_dataset_{now}'
    workdirs = [outdir / f'{i+1:03}' for i in range(N_jobs)]

    files_to_copy = {
        'main': main_script_path,
        'input': input_file,
        'config': config_file
    }

    outdir.mkdir(exist_ok=True)
    for wd in workdirs:
        wd.mkdir(exist_ok=True)
        dst_dict = {}
        for key, file in files_to_copy.items():
            dst_dict[key] = wd / file.name
            shutil.copy(src=file, dst=dst_dict[key])

        sh_file = wd / 'job.sh'
        with open(sh_file, 'w') as sh:
            sh.write('#!/bin/bash\n\n')
            sh.write(f'source {venv_path}/bin/activate\n')
            sh.write(f'python3 {dst_dict["main"]} {dst_dict["input"]} {wd} {modules_path} {dst_dict["config"]}\n')
        sh_file.chmod(0o755)

        err_file = wd / 'err_file.err'
        out_file = wd / 'out_file.out'
        bsub_cmd = f'bsub -P c7 -q long -n 4 -R"span[hosts=1]" -e {err_file} -o {out_file} {sh_file}'
        cmd = subprocess.Popen(bsub_cmd, shell=True, stdout=subprocess.PIPE)
        cmd_stdout, cmd_stderr = cmd.communicate()
        cmd_stdout = cmd_stdout.decode('utf-8') if cmd_stdout is not None else ''
        cmd_stderr = cmd_stderr.decode('utf-8') if cmd_stderr is not None else ''
        if len(cmd_stderr) > 0:
            raise Exception(cmd_stderr)
        else:
            print(cmd_stdout)


if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(*arguments)
