import os
import random
import string
import subprocess
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--script', help='set the script to run')
parser.add_argument('--options', help='set the options of the  script to run',
    default='')
parser.add_argument('--pname', help='add something to the name of the experiment',
    default='')
parser.add_argument('--efile', help='path of error log file',
    default='')
parser.add_argument('--name', help='path of error log file',
    default='')
args = parser.parse_args()

def random_id(length=8):
    return ''.join(random.sample(string.ascii_letters + string.digits, length))

TEMPLATE_SERIAL = """#!/bin/sh
#PBS -j eo
#PBS -l nodes=1:ppn=1
# ulimit -s unlimited
#PBS -e {logfile}
#PBS -o {logfile}
#PBS -N {name}
cd $PBS_O_WORKDIR

# echo PATH = $PATH


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dan/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dan/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/dan/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dan/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate photochempy_temp_ev
# echo PATH now = $PATH

python {script} {options}

wait
"""

opt = args.options.replace(' ', '').replace('-', '') if args.options is not\
    None else ''

name = args.name or 'job'
def submit_python_code(script=args.script, cleanup=True):
    # logfile = '../logs/output.log'
    errfile = '/home/dan/models/atmos-wogan/experiments_scripts/logs/' +  args.efile
    base = "submit_{0}".format(random_id())
    with open(base + '.qsub', 'wb') as f:
        job = TEMPLATE_SERIAL.format(script=script, name=name, options=args.options,
                logfile=errfile)
        f.write(job.encode())
    try:
        subprocess.call('qsub < ' + base + '.qsub', shell=True)
    finally:
        if cleanup:
            os.remove(base + '.qsub')

if __name__ == '__main__':
    submit_python_code()
