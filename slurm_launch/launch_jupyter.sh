#!/bin/bash -l
#SBATCH --chdir /home/flowers/dynamic_poly_nn
#SBATCH --job-name=ipython-trial
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output jupyter-log-%J.out

module purge
module load gcc/8.4.0 python/3.7.7
source /home/flowers/venvs/dpolynn_venv/bin/activate

ipnport=$(shuf -i8000-9999 -n1)

jupyter-notebook --no-browser --port=${ipnport} --ip=$(hostname -i)