#!/bin/bash

#SBATCH --job-name=poly_run
#SBATCH --partition=debug
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --chdir /scratch/izar/flowers/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 00:20:00

echo STARTING AT `date`

module purge
module load gcc/8.4.0 python/3.7.7
source /home/flowers/venvs/dpolynn_venv/bin/activate

python3 /home/flowers/dynamic_poly_nn/run.py --config_name "ensemble_run/ncp_low.yml"

deactivate

echo FINISHED at `date`