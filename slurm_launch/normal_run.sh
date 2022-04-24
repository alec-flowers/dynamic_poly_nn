#!/bin/bash

#SBATCH --job-name=poly_run
#SBATCH --chdir /scratch/flowers/
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --ntasks 8
#SBATCH --mem-per-cpu=16G

echo STARTING AT `date`

module purge
module load gcc/8.4.0 python/3.7.7
source /home/flowers/venvs/fidis_dpnn_venv/bin/activate

INPUT="/home/flowers/dynamic_poly_nn/configs/spectral_runs"
i=1
for file in $(ls ${INPUT})
do
  echo "Triggering processing for ${file}"
  COMMAND="python3 /home/flowers/dynamic_poly_nn/run.py --config_name spectral_runs/${file}"
  OUTPUT_FILE="slurm-${SLURM_JOB_ID}.${i}.${file}.out"

  srun --exclusive --ntasks 1 ${COMMAND} > ${OUTPUT_FILE} 2>&1 &

  # Increase counter
  let i+=1

  # Sleep for longer to allow only 1 to download data.
  if [ ${i} -eq 1 ]; then
    sleep 20
  fi
  sleep 1
done

	# Important!
# Makes the main job script wait for all the background srun commands
wait

echo FINISHED at `date`
deactivate