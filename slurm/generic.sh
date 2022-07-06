#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode:  pass a file with first the job tag and second the commands per line

#SBATCH --time=144:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB

set -e # fail fully on first line failure

# Customize this line to point to conda installation
path_to_conda="/data/vision/polina/shared_software/miniconda3"
# huggingface cache
export HF_HOME='/data/vision/polina/scratch/wpq/github/huggingface_cache'

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID
    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Find what was passed to --output_folder
regexp="--output_dir_toplvl(\s+|=)(\S+)"
if [[ $JOB_CMD =~ $regexp ]]
then
    JOB_OUTPUT=${BASH_REMATCH[2]}
else
    echo "Error: did not find a --output_dir_toplvl argument"
    exit 1
fi

source ${path_to_conda}/bin/activate mi
cd /data/vision/polina/scratch/wpq/github/mi/local_mi_peiqi

echo srun $JOB_CMD --gpu_id=$SLURM_JOB_GPUS
srun $JOB_CMD --gpu_id=$SLURM_JOB_GPUS

# If successfully ran, move the log file to the job folder
[ ! -f "/data/vision/polina/scratch/wpq/github/interpretability/scripts/slurm-${JOB_ID}.err" ] || mv "/data/vision/polina/scratch/wpq/github/interpretability/scripts/slurm-${JOB_ID}.err" "${JOB_OUTPUT}/"
[ ! -f "/data/vision/polina/scratch/wpq/github/interpretability/scripts/slurm-${JOB_ID}.out" ] || mv "/data/vision/polina/scratch/wpq/github/interpretability/scripts/slurm-${JOB_ID}.out" "${JOB_OUTPUT}/"
