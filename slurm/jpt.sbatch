#!/bin/bash
#SBATCH --job-name=jpt
#SBATCH --time=1000-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodelist=perilla
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --output=/data/vision/polina/scratch/wpq/github/rosemary/slurm/jpt.out

# Customize this line to point to conda installation
path_to_conda="/data/vision/polina/shared_software/miniconda3"
# huggingface cache
export HF_HOME='/data/vision/polina/scratch/wpq/huggingface_cache'

source ${path_to_conda}/bin/activate cxr
cd /data/vision/polina/scratch/wpq/github/
jupyter notebook --no-browser --port=8899