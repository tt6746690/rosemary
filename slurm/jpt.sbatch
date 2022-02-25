#!/bin/bash
#SBATCH --job-name=jpt
#SBATCH --output=/data/vision/polina/scratch/wpq/github/code_snippets/slurm/jpt.out
#SBATCH --error=/data/vision/polina/scratch/wpq/github/code_snippets/slurm/jpt.err

#SBATCH --time=1000-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodelist=wasabi
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB

source ~/.bashrc
conda init zsh
conda activate cxr

export XDG_RUNTIME_DIR=""

cd /data/vision/polina/scratch/wpq/github/

jupyter notebook --no-browser --port=8895