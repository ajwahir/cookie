#!/bin/bash
#
SBATCH --job-name=ourProsTrainJob
SBATCH --nodes=1
SBATCH --cpus-per-task=4
SBATCH --time=12:00:00
SBATCH --mem=8GB
SBATCH --gres=gpu:0


module purge

cd/scratch/$USER/mytest1

python ddpg_train.py --model sads_walk