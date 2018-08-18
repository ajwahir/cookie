#!/bin/bash
#
#SBATCH --job-name=myProsTrainJob
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1


module purge

cd/scratch/$USER/mytest1

python ddpg_train.py --visualize --train --model sads_walk