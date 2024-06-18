#!/bin/bash

#SBATCH --job-name=add_feature_y0
#SBATCH --output=add_feature_y0.out
#SBATCH --error=add_feature_y0.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:4
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd


module load python
source /home/kaiwen1/30123-Project-Kaiwen/myenv/bin/activate

# Run the Python script using mpiexec
mpiexec -n 10 python add_feature_y0.py