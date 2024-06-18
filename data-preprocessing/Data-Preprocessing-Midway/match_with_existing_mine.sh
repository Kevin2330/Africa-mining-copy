#!/bin/bash

#SBATCH --job-name=ssd-simple-method
#SBATCH --output=ssd-match.out
#SBATCH --error=ssd-match.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd


module load python
source /home/kaiwen1/30123-Project-Kaiwen/myenv/bin/activate

# Run the Python script using mpiexec
mpiexec -n 10 python match_with_existing_mine.py