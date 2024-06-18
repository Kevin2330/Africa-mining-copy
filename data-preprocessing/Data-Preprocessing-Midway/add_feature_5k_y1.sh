#!/bin/bash

#SBATCH --job-name=5k
#SBATCH --output=5k.out
#SBATCH --error=5k.err
#SBATCH --ntasks=10
#SBATCH --partition=caslake
#SBATCH --account=macs30123
#SBATCH --time=18:00:00
#SBATCH --mem=32G

module load python
source /home/kaiwen1/30123-Project-Kaiwen/myenv/bin/activate

# Run the Python script using mpiexec
mpiexec -n 10 python add_feature_5k_y1.py