#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:lgpu:4   
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --time=0-07:30
module load python/3.6.3
source ../tensorflow/bin/activate
python3 main.py 
