#!/bin/bash
#SBATCH --job-name=nsim
#SBATCH --output=nsim-%j-%a.out
#SBATCH --error=nsim-%j-%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
####SBATCH -a 0-179%3
#SBATCH -a 0-179%3
#SBATCH -t 900
python surrogate_nsim_test.py misc_config/nsim_24x48_10yr_ctrl.yaml -p $1 -v $SLURM_ARRAY_TASK_ID
python surrogate_nsim_test.py misc_config/nsim_24x48_5yr_ctrl.yaml -p $1 -v $SLURM_ARRAY_TASK_ID
python surrogate_nsim_test.py misc_config/nsim_180x360_10yr_ctrl.yaml -p $1 -v $SLURM_ARRAY_TASK_ID
python surrogate_nsim_test.py misc_config/nsim_180x360_5yr_ctrl.yaml -p $1 -v $SLURM_ARRAY_TASK_ID
