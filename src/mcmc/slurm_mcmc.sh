#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --output=res_mcmc-%j-%a.out
#SBATCH --error=res_mcmc-%j-%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -t 900

export OPENBLAS_NUM_THREADS=1
python mcmc.py config_mcmc.yaml
