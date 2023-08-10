#!/bin/bash
#SBATCH --time=600
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source  /lcrc/soft/climate/e3sm-unified/load_latest_e3sm_unified.sh  # chrysalis

export OMP_NUM_THREADS=2
python call_ncclimo.py
