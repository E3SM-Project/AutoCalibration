#!/bin/bash
#SBATCH  --qos=regular
#SBATCH  --account=e3sm
#SBATCH  --job-name=run_dakota
#SBATCH  --nodes=1
#SBATCH  --output=run_dakota.%j 
#SBATCH  --exclusive 
#SBATCH  --constraint=cpu
#SBATCH  --time=12:00:00


python run_dakota.py
