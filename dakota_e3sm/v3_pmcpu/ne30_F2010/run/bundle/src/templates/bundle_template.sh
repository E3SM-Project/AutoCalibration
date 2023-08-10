#!/bin/bash -l


#SBATCH --job-name=NAME_ARG
#SBATCH --nodes=NODES_TOT_ARG
#SBATCH --output=mybundle.o%j
#SBATCH --exclusive
#SBATCH  --constraint=cpu
#SBATCH  --qos=regular
#SBATCH --time=MINUTES_ARG


# Number of nodes required by each bundled job export
SLURM_NNODES=NODES_PER_ARG

