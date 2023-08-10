#!/bin/bash -l


#SBATCH --job-name=1-2
#SBATCH --nodes=16
#SBATCH --output=mybundle.o%j
#SBATCH --exclusive
#SBATCH  --constraint=cpu
#SBATCH  --qos=regular
#SBATCH --time=10


# Number of nodes required by each bundled job export
SLURM_NNODES=8

cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/bundle/ne30_F2010/ens//workdir.1/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/bundle/ne30_F2010/ens//workdir.2/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
wait 
