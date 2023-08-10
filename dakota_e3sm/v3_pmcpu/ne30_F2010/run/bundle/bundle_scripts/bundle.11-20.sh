#!/bin/bash -l


#SBATCH --job-name=11-20
#SBATCH --nodes=80
#SBATCH --output=mybundle.o%j
#SBATCH --exclusive
#SBATCH  --constraint=cpu
#SBATCH  --qos=regular
#SBATCH --time=900


# Number of nodes required by each bundled job export
SLURM_NNODES=8

cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.11/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.12/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.13/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.14/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.15/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.16/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.17/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.18/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.19/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.20/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
wait 
