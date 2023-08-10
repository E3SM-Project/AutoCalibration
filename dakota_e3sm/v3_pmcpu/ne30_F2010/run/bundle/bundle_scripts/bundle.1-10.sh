#!/bin/bash -l


#SBATCH --job-name=1-10
#SBATCH --nodes=80
#SBATCH --output=mybundle.o%j
#SBATCH --exclusive
#SBATCH  --constraint=cpu
#SBATCH  --qos=regular
#SBATCH --time=900


# Number of nodes required by each bundled job export
SLURM_NNODES=8

cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.1/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.2/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.3/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.4/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.5/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.6/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.7/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.8/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.9/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
cd /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens//workdir.10/20230802.v3alpha02.F2010.pmcpu.intel.8N
./case.submit --no-batch >LOG 2>&1 & 
wait 
