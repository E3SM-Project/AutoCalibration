#!/bin/bash
# BMW modified the original zstash script to work with his ensemble.
# That orig. script is found here: https://acme-climate.atlassian.net/wiki/spaces/ED/pages/2309226536/Running+E3SM+step-by-step+guide#Long-Term-Archiving-with-zstash
# Run on chrysalis
# Load E3SM Unified
source /lcrc/soft/climate/e3sm-unified/load_latest_e3sm_unified_chrysalis.sh

# List of experiments to archive with zstash
ENS_name=5d_chr_500_ne30
cd /lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/${ENS_name}/ens
WORKDIRS=($(ls ))

# Loop over workdirs
for WD in "${WORKDIRS[@]}"
do
    echo === Archiving ${WD} ===
    cd ${WD}/
    mkdir -p zstash
    stamp=`date +%Y%m%d`
    time zstash create -v --hpss=globus://nersc/home/w/wagmanbe/E3SMv2/dakota/${ENS_name}/${WD}  --maxsize 128 . 2>&1 | tee zstash/zstash_create_${stamp}.log
    cd ..
done

# If this works, add another command to do the control simulation, below. 
cd /lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/${ENS_name}/ctrl/20210813.F2010.ne30pg2_oECv3_control.chrysalis
mkdir -p zstash
stamp=`date +%Y%m%d`
time zstash create -v --hpss=globus://nersc/home/w/wagmanbe/E3SMv2/dakota/${ENS_name}/ctrl/20210813.F2010.ne30pg2_oECv3_control.chrysalis  --maxsize 128 . 2>&1 | tee zstash/zstash_create_${stamp}.log
