#!/bin/bash


#########################################################################
ROOT=/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ctrl
#########################################################################

cd $ROOT

zstash create --hpss=/home/w/wagmanbe/E3SMv3/dakota/v3_pmcpu/ne30_F2010/by_parts/ctrl --exclude="tmp*" . 2>&1 | tee zstash_create_20240131.log 

# query
zstash ls --hpss=/home/w/wagmanbe/E3SMv3/dakota/v3_pmcpu/ne30_F2010/by_parts/ctrl

# Update:
#cd $ROOT
#zstash update --hpss=/home/w/wagmanbe/E3SMv3/dakota/v3_pmcpu/ne30_F2010/by_parts/ctrl --exclude="tmp*"   2>&1 | tee zstash_update_20240131.log 





