#!/bin/csh 

#module load sems-env # Might need this on skybridge. 
# Make sure you have this version checked out! 


#parent_dir = $HOME/e3sm_builds/
set cwd=$PWD
set code_root = $HOME/E3SM/code/20210813/
set parent_dir = $SCRATCH/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ctrl
set parent_case=20210813.F2010.ne30pg2_oECv3_10nodes.chrysalis  #Must cp its casescripts dir subdir contents into its main dir. 
set run_root = .
set newcase=20210813.F2010.ne30pg2_oECv3.chrysalis

mkdir ./old

if ( -d ./${newcase} ) then
    echo "removing old dir"
    rm -rf old/*
    mv -f ${newcase} old
endif 

# NOTE: Copy the contents of the case_scripts dir into the case_dir if not already done!
 ${code_root}/cime/scripts/create_clone --keepexe --case ${run_root}/${newcase} --clone ${parent_dir}/${parent_case} 

# If you get a warning from CIME about "SourceMods" then you have the wrong CIME version and need to run git clean -d in cime. 

# change the run and build directories to be local. 
cd ${newcase}
set casedir=$PWD
./xmlchange RUNDIR=${casedir}/run
./xmlchange DOUT_S_ROOT=${casedir}/archive #BMW 20220124 (after running the ensemble and realizing I should have included this). 


# Append dakota input params to user_nl_eam
cat ../e3sm-inp.yaml >> user_nl_eam



# Don't submit the run.

# Let Dakota know we're finished. 
touch all_done.txt
date >> e3sm.log 
