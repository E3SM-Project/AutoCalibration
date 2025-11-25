## to run:  ./run_dakota.sh >& run_dakota.log &

name=${PWD##*/} # Get the path basename.
echo 'name is ' $name
echo 'workdir root is ' $SCRATCH/E3SM_simulations/dakota/$name/ens/workdir  #YOU MUST SET THIS IN dakota_e3sm.in. 
echo 'stop now and set the workdir to match this in dakota_e3sm.in'
echo 'then uncomment the exit'

#exit


#/bin/rm -rfi $SCRATCH/E3SM_simulations/dakota/${name}/ens/workdir.* #Remove any existing workdirs in scratch for this basename. 

## Clean up
/bin/rm -f LHS_* lhs*
/bin/rm -f *.out *.err *.rst
/bin/rm -f fwdUQ_preds.dat

## Clean up modules
#module purge
# Choose these for Skybridge. 
#module load dakota
#module load anaconda    # Choose this for HPC cluster

# if chrysalis
export PATH=$PATH:/soft/dakota/6.8/bin/


## Run dakota
#dakota -i fwdUQ_freestream.in -o fwdUQ.out  -e fwdUQ.err
dakota -i dakota_e3sm.in -o dakota_e3sm.out  -e dakota_e3sm.err

