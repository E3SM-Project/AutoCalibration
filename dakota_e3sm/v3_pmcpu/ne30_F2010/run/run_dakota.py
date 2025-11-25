# 20230412
# Benjamin Wagman

# To create ensemble:
# Set your parameters in config_ensemble.yaml

# Finally, 
# python run_dakota.py

import os
import pdb
import shutil
import glob
import yaml

#################  All user set via .yaml file #################

with open("config_ensemble.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream)

# Read from .yaml file
ensemble_name = cfg["ensemble_name"]
run_root = cfg["run_root"]
infile = cfg["infile"]
nsamples = cfg["nsamples"]

#################  Ensemble setup  #################  

workdir_root = '{}/{}/ens/'.format(run_root, ensemble_name)
workdir_name = '{}/workdir'.format( workdir_root) # Dakota will append a .1, .2, etc to 'workdir'

## Clean up this directory.
cleanupfiles = glob.glob('LHS_*') + glob.glob( 'lhs*') + glob.glob('*.log')+ glob.glob('*.out') + glob.glob('*.err') + glob.glob('*.rst') + glob.glob('fwdUQ_preds.dat')
for f in cleanupfiles:
    if os.path.exists( f ) :
        os.remove( f )

# If ensemble workdir root already exist, exit.
if os.path.isdir( workdir_root ):
    print('{} already exists. Exiting'.format(workdir_root))
    exit()


# Create a local dakota_e3sm.in from templates/[infile]. 
def update_infile( infile ):
    if not os.path.isdir('old'):
        os.mkdir('old')

    # Remove dakota_e3sm.in if it exists.
    if os.path.isdir( 'dakota_e3sm.in' ):
        print(' moving existing dakota_e3sm.in to old')
        shutil.move('dakota_e3sm.in', 'old/')

    # Create a new dakota_e3sm.in from the template. 
    print( 'setting ensemble_run_root in dakota_e3sm.in')
    shutil.copy( '{}'.format(infile), 'dakota_e3sm.in' ) 
    outfile = open( 'dakota_e3sm.in',"a")
        
    # Replace all instances of $WD_FROM_RUN_DAKOTA.PY with ensemble root dir. 
    infile = open('dakota_e3sm.in','r')
    list_of_lines = infile.readlines()
    infile.close()
    for i in range(len(list_of_lines)):
        if '$WD_FROM_RUN_DAKOTA.PY' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$WD_FROM_RUN_DAKOTA.PY',workdir_name)
            print('wrote workdir path from config file to dakota_e3sm.in')
        if '$N_SAMPLES_FROM_RUN_DAKOTA.PY' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$N_SAMPLES_FROM_RUN_DAKOTA.PY',str(nsamples))
            print('wrote nsamples from config file to dakota_e3sm.in')
    outfile = open('dakota_e3sm.in', "w") # write over it. 
    outfile.writelines(list_of_lines)
    outfile.close()
    print('wrote dakota_e3sm.in')


update_infile( infile )
# To-do. Create zppy config file from template here? 

cmd = 'dakota -i dakota_e3sm.in -o dakota_e3sm.out  -e dakota_e3sm.err'
print('running dakota with {}'.format(cmd))
## Run dakota
os.system(cmd)
