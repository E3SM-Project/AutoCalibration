import os
import shutil
import yaml

# This script gets called from within the workdir by "run.py"
# It sets up the E3SM run according to the yaml file, and passes the parameter values from Dakota into namelist. 
# Depends on yaml file being  named "config_ensemble.yaml" 
# Can run on compy or chrysalis without modifying this script. It detects which machine. 

with open("config_ensemble.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream) 

# Set paths.
machine = cfg['machine'] # options are chrysalis, compy, and pm-cpu
code_root = cfg['e3sm_code']
stopmonths = cfg['stopmonths']
stopmonths_p4k = cfg['stopmonths_p4k']
startdate=cfg['startdate']
if machine=='compy':
    parent_dir = cfg['parent_case_dir_compy']
    parent_case = cfg['parent_case_name_compy']
if machine=='chrysalis':
    parent_dir = cfg['parent_case_dir_chry']
    parent_case = cfg['parent_case_name_chry']
if machine=='pm-cpu':
    parent_dir = cfg['parent_case_dir_pm-cpu']
    parent_case = cfg['parent_case_name_pm-cpu']

clone_case= parent_case


# Clean up  
if not os.path.isdir('old'):
    os.mkdir('old')

if os.path.isdir( clone_case ):
    print( "moving existing clone case dir to old" ) 
    shutil.move( clone_case, 'old/')


# Setup the case.
# Clone the control run, which is a branch run initialized from an 11-month spinup. The ens runs will inherit run type from control. 
cmd = '{}/cime/scripts/create_clone --keepexe --case {} --clone {}/{}/case_scripts'.format(code_root, clone_case, parent_dir, parent_case  )
print(cmd) 
os.system( cmd )
os.chdir( clone_case )
if machine == 'compy':
    os.system( './xmlchange JOB_WALLCLOCK_TIME=24:00:00') # Compy. 
else:
    os.system( './xmlchange JOB_WALLCLOCK_TIME=14:00:00') # Chrysalis, pm-cpu
# change the run and build directories to be local. By default they are the same as the parent case. 
os.system( './xmlchange RUNDIR={}/run'.format(os.getcwd()))
os.system( './xmlchange DOUT_S_ROOT={}/archive'.format(os.getcwd()))
os.system( './xmlchange DOUT_S=FALSE') # Unfortunately I'll have to come back and submit this later with sbatch -N 1 ./case.st_archive because otherwise it uses too many nodes. 
os.system( './xmlchange STOP_N={}'.format(stopmonths))
# Append dakota input params to user_nl_eam
os.system( 'cat ../e3sm-inp.yaml >> user_nl_eam')
#os.system( 'cat ../fexcl1.txt >> user_nl_eam')
os.chdir("..")

# Set up a +4k run.
clone_case_p4k = clone_case + '.p4k'
cmd = '{}/cime/scripts/create_clone --keepexe --case {} --clone {}/{}/case_scripts'.format(code_root, clone_case_p4k, parent_dir, parent_case  )
print(cmd) 
os.system( cmd )
os.chdir( clone_case_p4k )
if machine == 'compy':
    os.system( './xmlchange JOB_WALLCLOCK_TIME=10:00:00')
else:
    os.system( './xmlchange JOB_WALLCLOCK_TIME=05:00:00')
# change the run and build directories to be local. By default they are the same as the parent case. 
os.system( './xmlchange RUNDIR={}/run'.format(os.getcwd()))
os.system( './xmlchange DOUT_S_ROOT={}/archive'.format(os.getcwd()))
os.system( './xmlchange DOUT_S=FALSE')
os.system( './xmlchange STOP_N={}'.format(stopmonths_p4k))
os.system( './xmlchange REST_N={}'.format(stopmonths_p4k))
# Append dakota input params to user_nl_eam
os.system( 'cat ../e3sm-inp.yaml >> user_nl_eam')
#os.system( 'cat ../fexcl1.txt >> user_nl_eam')
# +4k SSTICE
if machine == 'compy':
    os.system('./xmlchange -file env_run.xml -id SSTICE_DATA_FILENAME -val \'/compyfs/wagm505/e3sm_input/ac.xzheng/E3SM/inputdata/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_c20190821_plus4K.nc\'')
if machine == 'chrysalis':
    os.system('./xmlchange -file env_run.xml -id SSTICE_DATA_FILENAME -val \'/home/ac.xzheng/E3SM/inputdata/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_c20190821_plus4K.nc\'')
if machine == 'pm-cpu':
    os.system('./xmlchange -file env_run.xml -id SSTICE_DATA_FILENAME -val \'/global/cfs/cdirs/e3sm/emulate/inputdata/ac.xzheng/E3SM/inputdata/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_c20190821_plus4K.nc\'')
