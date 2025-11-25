import pdb
import os
from shutil import copy
import time
import subprocess
import shlex
import yaml
import glob

# 20240425
# Modifying script to run on validation cases, too. So I will only have 1 script.
# Dont set casename from yaml.

#20230829
# BMW 
# Script to customize the zppy_cfg for each run and submit it.  
validate = True # set to true to run on validation dir. Otherwise, runs on ensemble dir. 

######################################################################
# Automatically set paths. 
with open("../../run/config_ensemble.yaml",'r') as stream:
    cfg = yaml.safe_load(stream)
root = cfg['run_root']
ens_name = cfg['ensemble_name']
if validate:
    ens_root = os.path.join( root, ens_name, 'validate' )
    wwwsubdir = 'validate'
else:
    ens_root = os.path.join( root, ens_name, 'ens' )
    wwwsubdir = 'ens'
cfg_template = cfg['cfg_template']
cfg_template_p4k = cfg['cfg_template_p4k']

######################################################################

dic={
    'template':cfg_template,
    'ensemblename':ens_name,
    'ens_root':ens_root,
    'validate':validate,
    'wwwsubdir': wwwsubdir
    }

def mk_cfg( dic ):
    cwd = os.getcwd() # should be inside the case directory.
    parentpath = os.path.abspath(os.path.join(cwd, os.pardir)) # workdir path. 
    workdir = dic['wd']
    casename=dic['casename']
    if not validate:
        casename_w_wd = casename +'.' + workdir
    else:
        casename_w_wd = casename
    outfn = 'zppy.cfg'
    copy( dic['template']  , outfn ) 
    outfile = open( outfn,"a")
    # Replace all instances of $CWD with the current working dir
    # Replace all instances of $CASENAME with the casename.
    # Replace all instances of $ENSEMBLENAME with the ensemblename.
    # Replace all instances of $WORKDIR with the workdir.
    # Replace all instances of $CASENAME_w_WD with the casename + WD number. 
    infile = open(outfn,'r')
    list_of_lines = infile.readlines()
    infile.close()
    for i in range(len(list_of_lines)):
        if '$CWD' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$CWD',cwd)
        if '$CASENAME' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$CASENAME',dic['casename'])
        if '$ENSEMBLENAME' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$ENSEMBLENAME',dic['ensemblename'])
        if '$WORKDIR' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$WORKDIR',workdir)
        if '$SN_w_WD' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$SN_w_WD',casename_w_wd)
        if '$WWWSUBDIR' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$WWWSUBDIR',wwwsubdir)
    outfile = open(outfn, "w") # write over it. 
    outfile.writelines(list_of_lines)
    outfile.close()
    

# By default this loops over the workdirs.
# If workdir_spec is not empty, it will run on that specific workdir. 
def submit_zppy( dic , workdir_spec=[]):
    full_path_of_cases_to_run = []
    thisdir=os.getcwd()
    if not validate: 
        work_directory_list = [wd for wd in work_directory_list if any(glob.glob (os.path.join( dic['ens_root'], wd, '**','archive','atm','hist','*.nc' )))  ]
        for wd in work_directory_list:
            cases_to_run = [d for d in os.listdir( os.path.join( dic['ens_root'], wd)) if any(glob.glob (os.path.join( dic['ens_root'], wd, d,'archive','atm','hist','*.nc' ))) ]
            full_path_of_cases_to_run = [os.path.join( dic['ens_root'], wd, case) for case in cases_to_run ]
    if validate:
        cases_to_run = [d for d in os.listdir( dic['ens_root']) if any(glob.glob (os.path.join( dic['ens_root'], d,'archive','atm','hist','*.nc' ))) ]    
        full_path_of_cases_to_run = [os.path.join( dic['ens_root'], case) for case in cases_to_run ]

    if  workdir_spec: # if workdir spec, overwrite cases to run. 
        full_path_of_cases_to_run = []
        work_directory_list = workdir_spec
        if validate:
            cases_to_run = [d for d in workdir_spec if any(glob.glob (os.path.join( dic['ens_root'], d,'archive','atm','hist','*.nc' ))) ]    
            full_path_of_cases_to_run = [os.path.join( dic['ens_root'], case) for case in cases_to_run ]
        else:
            for wd in workdir_spec:
                cases_to_run = [d for d in os.listdir( os.path.join( dic['ens_root'], wd)) if any(glob.glob (os.path.join( dic['ens_root'], wd, d,'archive','atm','hist','*.nc' ))) ]
                full_path_of_cases_to_run = [os.path.join( dic['ens_root'], wd, case) for case in cases_to_run ]

    if len(full_path_of_cases_to_run) < 1:
        print( 'found no data. pausing so you can look around')
        pdb.set_trace()
    for case_path in full_path_of_cases_to_run:
        dic['casename']=os.path.basename(case_path)
        if 'p4k' in case_path:
            dic['template'] = cfg_template_p4k
        if not 'p4k' in case_path:
            dic['template'] = cfg_template
        if validate:
            dic['wd'] = os.path.basename(case_path)
        if not validate:
            dic['wd'] = [wd for wd in case_path.split('/') if wd.startswith('workdir')][0]

        copy( dic['template'], case_path ) 
        os.chdir( case_path )
        mk_cfg( dic )
        os.remove( dic['template'] ) # clean up by deleting the template.
        execute=True
        if execute:
            print('submitting zppy for {} {}'.format( dic['wd'], case_path ))
            os.system('zppy -c zppy.cfg')
        os.chdir( thisdir )
            
                

# Two ways to run this script: 
# Loop over all workdirs:
submit_zppy(dic)

# Run on a specific list of workdirs:
#submit_zppy( dic, ['workdir.28']) # with validate=False

# submit_zppy( dic, ['validate.dnet-2.1_opt_rw2_20240523104406']) # with validate=True
# submit_zppy( dic, ['validate.dnet-1.5_opt_20240523084559','validate.dnet-1.5_opt_20240523084559.p4k']) # with validate=True
# submit_zppy( dic, ['validate.dnet-2.1_opt_rw2_20240523104406','validate.dnet-2.1_opt_rw2_20240523104406.p4k']) # with validate=True
