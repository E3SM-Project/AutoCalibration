import pdb
import os
from shutil import copy
import time
import subprocess
import shlex
import yaml

#202301003
# BMW 
# Script to customize the zppy_cfg for each run and submit it.  
# Unlike for the ensemble, the validation simulations have unique casenames.
# So, casename cannot be set from the yaml. 

do_p4k = True  # Set to true to do p4k runs. Only does those runs.

######################################################################
# Automatically set paths. 
with open("config_ensemble.yaml",'r') as stream:
    cfg = yaml.safe_load(stream)
root = cfg['run_root']
ens_name = cfg['ensemble_name']
hm_root = os.path.join( root, ens_name, 'hm' )
cfg_template = cfg['cfg_template']
cfg_template_p4k = cfg['cfg_template_p4k']
######################################################################


dic={
    'template':cfg_template,
    'ensemblename':ens_name,
    'ens_root':hm_root,
    }

if do_p4k:
    dic['template'] = cfg_template_p4k


def mk_cfg( dic ):
    cwd = os.getcwd() # should be inside the case directory.
    parentpath = os.path.abspath(os.path.join(cwd, os.pardir)) # workdir path. 
    #workdir = os.path.basename( parentpath )
    #casename_w_wd = casename +'.' + workdir
    casename=dic['casename']
    outfn = 'zppy.cfg'
    copy( dic['template']  , outfn ) 
    outfile = open( outfn,"a")
    # Replace all instances of $CWD with the current working dir
    # Replace all instances of $CASENAME with the casename.
    # Replace all instances of $ENSEMBLENAME with the ensemblename.
    # Replace all instances of $WORKDIR with the workdir <--- Obsolete for validation runs. 
    # Replace all instances of $CASENAME_w_WD with the casename + WD number. <--- Use casename for validation runs. 
    infile = open(outfn,'r')
    list_of_lines = infile.readlines()
    infile.close()
    for i in range(len(list_of_lines)):
        if '$CWD' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$CWD',cwd)
        if '$CASENAME' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$CASENAME',casename)
        if '$ENSEMBLENAME' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$ENSEMBLENAME',dic['ensemblename'])
        #if '$WORKDIR' in list_of_lines[i]:
        #    list_of_lines[i] = list_of_lines[i].replace('$WORKDIR',workdir)
        if '$SN_w_WD' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('$SN_w_WD',casename)
    outfile = open(outfn, "w") # write over it. 
    outfile.writelines(list_of_lines)
    outfile.close()
    

# By default this loops over all the case directories inside of hm_root, e.g. /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/hm/[cases]
# If casename_spec is not empty, it will run on that specific workdir. 
# Has not been tested on p4k simulations. 
def submit_zppy( dic , casename_spec=[]):
    thisdir=os.getcwd()
    if not casename_spec:
        case_list = os.listdir( dic['ens_root'])
    else:
        case_list = casename_spec
    if do_p4k:
        case_list = [case for case in case_list if 'p4k' in case]
    for casename in case_list:
        dic['casename']=casename
        case_path = os.path.join( dic['ens_root'], casename )
        copy( dic['template'], case_path ) 
        os.chdir( case_path )
        mk_cfg( dic )
        os.remove( dic['template'] ) # clean up by deleting the template.
        execute=True
        if execute:
            if os.path.exists( os.path.join(case_path,'archive','atm','hist')):
                os.system('zppy -c zppy.cfg')
        os.chdir( thisdir )


# Two ways to run this script: 
# Loop over all cases in hm_root
submit_zppy(dic)


# Run on a specific list of dirs:
#submit_zppy( dic, ['v3alpha02.hm_next_11_mincdnc_7.5e06.p4k'])

