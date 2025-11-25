
# 20230824
# BMW
import pdb
import os
import fileinput
import sys
import yaml


######################################################################
# Run short time archiver if necessary. 
# No need to set anything. Automatically reads relevant info from yaml.
# Can run repeatedly without harm.
# Can submit each run as its own job. Or just open an interactive job and run it without the sbatch call.
# salloc --nodes 1 --qos interactive --time 01:00:00  --account e3sm --constraint cpu

######################################################################

# if "ensemble" is true, set hm to false. If hm is true, set ensemble to false. 
do_p4k=True # If true and enssemble = false: do p4k in addition to F2010. If true and ensemble=True: do only p4k. 
ensemble=False # run on "ens"
hm=False # run on history matching. 
validate=True # run on validate sims

# Automatically set paths. 
with open("../../run/config_ensemble.yaml",'r') as stream:
    cfg = yaml.safe_load(stream) 
root = cfg['run_root']
casename = cfg['parent_case_name_pm-cpu']  # Could set here to +4K casename if wanted. 
ens_name = cfg['ensemble_name']

if ensemble:
    ens_root = os.path.join( root, ens_name, 'ens' )
else:
    if hm:
        ens_root = os.path.join( root, ens_name, 'hm')
    if validate:
        ens_root = os.path.join( root, ens_name, 'validate')
if do_p4k:
    casename=casename+'.p4k'
######################################################################
#### No longer needed. node count is now 1 by default ################
# Retaining as an example because workflow useful for modifying cfg files if I want.
def change_node_count( fname, n_nodes):
    for line in fileinput.input(fname, inplace=1):
        if line.strip().startswith('#SBATCH  --nodes'):
            line = '#SBATCH  --nodes={}\n'.format(n_nodes)
        sys.stdout.write(line)
######################################################################

# Preconditions decides whether to run case.st_archive.
# return true if all of the following are met:
#1) The last run of the E3SM model was successful 
    # There is a line CaseStatus that contains case.run  
    # True if there is no 'error' in the last line of CaseStatus that contains 'case.run'   
#2) case.st_archive has not been run since the last time E3SM ran. 
    # True if there is no 'st_archive success' in  CaseStatus after the last line with case.run.  

def archive_preconditions( status_dic ):
    meets_preconditions=False
    e3sm_has_run=True
    archive_has_run=False
    if 'ens' in os.getcwd():
        dir_short = os.getcwd().split("ens",1)[1]
    if 'hm' in os.getcwd():
        dir_short = os.getcwd().split("hm",1)[1]
    if 'validate' in os.getcwd():
        dir_short = os.getcwd().split("validate",1)[1]
    print( dir_short)
    with open("CaseStatus") as f:
        lines = f.readlines()
        actual_lines = [i for i in lines if i.startswith('202')]  # Remove the lines that are just dashes.
        last_line_with_run= ''
        for line in actual_lines:
            if 'case.run' in line:
                last_line_with_run = line
        if not last_line_with_run:
            print('this case has not yet run')
            e3sm_has_run=False
            status_dic['e3sm_not_yet_run'].append(dir_short)
        elif 'error' in last_line_with_run:
            print('this case failed to complete')
            e3sm_has_run=False
            status_dic['e3sm_failed'].append(dir_short)
        if e3sm_has_run:
            # If here, the model has run successfully at least once.
            # Check if case.st_archive has been run since then.
            # Isolate the lines after the last line containing 'case.run'
            # Check those lines for st_archive success
            lines_after_caserun = []
            for line in actual_lines[::-1]:
                if not 'case.run' in line:
                    lines_after_caserun.append(line)
                else:
                    break
            for line in lines_after_caserun:
                if 'st_archive success' in line:
                    archive_has_run=True
            if not archive_has_run:
                meets_preconditions=True
                status_dic['called_st_archive'].append( dir_short )
            else:
                print('already archived, will not run archiver again')
                status_dic['both_previously_run'].append( dir_short )
    return meets_preconditions, status_dic

def do_st_archive( casedir, status_dic ):
    os.chdir(casedir )
    meets_preconditions, status_dic = archive_preconditions( status_dic )
    if meets_preconditions:
        #os.system( 'sbatch --qos regular --time 00:00:30 ./case.st_archive')
        os.system( './case.st_archive')
    return status_dic


# Each run gets sorted into a category: 1) e3sm has run and st archive already run. 2) e3sm has run and now submtting archive 3). e3sm has not yet run 4) e3sm has failed.
status_dic={'both_previously_run':[],
            'called_st_archive':[],
            'e3sm_not_yet_run':[],
            'e3sm_failed':[]} 

# To run on a particular simulation:
#status_dic = do_st_archive('/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.26/20230802.v3alpha02.F2010.pmcpu.intel.8N/', status_dic)



# Loop over ensemble.
if ensemble:
    directories = [d for d in os.listdir( ens_root ) if os.path.isdir( os.path.join( ens_root, d))]
    for wd in directories:
        if casename in os.listdir( os.path.join( ens_root, wd )):
            status_dic = do_st_archive( os.path.join( ens_root, wd, casename ), status_dic)
    # print summary stats. 
    for k in status_dic.keys():
        print( k )
        print( len(status_dic[k]))
    print( 'list of failed runs is ')
    print( status_dic['e3sm_failed'] )

# loop over history matching ensemble. 
if hm or validate:
    #directories = os.listdir( ens_root )
    directories = [d for d in os.listdir( ens_root ) if os.path.isdir( os.path.join( ens_root, d))]
    directories = [d for d in directories if not 'zstash' in d]
    if not do_p4k:
        directories = [d for d in directories if not 'p4k' in d]
    for casename in directories:
        print( os.path.join( ens_root, casename )) 
        status_dic = do_st_archive( os.path.join( ens_root, casename ), status_dic)
        
                
    # print summary stats. 
    for k in status_dic.keys():
        print( k )
        print( len(status_dic[k]))
    print( 'list of failed runs is ')
    print( status_dic['e3sm_failed'] )

    
