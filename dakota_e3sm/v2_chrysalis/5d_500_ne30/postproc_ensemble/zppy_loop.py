import pdb
import os
from shutil import copy
import time
import subprocess
import shlex

#20220125
# BMW 
# Script to customize the zppy_cfg for each run. 
# in the future I should have dakota edit this config file and place it in the workdir. But I am doing it after the fact. 

def mk_cfg( dic ):
    cwd = os.getcwd() # should be inside the workdir directory.
    workdir = os.path.basename( cwd )
    outfn = 'zppy.cfg'
    copy( dic['template'], outfn ) 
    outfile = open( outfn,"a")
        
    # Replace all instances of $CWD with the current working dir
    # Replace all instances of $CASENAME with the casename.
    # Replace all instances of $ENSEMBLENAME with the ensemblename.
    # Replace all instances of $WORKDIR with the workdir. 
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
    outfile = open(outfn, "w") # write over it. 
    outfile.writelines(list_of_lines)
    outfile.close()
    

def mk_zppy_cfg( dic ):
    thisdir=os.getcwd()
    for wd in os.listdir( dic['ensemble_root_dir']):
        wd_path = os.path.join( dic['ensemble_root_dir'], wd )
        copy( dic['template'], wd_path )
        os.chdir( wd_path )
        mk_cfg( dic ) 
        os.remove( dic['template'] ) # clean up by deleting the template.
        execute=True
        if execute:
            if os.path.exists( os.path.join(os.getcwd(), dic['casename'],'archive','atm','hist')):
                # insert a condition here that tests whether I have more than 10 runs in queue, e.g. uses
                # example: https://stackoverflow.com/questions/12828771/how-to-go-back-to-first-if-statement-if-no-choices-are-valid
                cmd = 'squeue -u ac.wagman'
                formatted_cmd = shlex.split(cmd) # ['squeue', '-u', 'ac.wagman']
                while True:
                    getq = subprocess.run( formatted_cmd,  capture_output=True, text=True )
                    nq = getq.stdout.count('\n')
                    print( 'you have {} runs in queue'.format( nq-1 ))
                    if nq <= 40:
                        os.system('zppy -c zppy.cfg')
                        break
                    print( 'too many runs in queue. sleeping 10 sec')
                    time.sleep( 10 )
        os.chdir( thisdir )

if __name__ == "__main__":
    dic={
    'template':'zppy_template.cfg',
    'casename':'20210813.F2010.ne30pg2_oECv3.chrysalis',
    'ensemblename':'5d_chr_500_ne30',
    'ensemble_root_dir':'/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ens'
    }
    
    mk_zppy_cfg(dic)


