# 20220120
# changes case.st_archive to 1 node and makes sure DOUT_S_ROOT is updated to the proper case (and not a clone's parent)
from ncclimo_surrogate_fields import call_ncclimo_sur
import pdb
import os
import fileinput
import sys

# user set #
root = '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/'
case_ens = '20210813.F2010.ne30pg2_oECv3.chrysalis'
case_ctrl = '20210813.F2010.ne30pg2_oECv3_10nodes.chrysalis'
# end user set #
##############################################################################
def change_node_count( fname, n_nodes):
    for line in fileinput.input(fname, inplace=1):
        if line.strip().startswith('#SBATCH  --nodes'):
            line = '#SBATCH  --nodes={}\n'.format(n_nodes)
        sys.stdout.write(line)

def do_st_archive( caseroot ):
    os.chdir(caseroot )
    cmd = './xmlchange DOUT_S_ROOT={}'.format( os.path.join( caseroot, 'archive')) #
    os.system(cmd)
    change_node_count( 'case.st_archive', 1 )
    os.system( './case.st_archive')
    


# To run on a particular simulation:
#do_st_archive('/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ens/workdir.33/20210813.F2010.ne30pg2_oECv3.chrysalis' )



# all runs:
for ctrl in os.listdir( os.path.join( root, 'ctrl')):
    do_st_archive( os.path.join( root, 'ctrl',case_ctrl ))
for wd in os.listdir(  os.path.join( root, 'ens')):
    do_st_archive( os.path.join( root, 'ens',wd, case_ens ))

