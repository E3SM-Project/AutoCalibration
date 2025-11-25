# 20220125
import pdb
import os
import sys

# loop through runs, calling zppy. 

# user set #
root = '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/'
case_ens = '20210813.F2010.ne30pg2_oECv3.chrysalis'
case_ctrl = '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ctrl/20210813.F2010.ne30pg2_oECv3_control.chrysalis'
# end user set #
##############################################################################

def do_call_zppy( caseroot ):
    os.chdir(caseroot )
    cmd = './xmlchange DOUT_S_ROOT={}'.format( os.path.join( caseroot, 'archive')) #
    os.system(cmd)
    change_node_count( 'case.st_archive', 1 )
    os.system( './case.st_archive')
    
# To run on a particular simulation:
#do_st_archive('/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ens/workdir.33/20210813.F2010.ne30pg2_oECv3.chrysalis' )

# all runs:
for ctrl in os.listdir( os.path.join( root, 'ctrl')):
    do_call_zppy( os.path.join( root, 'ctrl',case_ctrl ))
for wd in os.listdir(  os.path.join( root, 'ens')):
    do_call_zppy( os.path.join( root, 'ens',wd, case_ens ))

