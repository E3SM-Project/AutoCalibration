import pdb
import os
from shutil import copyfile

# BMW 4-29-2021
# Script to make a bundle.

### Arguments ###
# template e.g. 'bundle_template.sh'
# time [minutes] e.g. 150 or 1020 (17 hours) 
# run_root e.g. '/lcrc/group/e3sm/ac.wagman/scratch/dakota/5d_chr_500' 
# e3sm_case e.g. 'cloned.E3SM.ne4pg2_ne4pg2'
# nodes per E3SM simulation e.g. 2
# startdir e.g. 71
# enddir e.g. 90
# Also requires: existing template file. 


def mk_bundle( template, minutes, run_root, e3sm_case, nodes_per, startdir, enddir ):
    nruns = 1 + enddir - startdir
    nodes_tot = nruns * nodes_per

    if not os.path.exists("../bundle_scripts"):
        os.makedirs("../bundle_scripts")
        
    outfn = '../bundle_scripts/bundle.{}-{}.sh'.format(startdir,enddir)
    jobname = '{}-{}'.format(startdir,enddir)

    copyfile( template, outfn ) 
    outfile = open( outfn,"a")
        
    # Copy the following lines nruns times.
    for i in range(nruns):
        n = startdir + i
        l1 = 'cd {}/workdir.{}/{}\n'.format(run_root,n,e3sm_case) 
        l2 = './case.submit --no-batch >LOG 2>&1 & \n'
        outfile.write(l1)
        outfile.write(l2)
    outfile.write('wait \n')
    outfile.close()

    # edit the sbatch lines. 
    infile = open(outfn,'r')
    list_of_lines = infile.readlines()
    infile.close()
    for i in range(len(list_of_lines)):
        if 'NAME_ARG' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('NAME_ARG',jobname)
        if 'NODES_TOT_ARG' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('NODES_TOT_ARG',str(nodes_tot))
        if 'NODES_PER_ARG' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('NODES_PER_ARG',str(nodes_per))
        if 'MINUTES_ARG' in list_of_lines[i]:
            list_of_lines[i] = list_of_lines[i].replace('MINUTES_ARG',str(minutes))

    outfile = open(outfn, "w") # write over it. 
    outfile.writelines(list_of_lines)
    outfile.close()
    
# Test
mk_bundle('templates/bundle_template.sh', 10, '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/bundle/ne30_F2010/ens/', '20230802.v3alpha02.F2010.pmcpu.intel.8N',8, 1, 2 ) 

# Real
#mk_bundle('templates/bundle_template.sh', 900, '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/', '20230802.v3alpha02.F2010.pmcpu.intel.8N',8, 1, 10 ) 
#mk_bundle('templates/bundle_template.sh', 900, '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/', '20230802.v3alpha02.F2010.pmcpu.intel.8N',8, 11, 20 ) 


