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
    outfn = 'bundle.{}-{}.sh'.format(startdir,enddir)
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
    
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 2, 4 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 5, 7 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 8, 10 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 11, 13 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 14, 16 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 17, 19 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 20, 22 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 23, 25 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 26, 28 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 29, 31 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 32, 34 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 35, 37 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 38, 40 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 41, 43 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 44, 46 )
# mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 47, 49 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 50, 52 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 53, 55 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 56, 58 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 59, 61 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 62, 64 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 65, 67 )
#mk_bundle('bundle_template.sh', 1020, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',30, 68, 70 )

# Switching over to the 29 node case. 
#mk_bundle('bundle_template.sh', 600, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list_29nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',29, 71, 73 )
#mk_bundle('bundle_template.sh', 600, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list_29nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',29, 74, 76 )
#mk_bundle('bundle_template.sh', 600, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list_29nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',29, 77, 79 )
#mk_bundle('bundle_template.sh', 1800, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 80, 89 )
#mk_bundle('bundle_template.sh', 1800, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_list_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 90, 99 )
# Note BMW changed the directory structure after this line 11-8-2021
# this means all cases run before this date have a CASEROOT and RUNDIR that are inconsistent with their current locations, which can break case.st_archive
#mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 101, 110 )
#mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 111, 120 )
#mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 121, 130 )
#mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 131, 140 )
#mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 141, 150 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 151, 160 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 161, 170 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 171, 180 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 181, 190 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 191, 200 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 201, 210 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 211, 220 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 221, 230 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 231, 240 )
# mk_bundle('bundle_template.sh', 1700, '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ens', '20210813.F2010.ne30pg2_oECv3.chrysalis',10, 241, 250 )


