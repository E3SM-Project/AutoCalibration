import pdb
import os
from shutil import copyfile


### Arguments ###
# template e.g. 'bundle_template.sh'
# startdir e.g. 71
# enddir e.g. 90
# Also requires: existing template file. 


def mk_bundle( startdir, enddir, update=False ):
    nruns = 1 + enddir - startdir

    if not os.path.exists("../bundle_scripts"):
        os.makedirs("../bundle_scripts")
        
    outfn = '../bundle_scripts/zstash_create.{}-{}.sh'.format(startdir,enddir)
    if update:
        outfn = '../bundle_scripts/zstash_update.{}-{}.sh'.format(startdir,enddir)
        
    
    outfile = open( outfn,"a")
        
    outfile.write('ROOT=/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/ \n')
    outfile.write('cd $ROOT \n')
    # Copy the following lines nruns times.
    for i in range(nruns):
        n = startdir + i
        if not update:
            l1 = f'zstash create --hpss=/home/w/wagmanbe/E3SMv3/dakota/v3_pmcpu/ne30_F2010/by_parts/ens/workdir.{n} workdir.{n}   2>&1 | tee zstash_create_wd.{n} \n'
        else:
            l1 = f'zstash update --hpss=/home/w/wagmanbe/E3SMv3/dakota/v3_pmcpu/ne30_F2010/by_parts/ens/workdir.{n}   2>&1 | tee zstash_update_wd.{n} \n' 
        outfile.write(l1)
    outfile.close()
    
    

#Real
# mk_bundle(1, 20 ) 
# mk_bundle(21, 40 ) 
# mk_bundle(41, 60 ) 
# mk_bundle(61, 80 ) 
# mk_bundle(81, 100 ) 
# mk_bundle(101, 120 ) 
# mk_bundle(121, 140 ) 
# mk_bundle(141, 160 ) 
# mk_bundle(161, 180 ) 
# mk_bundle(181, 200 ) 
# mk_bundle(201, 220 ) 
# mk_bundle(221, 240 ) 
# mk_bundle(241, 260 ) 
# mk_bundle(261, 280 ) 
# mk_bundle(281, 300 ) 
# mk_bundle(301, 320 ) 
# mk_bundle(321, 340 ) 
mk_bundle(341, 360, update=True ) 
mk_bundle(341, 360) 
#mk_bundle(361, 380 ) 
#mk_bundle(381, 400 ) 




