from ncclimo_surrogate_fields import call_ncclimo_sur
import pdb
import os
import glob
from shutil import copyfile

# SET CASE
lhscase = '5d_chr_500_ne30'

print(os.environ["SCRATCH"]) # This must be exported by your shell to read it in python.
scratchh = os.environ["SCRATCH"]
# E.g. in your .bash_profile, write "export SCRATCH=/lcrc/group/e3sm/ac.wagman/scratch" 
# Doing it this way enables machine independence. 
clobber = False # Write over existing climos or not.
allvarclimo = True # Set to False to only compute climos from the vlist below. You won't be able to run diagnostics.
vlist = 'Z500,RH500,T500,U850,U200,TREFHT,PRECC,PRECL,SWCF,LWCF,PSL,FLNT,FSNT,Z3,U,V,RELHUM,T' # Subset for regridding to pressure levels and making targets. 
#yrlist = [2,4,6,8,10] # Run a 2, 4, 6, and 8 yr climo for each workdir.
yrlist = [6,10] # Run a 2, 4, 6, and 8 yr climo for each workdir.
#yrlist = [2] # Run a 2, 4, 6, and 8 yr climo for each workdir.
yrstart = 11
mapfiles_dict = {'24x48': 'map_ne30pg2_to_24x48_aavg.20211130.nc',
                 '129x256':'map_ne30pg2_to_fv129x256_nco.20200901.nc'}
runtype='ens' # options are 'ens','ctrl' # ctrl runs on the control. ens loops over the root directory for the ensemble. 
lhsroot = os.path.join( scratchh,'E3SM_simulations', 'dakota', lhscase, runtype)
print('LHS root is {}'.format(lhsroot))
case = '20210813.F2010.ne30pg2_oECv3.chrysalis'
testing=False
if testing:
    max_i=1
else:
    max_i=1e6
#specify_wd = 'workdir.186' # Either leave empty e.g. '' or write the name of a workdir e.g. 'workdir.194'
specify_wd = ''
i=0
#max_i=1e6 # Set to 1 for debugging or to run on command line. 

destroot = os.path.join( scratchh, 'e3sm_climo','lhs',lhscase, runtype )

print('destination root is {}'.format(destroot))

    
def grep_for_success_gz( statusfile ):
    complete=False
    result = os.system( 'zgrep "SUCCESSFUL TERMINATION OF CPL7" {}'.format(statusfile))
    print(result)
    if (result==0):
        complete=True
    return( complete )
        
def mk_if_not_exist( path ):
    if not os.path.isdir( path ):
        os.mkdir( path )


for wd in os.listdir( lhsroot ):
    if specify_wd:
        wd = specify_wd
        max_i=1
        print('wd specified as {}'.format(specify_wd))
    print('workdir is {}'.format(wd))
    mk_if_not_exist( destroot )
    paramsfile=glob.glob(os.path.join( lhsroot, wd, 'params.in*'))
    if paramsfile:
        params = paramsfile[0]
    if runtype=='ens':
        cpl_log_file = glob.glob(os.path.join( lhsroot, wd, case, 'run', 'cpl.log*.gz'))
    if runtype=='ctrl':
        cpl_log_file = glob.glob(os.path.join( lhsroot,case, 'run', 'cpl.log*.gz'))

    if cpl_log_file:
        completeness = grep_for_success_gz( cpl_log_file[0])
        #completeness = os.system( 'zgrep cpl.log*.gz')
        print('Passes cpl log completeness grep test? {}'.format(completeness ))
        if completeness:
            mk_if_not_exist( os.path.join( destroot, wd ))
            if runtype=='ens':
                rundir = os.path.join( lhsroot, wd, case, 'run')
            if runtype=='ctrl':
                rundir = os.path.join( lhsroot, case, 'run')
            for yr in yrlist:
                doclimo = True
                if runtype=='ens':
                    destpath =  os.path.join(destroot, wd, str(yr) )
                if runtype=='ctrl':
                    destpath =  os.path.join(destroot, case, str(yr) )
                mk_if_not_exist( destpath )
                if paramsfile:
                    copyfile( params, os.path.join( destroot, wd,'params.in'))
                if os.path.isdir( os.path.join(destpath,'unstruc') ):
                    dircontents = os.listdir(os.path.join(destpath,'unstruc'))
                    if len(dircontents)> 0:
                        print("{} is non-empty destination".format(destpath))
                        if not clobber:
                            doclimo=False
                            print('not clobbering existing climo. set clobber to True if clobber desired')
                if doclimo:
                    call_ncclimo_sur( yrstart, yrstart + yr -1, case, rundir, destpath, vlist, mapfiles_dict, allvarclimo   )
            i+=1
            print(i)
            if i >= max_i:
                exit()








