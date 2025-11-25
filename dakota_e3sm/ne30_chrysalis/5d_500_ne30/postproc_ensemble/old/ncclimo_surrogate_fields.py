# Benjamin Wagman 3-29-2021
# Different from the other calls to ncclimo is  removed the --no-amwg-link flag.
# This allows the e3sm diags to use it as a refcase. 

import os
import pdb
import shutil


def mk_if_not_exist( path ):
    if not os.path.isdir( path ):
        os.mkdir( path )

def call_ncclimo_sur(start,end, case, rundir, destpath, vlist, mapfile_dict, allvarclimo):
    for m in mapfile_dict:
        mapfile = mapfile_dict[m]
        mk_if_not_exist( os.path.join( destpath, m )) 
        ncclimo( start, end, case, rundir, os.path.join( destpath, m ), vlist, mapfile , allvarclimo)
        # Write a subset of vars to ERA-I plev
        dest_mod_to_obs  = os.path.join(destpath, m, 'target')
        mod_to_obs_plevs( vlist, case, os.path.join( destpath,m), dest_mod_to_obs)
        targets_only( case, dest_mod_to_obs) 
        
def ncclimo(startyr, endyr, casename, rundir, destpath, vlist, mapfile, allvarclimo):
    # Remove any preexisting climo files.
    for old_f in os.listdir( destpath ):
        if old_f.endswith("_aavg.nc"):
            os.remove( os.path.join( destpath, old_f))
    
    cmd = 'ncclimo -s {} -e {} -v {} -c {} -i {} -o {} -r {} -a sdd -m eam'.format(startyr, endyr, vlist, casename,  rundir, destpath, mapfile)
    if allvarclimo: 
        cmd = 'ncclimo -s {} -e {} -c {} -i {} -o {} -r {} -a sdd -m eam'.format(startyr, endyr, casename,  rundir, destpath, mapfile) # If allvarclimo don't include the -v flag. 
    print(cmd)
    os.system(cmd)

    # Clean up. 
    # find the .nc files in the destpath that are not remapped. Move them into a subdir to hide them from e3sm diags.
    # Do the same with monthly data. Probably no reason to keep this. 
    mk_if_not_exist( os.path.join( destpath, 'unstruc'))
    mk_if_not_exist( os.path.join( destpath, 'monthly'))
    for f in os.listdir( destpath ):
        if 'nc' in f:
            if not ('fv129' in f or '24x48' in f):
                shutil.move( os.path.join( destpath, f ), os.path.join( destpath, 'unstruc', f ))
    # Move the monthly files because we will not be using them. Can delete to save space. 
    for f in os.listdir( destpath):
        if 'nc' in f:
            if not ('DJF' in f or 'MAM' in f or 'JJA' in f or 'SON'	in f or 'ANN' in f ):
                shutil.move( os.path.join( destpath, f ), os.path.join( destpath, 'monthly', f ))
    # Delete the monthly and monthly unstructured files.
    for f in os.listdir( os.path.join( destpath,'monthly')):
        os.remove( os.path.join( destpath,'monthly',f))
    os.rmdir( os.path.join( destpath,'monthly'))
    for f in os.listdir( os.path.join( destpath, 'unstruc')):
        if not ('DJF' in f or 'MAM' in f or 'JJA' in f or 'SON' in f or 'ANN' in f ):
            os.remove( os.path.join( destpath,'unstruc',f))
        
def mod_to_obs_plevs( vlist, casename, sourcepath, destpath): # Interpolate seasonal  model to obs plevs and include only evaluated fields. 
    #os.system('./mk_ERAI_plevs.sh') # Make the pressure level axis 'ERAI_L37.nc' # Already made this on NERSC. 
    # append 'gw' to the variables list.
    vlist += ',gw'
    mk_if_not_exist( destpath )
    for f in os.listdir( sourcepath ):
        if ('DJF' in f or 'MAM' in f or 'JJA' in f or 'SON' in f ):
            if ( 'fv' in f or 'aavg' in f):
                cmd = 'ncremap -v {} -i {} -O {} --vrt_fl={} --vrt_xtr=mss_val'.format( vlist, os.path.join( sourcepath, f ), os.path.join( destpath), 'ERAI_L37.nc')
                os.system( cmd )  
                # E.g. 'ncremap -v TREFHT,PRECC,PRECL,SWCF,LWCF,PSL,FLNT,FSNT,Z3,U,V,RELHUM,T -i /global/cscratch1/sd/wagmanbe/e3sm_climo/s2n/noise/default.E3SM.ctrl_branch.ne4pg2_ne4pg2/default.E3SM.ctrl_branch.ne4pg2_ne4pg2_JJA_001106_009008_climo_24x48_aavg.nc -O /global/cscratch1/sd/wagmanbe/e3sm_climo/s2n/noise/default.E3SM.ctrl_branch.ne4pg2_ne4pg2/target --vrt_fl=ERAI_L37.nc --vrt_xtr=mss_val'

# Write a 2-D lat-lon file and a 2-D pressure vs zonal-mean file. 
def targets_only( casename, sourcepath): 
    for f in os.listdir( sourcepath):
        if (not 'tmp' in f) and ( f.endswith('fv129x256_nco.nc') or f.endswith('24x48_aavg.nc') ):
            dest = sourcepath
            # extract 2d lat-lon vars and put them in their own target file.  a slice of U at 800 and 250 hPa, and Z at 500 hPa
            lat_lon_2d_vars = 'TREFHT,Z500,RH500,T500,U850,U200,PRECC,PRECL,SWCF,LWCF,PSL,FLNT,FSNT,gw'
            lat_plev_2d_vars = 'RELHUM,T,U,gw'
            # 1. Write 2d lat-lon vars to their own file.
            # 2. Take zonal mean of plev x lat x lon files and write to own file.
            # 3 and 4. Create PRECT.
            #
            fn = f.split('climo')[0] # split the filename. 
            cmd1 = 'ncks -O -v {} {} {}'.format(lat_lon_2d_vars, os.path.join( sourcepath, f ), os.path.join( dest, fn+'lat_lon.nc'))
            cmd2 = 'ncwa -O -a lon -v {} {} {}'.format(lat_plev_2d_vars, os.path.join( sourcepath, f ), os.path.join( dest, fn+'lat_plev.nc'))
            cmd3 = 'ncap2 -O -s "PRECT=PRECC+PRECL" {} {}'.format(os.path.join( dest, fn+'lat_lon.nc'),os.path.join( dest, fn+'lat_lon.nc'))
            cmd4 = 'ncatted -O -a long_name,PRECT,m,c,"Total precipitation rate (PRECC+PRECL)" {}'.format(os.path.join( dest, fn+'lat_lon.nc'))
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)
            os.system(cmd4)
            
            


