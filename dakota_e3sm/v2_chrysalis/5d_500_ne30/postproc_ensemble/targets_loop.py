# Benjamin Wagman 20210127
# Works on zppy output.
# Remaps to 37 levels, takes zonal means of seasonal files.
# Creates and writes to a "targets" directory 

# depends: conda activate /lcrc/soft/climate/e3sm-unified/base/envs/e3sm_unified_1.5.1_chrysalis

import os
import pdb
import shutil


def mk_if_not_exist( path ):
    if not os.path.isdir( path ):
        os.mkdir( path )


def process_a_file( nc, d_in, d_out ): # does vertical remapping when necessary, zonal mean when necessary, divides files into lat x lon and lat x plev. 
    # 1. Write 2d lat-lon vars to their own file.
    # 2. Take zonal mean of plev x lat x lon files and write to own file.
    # 3 and 4. Create PRECT.
    lat_lon_2d_vars = 'TREFHT,Z500,RH500,T500,U850,U200,PRECC,PRECL,SWCF,LWCF,PSL,FLNT,FSNT,gw'
    lat_plev_2d_vars = 'RELHUM,T,U,gw'
    fn = nc.split('climo')[0] # split the filename.
    cmd1 = 'ncks -O -v {} {} {}'.format(lat_lon_2d_vars, os.path.join(d_in, nc ), os.path.join( d_out, fn +'lat_lon.nc'))
    # regrid the variables that are on levels to plevs.
    cmd_to_plevs = 'ncremap -v {} -i {} -O {} --vrt_fl={} --vrt_xtr=mss_val'.format( lat_plev_2d_vars, os.path.join( d_in, nc ), os.path.join( d_out ), 'plevs/ERAI_L37.nc')
    cmd2 = 'ncwa -O -a lon -v {} {} {}'.format(lat_plev_2d_vars, os.path.join( d_out, nc ), os.path.join( d_out, fn+'lat_plev.nc'))
    cmd3 = 'ncap2 -O -s "PRECT=PRECC+PRECL" {} {}'.format(os.path.join( d_out, fn+'lat_lon.nc'),os.path.join( d_out, fn+'lat_lon.nc'))
    cmd4 = 'ncatted -O -a long_name,PRECT,m,c,"Total precipitation rate (PRECC+PRECL)" {}'.format(os.path.join(d_out, fn+'lat_lon.nc'))
    os.system(cmd1)
    os.system(cmd_to_plevs)
    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)



def targets( casename, workdirpath): # for a particular workdir, loops through files, and calls process_a_file for each file. 
    mk_if_not_exist( os.path.join( workdirpath, casename, 'targets' ))
    mk_if_not_exist( os.path.join( workdirpath, casename, 'targets','atm' ))
    if os.path.isdir( os.path.join(  workdirpath, casename, 'post','atm')):
        for grid in os.listdir( os.path.join(  workdirpath, casename, 'post','atm')):
            if (not 'tmp' in grid) and ( '24x48' in grid) or ('180x360' in grid) :
                mk_if_not_exist( os.path.join( workdirpath, casename, 'targets','atm', grid ))
                for nyears in os.listdir( os.path.join(  workdirpath, casename, 'post','atm',grid,'clim')):
                    d_in = os.path.join(  workdirpath, casename, 'post','atm',grid,'clim', nyears)
                    d_out =  os.path.join( workdirpath, casename, 'targets','atm', grid, nyears )
                    mk_if_not_exist( d_out )
                    for nc in os.listdir( d_in ):
                        if ('DJF' in nc or 'MAM' in nc or 'JJA' in nc or 'SON' in nc or 'ANN' in nc ):
                            process_a_file(nc, d_in, d_out )
       
def targets_loop( casename, ens_root ):
    for wd in os.listdir( ens_root ):
        targets( casename, os.path.join( ens_root, wd ))

                        
if __name__ == "__main__":
    #targets( '20210813.F2010.ne30pg2_oECv3_control.chrysalis', '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ctrl')  # E.g. just the control. 
    targets( '20210813.F2010.ne30pg2_oECv3.chrysalis', '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ens/workdir.78')  # E.g. just one ens. 
    #targets_loop( '20210813.F2010.ne30pg2_oECv3.chrysalis', '/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ens' )      # E.g. all ensemble members.
    
    
    
