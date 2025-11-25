# Benjamin Wagman 20210127
# Works on zppy output.
# Remaps to 37 levels, takes zonal means of seasonal files.
# Creates and writes to a "targets" directory 

import os
import pdb
import shutil
import xarray as xr

def mk_if_not_exist( path ):
    if not os.path.isdir( path ):
        os.mkdir( path )


def get_params( path_to_atm_in ):
    param_dic={}
    params_to_get=['clubb_c1','clubb_gamma_coef','zmconv_tau','zmconv_dmpdz','zmconv_micro_dcs','nucleate_ice_subgrid','p3_nc_autocon_expon','p3_qc_accret_expon','zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','cldfrc_dp1','p3_embryonic_rain_size','p3_mincdnc']
    for param in params_to_get:
        infile = open(path_to_atm_in,'r')
        list_of_lines = infile.readlines()
        infile.close()
        for i in range(len(list_of_lines)):
            if param in list_of_lines[i]:
                value_str = list_of_lines[i].split("=",1)[1]
                value_str = value_str.strip('\n')
                if 'D' in value_str:
                    value_str = value_str.replace('D','e')
                param_dic[param]= float( value_str )
    return param_dic 
                
def process_a_file( nc, d_in, d_out, param_dic ): # does vertical remapping when necessary, zonal mean when necessary, divides files into lat x lon and lat x plev. 
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
    # Create one file instead of separarting lat-lon from lat-plev. Nco failing me so using xarray.
    if os.path.isfile(os.path.join(d_out, f'{fn}lat_plev.nc')):
        if os.path.isfile(os.path.join(d_out, f'{fn}lat_lon.nc')):
            dslatlev=xr.open_dataset(os.path.join(d_out, f'{fn}lat_plev.nc'))
            dslatlev=dslatlev.drop(['area','lon','lon_bnds'])
            dslatlon=xr.open_dataset(os.path.join(d_out, f'{fn}lat_lon.nc'))
            dsmerged=xr.merge([dslatlev, dslatlon])
            merged_outfn = os.path.join( d_out, fn + 'merged.nc' )
            print('combined lat_lon and lat_plev files into a merged file')
            # Include parameter values in the netcdf
            dsmerged.attrs.update( param_dic)
            dsmerged.to_netcdf(path=merged_outfn)


def targets( casename, workdirpath, clobber=False): # for a particular workdir, loops through files, and calls process_a_file for each file. 
    mk_if_not_exist( os.path.join( workdirpath, casename,'targets' ))
    mk_if_not_exist( os.path.join( workdirpath, casename, 'targets','atm' ))
    if not os.path.isdir( os.path.join(  workdirpath, casename, 'post','atm')):
        print(f'no atm postprocessed data for {workdirpath}/{casename}')
    else:
        for grid in os.listdir( os.path.join(  workdirpath, casename, 'post','atm')):
            if (not 'tmp' in grid) and ( '24x48' in grid) or ('180x360' in grid) :
                mk_if_not_exist( os.path.join( workdirpath, casename, 'targets','atm', grid ))
                if os.path.isdir(os.path.join(  workdirpath, casename, 'post','atm',grid,'clim')):
                    param_dic = get_params( os.path.join(  workdirpath, casename, 'run', 'atm_in' ))
                    for nyears in os.listdir( os.path.join(  workdirpath, casename, 'post','atm',grid,'clim')):
                        d_in = os.path.join(  workdirpath, casename, 'post','atm',grid,'clim', nyears)
                        d_out =  os.path.join( workdirpath, casename, 'targets','atm', grid, nyears )
                        do_process = True
                        if os.path.isdir( d_in ):
                            if not clobber:
                                if os.path.isdir( d_out ):
                                    do_process=False
                                    print(f'{d_out} already exists')
                            if do_process:
                                mk_if_not_exist( d_out )
                                for nc in os.listdir( d_in ):
                                    if ('DJF' in nc or 'MAM' in nc or 'JJA' in nc or 'SON' in nc or 'ANN' in nc ):
                                        process_a_file(nc, d_in, d_out, param_dic  )
                            
                else:
                    print(f'no climo data for {workdirpath} {casename} + ' ' {grid}, so cant make targets')
                
def targets_loop( ens_root ): #Unlike the version of the script for the  ensemble, we don't provide a casename here, because they differ
    for wd in sorted(os.listdir( ens_root )):
        targets( wd,  ens_root , clobber=True)

                        
if __name__ == "__main__":
    targets( '20230914.v3alpha04.F2010.pmcpu.intel.8N', '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ctrl/', clobber=True)  # E.g. just one ens. 

    #targets_loop( '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/validate' )      # E.g. all validation cases. Unlike ensemble, we don't provide a casename here, because they differ. 
    
    
    
