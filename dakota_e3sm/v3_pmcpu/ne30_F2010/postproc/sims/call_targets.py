# Benjamin Wagman 2023
# Works on zppy output.
# Remaps to 37 levels, takes zonal means of seasonal files.
# Creates and writes to a "targets" directory 

# depends on e3sm_unified_1.8.1

# 20231116
# Modified this script to run on all sets of experiments, e.g. ens, ctrl, validate, hm...
# This negates the need to keep multiple copies of this script, so I'll delete the others. 

import os
import pdb
import shutil
import xarray as xr
import numpy as np
import json
import time
import glob
from datetime import datetime

def mk_if_not_exist( path ):
    if not os.path.isdir( path ):
        os.mkdir( path )

def get_params( path_to_atm_in ):
    param_dic={}
    params_to_get=['clubb_c1','clubb_gamma_coef','zmconv_tau','zmconv_dmpdz','zmconv_micro_dcs','nucleate_ice_subgrid','p3_nc_autocon_expon','p3_qc_accret_expon','zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','cldfrc_dp1','p3_embry\
onic_rain_size','p3_mincdnc']
    for param in params_to_get:
        if os.path.isfile( path_to_atm_in ): 
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


# create a dictionary of rmse and write it to json. 
def rmse_func( d_in, d_out, path_to_merged_obs, grid ):
    print('comparing targets to observations')
    rmse_dic= {}
    grid = grid.split('_')[0] # Removes the _aave if it's attached to the grid name. 
    for sn in ['DJF','MAM','JJA','SON','ANN']:

        rmse_dic[sn]={}
        for obs_f in os.listdir( path_to_merged_obs ):
            if sn in obs_f and 'merged' in obs_f and grid in obs_f:
                obs_file = obs_f
                print(obs_f)
        for sigma_f in os.listdir( obs_temporal_sigma_dir ):
            if grid in sigma_f:
                if 'RESTOM' in sigma_f:
                    restom_sigma_file = sigma_f
                else:
                    sigma_file = sigma_f
                print( sigma_f)
        mod_file = ''
        for mod_f in os.listdir( d_in ):
            # Grid is specified in the d_in path so dont need to include it in the if statement. 
            if sn in mod_f and 'merged' in mod_f:
                mod_file = mod_f
                print(mod_f)
        if not len(mod_file)>0:
            print(f'no merged file found for {sn} in {d_in}. Cannot compute RMSE') 
        obs_ds   = xr.load_dataset(os.path.join( path_to_merged_obs, obs_file))
        sigma_ds = xr.load_dataset(os.path.join( obs_temporal_sigma_dir, sigma_file)).rename({'plev':'lev'})
        restom_sigma_ds = xr.load_dataset(os.path.join( obs_temporal_sigma_dir, restom_sigma_file))
        
        mod_ds   = xr.load_dataset(os.path.join( d_in, mod_file))
        # Use E3SM variable names. 
        if not 'ANN' in sn:
            for var in ['RELHUM','T','U','TREFHT','U200','U850','Z500','SWCF','LWCF','PRECT','PSL']:
                rmse_dic[sn][var]={}

                if var in mod_ds:
                    model_field = mod_ds[var]
                else:
                    print(f'quitting because cannot find model variable {var}')
                    quit()
                if var in obs_ds:
                    obs_field = obs_ds[var].squeeze()
                    obs_field_sigma = sigma_ds.sel(season=sn)[var]
                else:
                    print(f'quitting because cannot find obs variable {var}')
                    quit()
                    
                # # Check on nans. 
                # # Model has nans for some data on the vertical levels, presumably where some zonal means encountered high elevation. This is okay. 
                # if np.sum(np.isnan( model_field.values) ) > 0 :
                #     print('model has nans for ')
                #     print(var)
                #     print(sn)
 
                # if np.sum(np.isnan( obs_field.values) ) > 0 :
                #     print('obs has nans for ')
                #     print(var)
                #     print(sn)

                se = (model_field - obs_field )**2
                #se_norm = ((model_field - obs_field) / np.nanstd(obs_field))**2
                se_norm = ((model_field - obs_field) / (obs_field_sigma ))**2 
                
                if "lat" in model_field.dims and "lon" in model_field.dims:
                    se_weighted = se.weighted(mod_ds.area)
                    rmse = np.sqrt( se_weighted.mean(("lon","lat")) )
                    
                    se_norm_weighted = se_norm.weighted(mod_ds.area)
                    rmse_norm = np.sqrt( se_norm_weighted.mean(("lon","lat")) )

                if "lat" in model_field.dims and not "lon" in model_field.dims:
                    se_weighted = se.weighted(mod_ds['area'][:,0])
                    #rmse = np.sqrt( se_weighted.mean(("lat","lev"),skipna=True))  # This keeps the global mean from being Nan. We are still masking out underground beacuse model has nans. 
                    rmse = np.sqrt( se.mean(("lat","lev"),skipna=True))
                    
                    se_norm_weighted = se_norm.weighted( mod_ds['area'][:,0] )
                    rmse_norm = np.sqrt( se_norm_weighted.mean(("lat","lev"),skipna=True) )
                    #rmse_norm = np.sqrt( se_norm.mean(("lat","lev"),skipna=True) )

                if np.isnan (rmse.values):
                    print(f'quitting because there are nans in the rmse for {var}')
                    quit()
                    
                rmse_dic[sn][var]['no_norm']= rmse.values.tolist()
                rmse_dic[sn][var]['norm']= rmse_norm.values.tolist()

        # Open the annual mean file for RESTOM.
        if 'ANN' in sn:
            model_restom = mod_ds['FSNT'] - mod_ds['FLNT']
            restom_weighted = model_restom.weighted(mod_ds.area)
            restom_glb_ann_mean = restom_weighted.mean(("lon","lat"))
            restom_rmse = np.sqrt(( restom_glb_ann_mean - 0.7 ))**2

            rmse_dic['RESTOM']={}
            rmse_dic['RESTOM']['no_norm']= restom_rmse.values.tolist()

    # Write.
    # RMSE for each season and field.
    # Conclusion: this rmse and surrogate rmse match for lat-lon, but not for lat-plev.
    # They almost match if i dont are-weight lat-plev.
    # 
    # if '24' in grid:
    #     print('rmse JJA TREFHT 24x48 is ')
    #     print(rmse_dic['JJA']['TREFHT'] ) 
    #     print('Gavins for workdir.3 is 1.3611, 0.067' )

    #     print('rmse JJA  T 24x48 is ')
    #     print(rmse_dic['JJA']['T'] ) 
    #     print('Gavins for workdir.3 is 3.05, 0.104' )

    json_object = json.dumps(rmse_dic, indent=4)
    with open(os.path.join( d_out , f"rmse_{grid}.json"), "w") as outfile:
        outfile.write(json_object)

    # Get total RMSE by averaging over the seasons and fields. 
    sum_over_all_seasons = 0
    for sn in ['DJF','MAM','JJA','SON']:
        field_sum = 0
        field_count = 0
        sn_dict = rmse_dic[sn]
        for field in sn_dict.keys():
            field_sum += sn_dict[field]['norm']
            field_count += 1
        sn_mean = field_sum / float(field_count )
        #print(sn_mean)
        sum_over_all_seasons += sn_mean
    mean_over_seasons = sum_over_all_seasons / 4.0
    print('normalized total rmse ' )
    print(mean_over_seasons)
    with open(os.path.join( d_out ,f"rmse_total_{grid}.json"), "w") as outfile:
        outfile.write(str(mean_over_seasons))
    
    # Raw RESTOM value. 
    print(' RESTOM ' )
    print( restom_glb_ann_mean.values )
    with open(os.path.join( d_out ,f"RESTOM_raw_{grid}.json"), "w") as outfile:
        outfile.write(str(restom_glb_ann_mean.values))
        
def process_a_file( nc, d_in, d_out, param_dic, casename ): # does vertical remapping when necessary, zonal mean when necessary, divides files into lat x lon and lat x plev. 
    # 1. Write 2d lat-lon vars to their own file.
    # 2. Take zonal mean of plev x lat x lon files and write to own file.
    # 3 and 4. Create PRECT.
    
    fn = nc.split('climo')[0] # split the filename.
    plev_file_fullpath_out=os.path.join( d_out, fn+'lat_plev.nc')
    latlon_file_fullpath_out = os.path.join(d_out, fn+'lat_lon.nc')
    fdbk_file_fullpath_out = os.path.join(d_out, fn+'feedbacks_lat_lon.nc')

    # copy the climo file to the targets dir. 
    # Check which vars are available. Add PRECT if its not there.
    os.system('cp {} {}'.format(os.path.join(d_in, nc), os.path.join(d_out, nc)))
    ds = xr.open_dataset(os.path.join(d_out, nc))
    if not 'PRECT' in ds:
        ds['PRECT']=ds['PRECL']+ds['PRECC']
        ds['PRECT'] = ds['PRECT'].assign_attrs(units=ds['PRECC'].units)
        ds.load()
        ds.to_netcdf(os.path.join(d_out, nc),compute=True)
        ds.close()
    has_targ_vars = True
    targ_vars = ['TREFHT','Z500','RH500','T500','U850','U200','SWCF','LWCF','PSL','FLNT','FSNT','gw']
    lat_lon_2d_vars = 'TREFHT,Z500,RH500,T500,U850,U200,PRECT,SWCF,LWCF,PSL,FLNT,FSNT,gw'
    for v in targ_vars:
        if not v in ds.data_vars:
            has_targ_vars=False
            lat_lon_2d_vars = 'TREFHT,PRECT,SWCF,LWCF,PSL,FLNT,FSNT,gw'
            vars_to_get_slices = 'Z3,RELHUM,T,U'
    lat_plev_2d_vars = 'RELHUM,T,U,gw'
    cmd_to_plevs = 'ncremap -v {} --vrt_fl={} --vrt_xtr=mss_val --ps_nm=PS {} {}'.format( lat_plev_2d_vars,  'plevs/ERAI_L37.nc',os.path.join( d_out, nc ), plev_file_fullpath_out)     # Create the lat x plev file in targets
    cmd1 = 'ncks -O -v {} {} {}'.format(lat_lon_2d_vars, os.path.join(d_out, nc ), latlon_file_fullpath_out)  # Create the lat-lon file in targets. 
    cmd2 = 'ncwa -O -a lon -v {} {} {}'.format(lat_plev_2d_vars, plev_file_fullpath_out, plev_file_fullpath_out )  # Take the zonal mean of the lat x plev file in targets.
    os.system(cmd1)
    os.system(cmd_to_plevs)
    os.system(cmd2)
    if not has_targ_vars:
        cmd_to_plevs = 'ncremap -v {} --vrt_fl={} --vrt_xtr=mss_val --ps_nm=PS {} {}'.format( vars_to_get_slices,  'plevs/ERAI_L37.nc',os.path.join( d_out, nc ), os.path.join( d_out, 'tmp.nc'))
        os.system(cmd_to_plevs) # Create a temporary file with only the vars i need to interpolate to plevs. 
        ds_tmp = xr.open_dataset( os.path.join( d_out, 'tmp.nc')) 
        if ('plev' in ds_tmp.dims) and (not 'lev' in ds_tmp.dims):
        #     ds_tmp_path = os.path.join( d_out, 'tmp.nc')
        #     ds_tmp.close()
        #    # RESUME HERE. Rename plev dim and coordinate to lev
        #     pdb.set_trace()

        #    cmd_rename = f'ncrename  -d plev,lev  {ds_tmp_path}'
            # os.system( cmd_rename )
            # ds_tmp_new = xr.open_dataset(ds_tmp_path)
            # ds_tmp_new['lev']=ds_tmp.plev.values
            # ds_tmp = ds_tmp_new
            # ds_tmp_new.close()
            ds_tmp = ds_tmp.rename({'plev':'lev'})
        ds = xr.open_dataset( latlon_file_fullpath_out)
        ds['Z500']=ds_tmp['Z3'].sel(lev=5e4)
        ds['RH500']=ds_tmp['RELHUM'].sel(lev=5e4)
        ds['T500']=ds_tmp['T'].sel(lev=5e4)
        ds['U850']=ds_tmp['U'].sel(lev=8.5e4)
        ds['U200']=ds_tmp['U'].sel(lev=2e4)
        ds.load()
        ds.to_netcdf(latlon_file_fullpath_out + '_tmp.nc',compute=True)
        ds.close()
        os.system( 'mv {} {}'.format( latlon_file_fullpath_out + '_tmp.nc', latlon_file_fullpath_out) )
        os.system( 'rm {}'.format(os.path.join( d_out, '*tmp.nc')))

    # Now remove time from the file. I think this might help with strange behavior I encounter later in script. 
    cmd5 = f'ncwa -O -a time {plev_file_fullpath_out} {plev_file_fullpath_out}'
    cmd6 = f'ncwa -O -a time {latlon_file_fullpath_out} {latlon_file_fullpath_out}'
    cmd7 = f'ncks -O -C -x -v time {plev_file_fullpath_out} {plev_file_fullpath_out}'
    cmd8 = f'ncks -O -C -x -v time {latlon_file_fullpath_out} {latlon_file_fullpath_out}'
    os.system(cmd5)
    os.system(cmd6)
    os.system(cmd7)
    os.system(cmd8)
    
    # Option to append Feedback variables from Bryce Harrop's analysis in
    # /pscratch/sd/b/beharrop/E3SMv3/ppe_ens/workdir.*/diag_feedback_E3SM_postdata
    # Append the annual mean feedback data from bryce to each season. So, each season has identical feedback data. 
    do_feedback=True
    if 'WCYCL20TR' in d_in or 'piControl' in d_in:
        do_feedback=False  # we don't have +4k simulations for coupled runs. 
    if do_feedback:
        feedback_vars = 'dnet_cld_dir,SWCRE_ano_grd_adj,LWCRE_ano_grd_adj'
        feedback_fname = 'lat-lon-gfdbk-CMIP6-v3PPE.nc'
        if 'workdir.' in d_in:
            wd = d_in.split('workdir.')[1].split('/')[0]
            fdbk_file = os.path.join('/pscratch/sd/b/beharrop/E3SMv3/ppe_ens',f'workdir.{wd}','diag_feedback_E3SM_postdata/data/lat-lon-gfdbk-CMIP6-v3PPE.nc')

        if 'hm' in d_in:
            # First get the hm number.
            # Then get the p3_mincdnc value
            wd = d_in.split('hm/')[1].split('/')[0]
            hm_number = wd.split('_')[2]
            p3_value =  wd.split('_')[-1]
            fdbk_file = os.path.join('/pscratch/sd/b/beharrop/E3SMv3/ppe_ens/next',f'diag_feedback_E3SM_postdata_{hm_number}_{p3_value}','data/lat-lon-gfdbk-CMIP6-v3PPEnext.nc')

        if 'validate' in d_in:
            fdbk_file = os.path.join('/pscratch/sd/b/beharrop/E3SMv3/alt_candidates/v3a02/',f'diag_feedback_E3SM_postdata_{casename}', f'data/lat-lon-gfdbk-CMIP6-{casename}.nc')
            if not os.path.isfile( fdbk_file):
                fdbk_file = os.path.join( '/pscratch/sd/w/wagmanbe/diag_feedbacks/validate/diag_feedback_E3SM_postdata/',f'{casename}', f'data/lat-lon-gfdbk-CMIP6-{casename}.nc')
            
        if 'ctrl' in d_in:
            wd = 'ctrl'
            fdbk_file = os.path.join('/pscratch/sd/b/beharrop/E3SMv3/ppe_ens',f'workdir.{wd}','diag_feedback_E3SM_postdata/data/lat-lon-gfdbk-CMIP6-v3PPE.nc')

            
        
        if os.path.exists(fdbk_file):
            print(f'found feedback file')
            target_ds = xr.open_dataset(latlon_file_fullpath_out)
            test_cmd = f'ncdump -v FLNT {latlon_file_fullpath_out}'
            #os.system(test_cmd)
                
            # Let xarray do the inteprolation to target grid. Crude but easy. Bryce's data is on 73x144
            fdbk_ds = xr.open_dataset(fdbk_file)
            fdbks_only = xr.Dataset({})
            fdbks_only['dnet_cld_dir']= fdbk_ds['dnet_cld_dir'].interp_like(target_ds, kwargs={"fill_value":"extrapolate"})
            fdbks_only['SWCRE_ano_grd_adj']= fdbk_ds['SWCRE_ano_grd_adj'].interp_like(target_ds, kwargs={"fill_value":"extrapolate"})
            fdbks_only['LWCRE_ano_grd_adj']= fdbk_ds['LWCRE_ano_grd_adj'].interp_like(target_ds, kwargs={"fill_value":"extrapolate"})
            test_cmd = f'ncdump -v FLNT {latlon_file_fullpath_out}'
            #print('netcdf file values')
            #os.system(test_cmd) # Prints the correct data? Not anymore. Now its all 0's. What is going on? 
            test_fdbk = f'ncdump -v LWCRE_ano_grid_adj {fdbk_file_fullpath_out}'
            #os.system(test_fdbk)
            fdbks_only.to_netcdf(path = fdbk_file_fullpath_out)


    ## Create one file instead of separarting lat-lon from lat-plev. Nco failing me so using xarray.
    if os.path.isfile(os.path.join(d_out, f'{fn}lat_plev.nc')):
        if os.path.isfile(  latlon_file_fullpath_out ):
            dslatlev=xr.open_dataset(os.path.join(d_out, f'{fn}lat_plev.nc'))
            dslatlev=dslatlev.drop(['area','lon','lon_bnds'])
            test_cmd = f'ncdump -v FLNT {latlon_file_fullpath_out}'
            print('netcdf file values')
            #os.system(test_cmd) # All zero if I don't use the sleep command. sleep command fixes it.
            dslatlon=xr.open_dataset(latlon_file_fullpath_out , decode_times=False).load()  # SOLVED "opening old file" problem. Must always do ds.close() when you're done. 
            dsmerged=xr.merge([dslatlev, dslatlon],compat='override')
            # IF the feedback only file exists, add it to the merged.
            if not 'fdbks_only' in locals(): # if this dataset is loaded OR If the local file exists.
                try:
                    fdbks_only = xr.load_dataset(os.path.join( d_out, fn + 'feedbacks_lat_lon.nc'))
                except:
                    print('could not find any feedback data for case' )
            if 'fdbks_only' in locals(): # 
                    print('adding feedbacks to merged file')
                    dsmerged = xr.merge([dsmerged, fdbks_only])
            # If 'plev' in dataset, change name to lev'
            if 'plev' in dsmerged.dims:
                if 'lev' in dsmerged.coords:
                    dsmerged = dsmerged.drop('lev') # IF this is present it should be singular (just one lev) and okay to drop. 
                dsmerged = dsmerged.rename({'plev':'lev'})
            merged_outfn = os.path.join( d_out, fn + 'merged.nc' )
            dsmerged.to_netcdf(path=merged_outfn)
            print('combined lat_lon and lat_plev files into a merged file')
            # Include parameter values in the netcdf
            dsmerged.attrs.update( param_dic)
            dsmerged.to_netcdf(path=merged_outfn)
            if 'dnet_cld_dir' in dsmerged.data_vars and 'ANN' in merged_outfn and '180x360' in merged_outfn and not '1yr' in merged_outfn:
                    dstamp = datetime.today().strftime("%Y-%m-%d")
                    cmd = f'echo {casename} >> sims_with_fdbk_{dstamp}.txt'
                    os.system(cmd)

            
def targets( casename, workdirpath, clobber=False, rmse=False): # for a particular workdir, loops through files, and calls process_a_file for each file.
    mk_if_not_exist( os.path.join( workdirpath, casename,'targets' ))
    mk_if_not_exist( os.path.join( workdirpath, casename, 'targets','atm' ))
    if not os.path.isdir( os.path.join(  workdirpath, casename, 'post','atm')):
        print(f'no atm postprocessed data for {workdirpath}/{casename}')
    else:
        # Check if 180x360 exists but 24x48 does not. This is often the case with runs not done by BMW.
        d = os.path.join(  workdirpath, casename, 'post','atm') 
        in_d = os.path.join( d, '180x360_aave')
        out_d = os.path.join(  d, '24x48_aave')
        # Remap each climo to 24x48 if not already exist. 
        if '180x360_aave' in os.listdir( d ):
            if (not os.path.isdir( out_d )) or (not os.path.isdir( os.path.join( out_d, 'clim' ))):
                print('found 180x360 postproc data but not 24x48. Remapping now')
                mk_if_not_exist(os.path.join( out_d))
                mk_if_not_exist(os.path.join( out_d, 'clim' ))
                if not os.listdir(os.path.join( out_d, 'clim' )):
                    for clim in os.listdir(os.path.join(in_d, 'clim')):
                        out_clim = os.path.join(out_d, 'clim', clim) 
                        in_clim = os.path.join( in_d, 'clim', clim)
                        mk_if_not_exist(out_clim)
                        if not os.listdir(out_clim):
                            mf = '../180x360_to_24x48/map_180x360_to_24x48.nc'
                            cmd = f'ncremap -O {out_clim} -m {mf} -I {in_clim}'
                            os.system(cmd)

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
                                        process_a_file(nc, d_in, d_out, param_dic , casename)
                        if rmse:
                            target_dir = os.path.join( workdirpath, casename, 'targets','atm', grid, nyears )
                            if os.path.isdir( target_dir ):
                                rmse_func( target_dir , target_dir , obs_merged_input_dir, grid)
                            
                else:
                    print(f'no climo data for {workdirpath} {casename} + ' ' {grid}, so cant make targets')
                    

# Path to merged obs netcdf files if rmse=True
obs_merged_input_dir = '/global/cfs/cdirs/e3sm/emulate/obs/targets_v3/merged/'
# Path to obs standard deviation file for rmse.
obs_temporal_sigma_dir = '/global/cfs/cdirs/e3sm/emulate/sigma_temporal/'


def targets_loop( ens_root , subset_of_workdirs=[] ): # optional subset of workdirs if you only want to do some. Subset of workdirs is a list [start, end ]
    list_of_workdirs = sorted(os.listdir( ens_root ))
    list_of_workdirs = [d for d in list_of_workdirs if os.path.isdir( os.path.join( ens_root, d))]
    list_of_workdirs = [d for d in list_of_workdirs if not 'zstash' in d]
    if len(subset_of_workdirs) > 0:
        ss = np.arange( subset_of_workdirs[0], subset_of_workdirs[1]+1)
        list_of_workdirs = [wd for wd in list_of_workdirs if any( str(s) in wd for s in ss )]
    for d in list_of_workdirs:
        if 'workdir' in d:  # This works for looping over the ensemble in which simulations are embedded inside 'workdirX'
            ens_root_expanded=os.path.join( ens_root, d )
            for dd in sorted(os.listdir( ens_root_expanded )):
                if os.path.isdir( os.path.join(ens_root_expanded, dd)) and os.path.exists( os.path.join(ens_root_expanded, dd, 'archive')):
                    targets( dd, ens_root_expanded, clobber=True, rmse=False) 
        else:
            targets( d,  ens_root, clobber=True, rmse=False) # This works for other types of ensembles, e.g. casenames inside one directory. 

                        
if __name__ == "__main__":
    targets_loop( '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/WCYCL20TR/') # Bryce's coupled runs. 
    #targets('v3alt.LR.highECS003.historical','/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/WCYCL20TR/')
    #targets_loop('/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/piControl/' )
    #targets( '20230802.v3alpha02.F2010.pmcpu.intel.8N', '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.293',clobber=True)      #re-run on PPE 293 to get diag_feedback data onto merged file. This now works even thorugh bryce purged his, since I have feedbcaks saved separately in the targets dir for most cases.   

    #targets_loop ( '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/')  # E.g. all ensemble members 
    #targets_loop ( '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/', subset_of_workdirs= [001, 005 ])  
    #targets( '20230802.v3alpha02.F2010.pmcpu.intel.8N','/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ctrl/',clobber=True, rmse=True) # Just the control. 
    #targets_loop( '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/hm/')        # All hm experiments.
    

    
    
    
