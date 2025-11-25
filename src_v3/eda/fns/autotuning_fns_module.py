import os
import numpy as np
from cftime import num2date
from datetime import date, timedelta, datetime
from dateutil.relativedelta import *
import pandas as pd
import pdb
import xarray as xr
import glob
import shutil
import sys
import fnmatch
import random
from cartopy import crs
import copy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt

def fix_lon(lon):
    return np.where(lon > 180, lon - 360, lon)


def match_opt_cases():
    opt_cases = {'H001':'validate/validate.v3alpha02.2023102',
                 'H002':'validate/validate.opt_params_dnet-1.5_reweight_mincdnc12.5e6_20240423132337',
                 'H003':'dnet-1.5_RESTOM2.5',     
                 'L001':'ens/workdir.293/20230802.v3alpha02.F2010.pmcpu.intel.8N',
                 'L002':'dnet-2.1_RESTOM-0.5_rw2',
                 'L003':'dnet-2.1_RESTOM-0.5_highR2'}
    return opt_cases
    

def reshape( sur , all_var_dic, mask):
    # Reshape the surrogate to : 1 dataset for each season for each var organized wordir x var x lat x lon (data_rshp)
    # Retain data in vector form (data)
    # Inserts nans according to mask. 
    
    # Use the masks specific for each field and season.  
    # Surrogate is shape: (ens_idx: 408, var: 46693 ) <-- V3 24x48

    # The mask fits the surrogate. 

    data_rshp = copy.deepcopy(all_var_dic)
    data      = copy.deepcopy(all_var_dic)
    nvars = sum(all_var_dic.values(), [])
    if not len( nvars )==len( mask ):
         print(' stopping because length of variable list does not match length of mask' )
         pdb.set_trace()

    i=0 # For applying the mask to the right variable even though variables are concatenated into one vector.  
    mask_i = 0 # For selecting the right mask from the mask list. 
    for sn,list_of_vars in all_var_dic.items():
        vars_as_dict = dict( (v,{'mod':False,'sur':False,'mask':False}) for v in list_of_vars)
        data_rshp[sn]=copy.deepcopy(vars_as_dict)
        data[sn]=copy.deepcopy(vars_as_dict)
        for v in list_of_vars:
            
            mask_vec = mask[mask_i]
            sur_vec      = sur['surrogate_preds'][:, i : i + mask_vec.sum()]  # .sum() gives the number of True in the mask vector. 
            mod_vec      = sur['model_output'][:, i : i + mask_vec.sum()]

            # Insert the nans back into the mod_vec and sur_vec wherever the mask has a nan.
            sur_vec_nans, mod_vec_nans = np.empty((sur_vec.shape[0], len(mask_vec))), np.empty((sur_vec.shape[0], len(mask_vec)))
            sur_vec_nans[:] = np.nan
            mod_vec_nans[:] = np.nan
            i_real = np.where(mask_vec)[0]
            sur_vec_nans[:,i_real] = sur_vec
            mod_vec_nans[:,i_real] = mod_vec

            
            if len(mask_vec)==24*48:
                #print(f'{v} is shape 24x48')
                nlat, nother = 24, 48
                dims   = ['ens_idx', 'lat', 'lon']
                dims_no_rshp = ['ens_idx','horiz_idx']
            if len(mask_vec)==24*37:
                nlat, nother = 24, 37
                dims   = ['ens_idx', 'lev', 'lat']
                dims_no_rshp = ['ens_idx','vert_idx']
            if len(mask_vec)==1:
                nlat, nother = 1, 1
                dims   = ['ens_idx']
                dims_no_rshp = ['ens_idx']

            sur_rshp = xr.DataArray( np.reshape( sur_vec_nans, (len(sur_vec.ens_idx),nlat,nother)).squeeze() , dims=dims)
            mod_rshp = xr.DataArray( np.reshape( mod_vec_nans, (len(mod_vec.ens_idx),nlat,nother)).squeeze(), dims=dims)
            msk_rshp = np.reshape( mask_vec,       (nlat,nother)).squeeze()

            if nother==37: # Bugfix! Flattening worked differently in the surrogate model for lat x lev data. 
                sur_rshp = xr.DataArray( np.reshape( sur_vec_nans, (len(sur_vec.ens_idx),nother,nlat)).squeeze() , dims=dims)
                mod_rshp = xr.DataArray( np.reshape( mod_vec_nans, (len(mod_vec.ens_idx),nother,nlat)).squeeze(), dims=dims)
                msk_rshp = np.reshape( mask_vec,       (nother,nlat)).squeeze()
                
            sur_no_rshp = xr.DataArray(  sur_vec_nans.squeeze() , dims=dims_no_rshp)
            mod_no_rshp = xr.DataArray(  mod_vec_nans.squeeze(), dims=dims_no_rshp)
            msk_no_rshp = mask_vec.squeeze()

            # Attach workdir info, unmodified, from input. 
            sur_rshp['workdir']=sur.workdir
            mod_rshp['workdir']=sur.workdir
            
            data_rshp[sn][v]['sur'], data_rshp[sn][v]['mod'],data_rshp[sn][v]['mask'] = sur_rshp, mod_rshp, msk_rshp
            data[sn][v]['sur'], data[sn][v]['mod'],data[sn][v]['mask'] = sur_no_rshp, mod_no_rshp, msk_no_rshp

            
            i+= mask_vec.sum()
            mask_i+=1

    # For each product (sur, mod) combine across a new dim "product" and then merge all variables and seasons.
    # Duplicates annual mean scalars RESTOM and dnet across all seasons.
    # Destroys 'ANN' data for most fields but that can be recovered by averaging seasons. 
    seasonal_data_rshp, seasonal_data = [], []
    for sn,list_of_vars in all_var_dic.items():
        allvars_one_season_rshp, allvars_one_season = xr.Dataset(),xr.Dataset()
        list_of_dics_rshp, list_of_dics = [],[]
        for v in list_of_vars:
            # No bug here: data_rshp differs across seasons for a given ens member.
            conc_rshp = xr.concat( [data_rshp[sn][v]['mod'], data_rshp[sn][v]['sur']],dim='product').assign_coords({'product':['mod','sur']}).to_dataset(name=v)
            conc = xr.concat( [data[sn][v]['mod'], data[sn][v]['sur']],dim='product').assign_coords({'product':['mod','sur']}).to_dataset(name=v)
            list_of_dics_rshp.append(conc_rshp)
            list_of_dics.append(conc)
            allvars_one_season_rshp = xr.merge(list_of_dics_rshp,compat='override')
            allvars_one_season      = xr.merge(list_of_dics,compat='override')
        allvars_one_season_rshp = allvars_one_season_rshp.expand_dims({'time':[sn]}) 
        allvars_one_season      = allvars_one_season.expand_dims({'time':[sn]})
        seasonal_data_rshp.append(allvars_one_season_rshp) 
        seasonal_data.append(allvars_one_season)
    merged_seasons_and_vars_rshp = xr.merge(seasonal_data_rshp) 
    merged_seasons_and_vars = xr.merge(seasonal_data)


    # Restore 'ANN' values, which get zero'd out in the merge above.
    sn_only_rshp = merged_seasons_and_vars_rshp.isel(time = [1,2,3,4])
    an_only_rshp = sn_only_rshp.mean(dim='time', skipna=True).expand_dims({'time':['ANN']})
    merged_rshp = xr.merge( [an_only_rshp, sn_only_rshp])
    sn_only = merged_seasons_and_vars.isel(time = [1,2,3,4])
    an_only = sn_only.mean(dim='time', skipna=True).expand_dims({'time':['ANN']})
    merged = xr.merge( [an_only, sn_only])

    # Attach parameter data.
    rshp_with_pdata = xr.merge( [merged_rshp, sur['params']]) 
    with_pdata = xr.merge( [merged, sur['params']]) 


    # Restore scalar 'ANN' values, which were never defined seasonally
    for v in merged_rshp:
        if 'lat' not in merged_rshp[v].dims:
            for prod in merged_rshp[v].product:
                rshp_with_pdata[v][0, 0,:] = data_rshp['ANN'][v]['mod'].data
                rshp_with_pdata[v][0,-1,:] = data_rshp['ANN'][v]['sur'].data
                with_pdata[v][0, 0,:] = data['ANN'][v]['mod'].data
                with_pdata[v][0,-1,:] = data['ANN'][v]['sur'].data
    return rshp_with_pdata, with_pdata




# Attach feedbacks to the annual file. We only have annual feedbacks, but they are saved to each seasonal merged file identically. 
def attach_feedbacks(ds, list_of_feedback_files, merged_root):
    list_of_sw=[]; list_of_lw=[]
    opt = match_opt_cases()
    for wd in ds.workdir:
        result = []
        substr = str(wd.values)
        print(substr)
        for s in list_of_feedback_files:
            if (substr in s) and ('ANN' in s):
                result.append(s)
            if not result:
                # Match the original name to the updated name for H00X and L00X. Feedback files retain old name. 
                if substr in opt.keys() and ('ANN' in s):
                    for f in list_of_feedback_files:
                        if opt[substr] in f and ('ANN' in f) :
                            result.append(f)
        if result:
            sw = xr.open_dataset( result[0] )['SWCRE_ano_grd_adj']
            lw = xr.open_dataset( result[0] )['LWCRE_ano_grd_adj']
            sw_exp = sw.expand_dims({'product':['mod'],'time':['ANN'],'ens_idx':[substr]})
            lw_exp = lw.expand_dims({'product':['mod'],'time':['ANN'],'ens_idx':[substr]})
            list_of_sw.append( sw_exp)
            list_of_lw.append( lw_exp)
        else:
            # fill with nans.
            print(f'could not find feedback data for {substr}')

    data_sw = xr.concat(list_of_sw, 'ens_idx')
    data_lw = xr.concat(list_of_lw, 'ens_idx')
    all_fbk = xr.merge( [data_sw, data_lw]) # Confirmed has different values for each ens_idx, but len(ens_idx=426)
    ds.update(all_fbk) 
    print('finished attach feedbacks')
    return ds


def attach_obs(model, obs): # expects a dataset for model and a dictionary for obs.

    # create one dataset from the 4 datasets in obs.
    obs_ds = xr.concat([obs['DJF'],obs['MAM'],obs['JJA'],obs['SON']],dim='time')
    obs_ds = obs_ds.assign_coords({'time':['DJF','MAM','JJA','SON']})
    obs_an_only = obs_ds.mean(dim='time', skipna=True).expand_dims({'time':['ANN']})
    obs_ds_expanded = xr.merge( [obs_ds, obs_an_only])
    obs_ds_expanded = obs_ds_expanded.expand_dims({'product':['obs']})
    # Create dummy RESTOM and dnet data in obs. 
    obs_ds_expanded['RESTOM']       = xr.DataArray( np.full((len(obs_ds_expanded["time"]), len(obs_ds_expanded["product"])), np.nan), dims=('time','product') )
    obs_ds_expanded['dnet_cld_dir'] = xr.DataArray( np.full((len(obs_ds_expanded["time"]), len(obs_ds_expanded["product"])), np.nan), dims=('time','product') )

    # Assign the lat and lon values from obs to the model so they will merge correctly.
    model_out      = copy.deepcopy(model)
    model_out.assign_coords( {'lat':obs_ds_expanded.lat })
    model_out.assign_coords( {'lon':obs_ds_expanded.lon })

    # Merge obs with model, surrogate, and return. 
    model_out_merged = xr.merge([model_out,obs_ds_expanded])
    # REmove unnecessary area dimensions. 
    model_out_merged['area'] = model_out_merged['area'].sel(product = 'obs', drop=True).sel(time='ANN',drop=True)
    
    return model_out_merged

    
def load_reshape_save(validation_key, clobber=False):

    # Returns                                                                                                                      
    # all_cases: dataset with obs, surrogate, and model on the same grid                                                           
    # all_cases_orig: dataset with model, surrogate on the original surrogate vector but with nans to ensure equal lengths         
    # obs: A dictionary of obs for each season, in case working with the obs in all_cases breaks.                                  

    # Behavior:                                                                                                                    
    # Will save netcdf files if do not exist, or if clobber=True.                                                                  
    # Will load from netcdf if exist and clobber=False                                                                             


    data_dic= {'H003':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-1.5_RESTOM2.5_10pc_20240529210909/validation/validation_H003_bugfix_20241021162937/output/',
          'root_local':'./surrogate_models/v3/surrogate_provenance_dnet-1.5_RESTOM2.5_10pc_20240529210909/validation/validation_H003_bugfix_20241021162937/output/'}
           }
    obs_dic = {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/obs/targets_v3/merged/',
           'root_local':'obs/targets_v3/merged/'}

    merged_e3sm_dic={'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/postproc_e3sm/v3_merged_targets/ne30_F2010/',
                'root_local':'./rsync_e3sm_data/v3_merged_targets/ne30_F2010/'}


    # Load the data if its already saved and clobber=False
    outdir = 'processed'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    savename_rshp = os.path.join(outdir, validation_key + '_rshp_w_obs.nc')
    savename      = os.path.join(outdir, validation_key + '.nc')
    if not clobber:
        if os.path.isfile(savename_rshp):
            all_cases_rshp = xr.load_dataset(savename_rshp)
        if os.path.isfile(savename_rshp):
            all_cases = xr.load_dataset(savename)

    obs = {}
    for d in obs_dic.keys():
        if os.path.isdir(obs_dic[d]):
            obs_root=obs_dic[d]
            for s in ['DJF','MAM','JJA','SON']:
                obs[s]=xr.open_dataset(f'{obs_root}/obs_merged_{s}_24x48.nc')

    if not clobber:
        print('returning data loaded from netcdf. did not reprocess because clobber=False')
        return all_cases_rshp, all_cases, obs


    # If clobber = True or the data is not already there, process it.
    
    ###############################################################
    ###### Derived data used in multiple figures.################## 
    ###############################################################
    # Decide from where to load the data. If on pelrmutter, load it from the original source.
    # If you're not on perlmutter, it looks locally in a csv dir that is currently not part of the repo. You can put the csv data there yourself.
    dic = data_dic[validation_key]
    if os.path.isdir(dic['root_pmcpu']):
            dic['root']=dic['root_pmcpu']
    else:
        if os.path.isdir(dic['root_local']):
            dic['root']=dic['root_local']
        else:
            dic['root']=''
            print(f'could not find data for {d} validation locally or on pm-cpu')

    surrogate_and_model_data = glob.glob(os.path.join( dic['root'], f'pred_{validation_key}*.nc') )
    if len(surrogate_and_model_data) > 0:
        if len(surrogate_and_model_data) > 1:
            print('found more than one pred file. opening the first one')
        all_cases = xr.open_dataset(surrogate_and_model_data[0])

    mask_f = os.path.join( dic['root'], 'mask.pkl' )
    mask =  pd.read_pickle( mask_f )


    # Load the target files
    merged_e3sm = {}; merged_files=[]
    for d in merged_e3sm_dic.keys():
        if os.path.isdir(merged_e3sm_dic[d]):
            merged_root=merged_e3sm_dic[d]
            merged_files = glob.glob(os.path.join(merged_root,'**/*merged.nc'),recursive=True)


    # Construct list of vars in the order that the surrogate has them. I used the "responses" in the surrogate provenance dir.
    scalars = ['RESTOM','dnet_cld_dir']
    fields  = ['SWCF','LWCF','PRECT','TREFHT','PSL','Z500','U850','U200','RELHUM','T','U']
    all_var_dic = {'ANN':scalars,'DJF':fields,'MAM':fields,'JJA':fields,'SON':fields}

    # Full table here: https://acme-climate.atlassian.net/wiki/spaces/NGDSA/pages/4414341121/2024-06-24+Meeting+notes
    all_cases_rshp, all_cases  = reshape( all_cases,all_var_dic, mask)

    # Attach obs 
    all_cases_rshp = attach_obs(all_cases_rshp, obs )

    # Attach feedbacks
    all_cases_rshp = attach_feedbacks( all_cases_rshp, merged_files, merged_root)
    
    # save
    if clobber or not os.path.isfile(savename):
        print(f'saving {savename_rshp} locally.') 
        all_cases_rshp.to_netcdf(savename_rshp)
        print(f'saving {savename} locally.') 
        all_cases.to_netcdf(savename)

        
    return all_cases_rshp, all_cases, obs 


# data, lat, and area are all shape ncol
def zonal_means_native(data, lat, area, lat_south, lat_north, dlat):

    lat1 = lat_south
    lat2= lat_north
    
    nbin      = np.round( ( lat2 - lat1 + dlat )/dlat ).astype(int)
    bin_coord = xr.DataArray( np.linspace(lat1,lat2,nbin) )

    shape,dims,coord = (nbin,),'bins',[('bins', bin_coord.data)]


    bin_cnt = xr.DataArray( np.zeros(shape,dtype=lat.dtype), coords=coord, dims=dims )
    bin_val = xr.DataArray( np.zeros(shape,dtype=lat.dtype), coords=coord, dims=dims )
    bin_cnt[:] = np.nan
    bin_val[:] = np.nan
    
    condition = xr.DataArray( np.full(lat.shape,False,dtype=bool), coords=lat.coords )

    wgt, *__ = xr.broadcast(area,lat)
    wgt = wgt.transpose()
    #data_area_wgt = (data*wgt) / wgt.sum()

    #-------------------------------------------------------------------------------
    # Loop through bins
    for b in range(nbin):
        bin_bot = lat1 - dlat/2. + dlat*(b  )
        bin_top = lat1 - dlat/2. + dlat*(b+1)
        condition.values = ( lat >=bin_bot )  &  ( lat < bin_top )
        bin_cnt[b] = condition[:].sum()
        if bin_cnt[b]>1 :

            numerator = (data * wgt)[:].where(condition[:],drop=True).squeeze()
            denominator = wgt[:].where(condition[:],drop=True)

            # Where the numerator has nans, assign nans to the denominator (the weights).
            # Previously, the weights were being summed where the numerator had nans. 
            numerator_no_nans = numerator[~np.isnan(numerator)]
            denominator_no_nans = denominator[~np.isnan(numerator)]
            
            bin_val[b] = np.sum( numerator_no_nans) /  np.sum( denominator_no_nans) 

        if bin_cnt[b] == 1:
            bin_val[b] = data[:].where(condition[:],drop=True)
    #-------------------------------------------------------------------------------
    # use a dataset to hold all the output
    bin_ds = xr.Dataset()
    bin_ds['bin_val'] = bin_val
    bin_ds['bin_cnt'] = bin_cnt 
    #bin_ds.to_netcdf(path=bin_tmp_file,mode='w')
    #-------------------------------------------------------------------------------
    return bin_ds

