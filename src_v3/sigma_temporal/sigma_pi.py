import os
import pdb
import shutil
import xarray as xr
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

# 20240403
# Create a single file for temporal sigma for targets for each season for v3  autotuning
# Uses v3 PI simulation from chrysalis. 

# When finished,
# SCP this to nersc:

#scp /lcrc/group/e3sm/ac.wagman/temporal/24x48_aave/*.nc wagmanbe@perlmutter.nersc.gov:/global/cfs/cdirs/e3sm/emulate/sigma_temporal
#scp /lcrc/group/e3sm/ac.wagman/temporal/180x360_aave/*.nc wagmanbe@perlmutter.nersc.gov:/global/cfs/cdirs/e3sm/emulate/sigma_temporal

# Function to set time to mid-month instead of beginning of next month. 
def set_ds_to_mid_month( ds, decode=False ):
    # Input a dataset 
    # Output a dataset with mid-month time. 
    # Attach a new coord to an existing dimension as in:
    #http://xarray.pydata.org/en/stable/generated/xarray.Dataset.assign_coords.html
    ds=ds.load()
    time_mid = ds.time_bnds.mean('nbnd')
    time_end_of_month = ds.time # save the old time
    ds_t=ds.assign_coords({'time':time_mid})  # Assign time_mid as a coordinate to the data array. Overwrites the old time. 
    # attributes are needed for decoding. Assign attributes. 
    ds_t.time.attrs= ds.time.attrs
    if decode:
        ds_t = xr.decode_cf( ds_t )
    return(ds_t) 


# Main code.
#res = '24x48_aave'  #'180x360_aave' # Options: '180x360_aave' or '24x48_aave'
res = '180x360_aave'
datadir_in = f'/lcrc/group/e3sm/ac.wagman/temporal/v3.LR.piControl/post/atm/{res}/ts/monthly/10yr/'
datadir_out = f'/lcrc/group/e3sm/ac.wagman/temporal/{res}'
datadir_plev_out = f'/lcrc/group/e3sm/ac.wagman/temporal/{res}/plev'



# remap 3-d vars to ERAI pressure levels. Copy the 2-d vars. 
def to_plevs(indir, outdir, fname, vrt_fl):
    cmd_to_plevs = 'ncremap  --vrt_fl={} --vrt_xtr=mss_val {} {}'.format(  vrt_fl , os.path.join( indir, fname ), os.path.join( outdir, fname) )   
    os.system( cmd_to_plevs)

create_plevs = False # Only need to do this once. 
if create_plevs:
    for f in glob.glob( os.path.join( datadir_in, '*4[56789]*101*.nc')):
        fname = os.path.basename(f)
        if (fname.startswith('U_') or fname.startswith('Z3') or fname.startswith('RELHUM') or fname.startswith('T_')):
            to_plevs( datadir_in, datadir_plev_out, fname, 'ERAI_L37.nc' )
        else:
            print(f'copying {f}')
            shutil.copyfile( f, os.path.join( datadir_plev_out, fname))

# Load all files into 1 dataset. If done incorrectly, the lat-lon files become nan. 
data = xr.open_mfdataset( glob.glob( os.path.join( datadir_plev_out, '*.nc')), coords='all')

# Derive additional variables to match up with our targets. 
data['PRECT'] = data['PRECC'] + data['PRECL']
data['LOGPRECT'] = np.log( data['PRECT'] ) # natural log. 
data['RESTOM'] = data['FSNT'] -  data['FLNT'] 
data['U850']  = data['U'].sel( {'plev': 8.5e4} )
data['U200']  = data['U'].sel( {'plev': 2e4} )
data['Z500']  = data['Z3'].sel( {'plev': 5e4} )
data['T500']  = data['T'].sel( {'plev': 5e4} )
data['RH500']  = data['RELHUM'].sel( {'plev': 5e4} )

# Take the zonal mean of the data on pressure levels.
for v in data.data_vars:
    if 'plev' in data[v].dims:
        data[v] = data[v].mean( dim='lon', skipna=True)

# Make into seasons, take sigma of the seasons. 
data_s = set_ds_to_mid_month( data ).load()
seasonal_std = data_s.groupby("time.season").std(dim='time', skipna=True)

# save
seasonal_std.to_netcdf(os.path.join(  datadir_out, f'sigma_temporal_pi_{res}.nc'))

# Compute an annual- global-mean RESTOM sigma and save as a separate file.
annual = data_s.groupby("time.year").mean(dim='time', skipna=True)
annual_w = annual.weighted( annual.area)
glb_mn = annual_w.mean(("lon", "lat"))
glb_annual_sigma_RESTOM = glb_mn['RESTOM'].std(skipna=True).to_netcdf( os.path.join(datadir_out,  f'glb_annual_sigma_RESTOM_{res}.nc')) # 0.44 for 24x48

# SCP all files to nersc:
# /global/cfs/cdirs/e3sm/emulate/sigma_temporal
