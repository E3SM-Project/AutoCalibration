import numpy as np
import xarray as xr
import glob
import os

# Data Directory
datadir = '/global/homes/g/gqcollli/Autotuning-NGD/data/v3_sens'
savedir = '/global/homes/g/gqcollli/Autotuning-NGD/data/v3_sens_copy'


# RESTOM
RESTOM_prefix = "lat_lon_5yr_180x360_ANN_"
lat_lon_files = glob.glob(os.path.join(datadir, RESTOM_prefix + "*"))

for f in lat_lon_files:
    dat = xr.open_dataset(f)
    if 'ens' in f.split('/')[-1]: # ensemble
        ens_area = dat['area']
        dat = dat.assign(RESTOM = np.multiply(dat['FSNT'] - dat['FLNT'], ens_area).sum(axis=1).sum(axis=1) / ens_area.sum(axis=1).sum(axis=1))
    elif 'obs' in f.split('/')[-1]: # obs
        dat = dat.assign(RESTOM = 0.7)
    else: # single sim
        sim_area = dat['area']
        dat = dat.assign(RESTOM = np.multiply(dat['FSNT'] - dat['FLNT'], sim_area).sum() / sim_area.sum())
        

    dat.to_netcdf(os.path.join(savedir, f.split('/')[-1]))
    
    
# Feedback
feedback_prefix = "feedbacks_lat_lon_5yr_180x360_ANN_"
feedback_files = glob.glob(os.path.join(datadir, feedback_prefix + "*"))
    
for f in feedback_files:
    dat = xr.open_dataset(f)
    if 'ens' in f.split('/')[-1]: # ensemble
        dnet = np.multiply(dat['dnet_cld_dir'], ens_area).sum(axis=1).sum(axis=1) / ens_area.sum(axis=1).sum(axis=1)
        dat = dat.assign(dnet_cld_dir_low = dnet)
        dat = dat.assign(dnet_cld_dir_high = dnet)
    else: # single sim
        dnet = np.multiply(dat['dnet_cld_dir'], sim_area).sum() / sim_area.sum()
        dat = dat.assign(dnet_cld_dir_low = dnet)
        dat = dat.assign(dnet_cld_dir_high = dnet)

    dat.to_netcdf(os.path.join(savedir, f.split('/')[-1]))

obs_feedback = xr.Dataset(
    data_vars=dict(
        dnet_cld_dir_low=(["val"], [-2.1]),
        dnet_cld_dir_high=(["val"], [-1.68]),
    ),
    coords=dict(
        val=[0]
    )
)

obs_feedback.to_netcdf(os.path.join(savedir, feedback_prefix + 'obs.nc'))