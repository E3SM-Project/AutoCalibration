import numpy as np
import xarray as xr
import glob
import os

# Data Directory
datadir = '/global/homes/g/gqcollli/Autotuning-NGD/data/v3_sens'
savedir = '/global/homes/g/gqcollli/Autotuning-NGD/data/v3_sens_copy'


all_files = os.listdir(datadir)

for f in all_files:
    if '.nc' in f:
        dat = xr.open_dataset(os.path.join(datadir, f))
        if 'lhs' in dat.keys():
            dat = dat.rename({'lhs': 'params'})
            dat.to_netcdf(os.path.join(savedir, f.split('/')[-1]))
        if 'x' in dat.dims:
            dat = dat.rename({'x': 'input_params'})
            dat.to_netcdf(os.path.join(savedir, f.split('/')[-1]))
