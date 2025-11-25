import numpy as np
from sklearn.metrics import make_scorer
import xarray as xr
import yaml
import os, time, sys
import clif.preprocessing as cpp
import joblib

import netCDF4 as nc

dataset = xr.open_dataset(
    os.path.join('/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ctrl/20210813.F2010.ne30pg2_oECv3_control.chrysalis/targets/atm/180x360_aave/10yr/20210813.F2010.ne30pg2_oECv3_control.chrysalis_ANN_001101_002012_climo.nc')
)
area = dataset['area']

ds = nc.Dataset('../../surrogate_models/area_180x360.nc', 'w', format='NETCDF4')
lat = ds.createDimension('lat', 180)
lon = ds.createDimension('lon', 360)
values = ds.createVariable('values', 'f4', ('lat', 'lon'))
values[:] = area
ds.close()

dataset = xr.open_dataset(
    os.path.join('/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_500_ne30/ctrl/20210813.F2010.ne30pg2_oECv3_control.chrysalis/targets/atm/24x48_aavg/10yr/20210813.F2010.ne30pg2_oECv3_control.chrysalis_ANN_001101_002012_climo.nc')
)
area = dataset['area']

ds = nc.Dataset('../../surrogate_models/area_24x48.nc', 'w', format='NETCDF4')
lat = ds.createDimension('lat', 24)
lon = ds.createDimension('lon', 48)
values = ds.createVariable('values', 'f4', ('lat', 'lon'))
values[:] = area
ds.close()

