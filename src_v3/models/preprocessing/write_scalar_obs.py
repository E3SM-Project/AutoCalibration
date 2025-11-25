import xarray as xr
import os

# Data Directory
savedir = '/pscratch/sd/g/gqcollli/data/scalar_obs'


# RESTOM
dat = xr.Dataset(dict(
        RESTOM=(["global_mean"], [2.5]),
    ))
dat.to_netcdf(os.path.join(savedir, "obs_ANN_RESTOM2.5.nc"))
    
    
# # Feedback
# dnet_cld_dir = [-2.1, -1.5]

# for dnet in dnet_cld_dir:
#     dat = xr.Dataset(dict(
#             dnet_cld_dir=(["global_mean"], [dnet]),
#         ))
#     dat.to_netcdf(os.path.join(savedir, f"obs_ANN_dnet_cld_dir_{dnet}.nc"))
    