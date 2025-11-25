import xarray as xr
import numpy as np

# Define functions.
def return_annual_cost(costs_only_ds, wgt=False):
    annual_ds =  costs_only_ds.copy()
    vars_in = np.array(list(costs_only_ds.keys()))
    vars_with_seasons = [i.split('_')[1] for i in vars_in if i.startswith(f'DJF_')]
    # Step 1. Apply weights if provided.
    for f in vars_in:
        if wgt:
            if wgt[f]:
                annual_ds[f].values = costs_only_ds[f].values * wgt[f].values
    # Step 2. Consolidate seasons by summing.
    for f in vars_with_seasons:
        season_varnames = [i for i in vars_in if i.endswith(f'_{f}')]
        annual_ds[f] = (('index'), np.sum(annual_ds[season_varnames].to_array().values , axis=0))
        annual_ds = annual_ds.drop_vars(season_varnames)
    return annual_ds
            

def cost_only(ds):
    var_list= ['PSL','TREFHT','Z500','U200','U850','RELHUM','T','U','RESTOM']
    sn_list =  ['ANN','DJF','MAM','JJA','SON']
    retain=False
    for dsv in ds.data_vars:
        for v in var_list:
            if v in dsv:
                retain=True
        if not retain:
            ds = ds.drop_vars(dsv)
    return( ds )

# Make the dataset into an array. This is necessary because each cost component is saved as its own variable. 
def ds_to_array(ds, wgt=False):
    ar = []
    for v in np.array(list(ds.keys())):
        if wgt:
            ar.append(ds[v].data * wgt[v].data.squeeze())
        else:
            ar.append(ds[v].data)
    return np.array(ar).squeeze()
