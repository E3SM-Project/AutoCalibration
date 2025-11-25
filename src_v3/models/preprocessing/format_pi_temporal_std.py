import xarray as xr
import pickle

# Read in data
field_std = xr.open_mfdataset('/global/cfs/cdirs/e3sm/emulate/sigma_temporal/sigma_temporal_pi_24x48_aave.nc')

# Adjust precip units
field_std.PRECT.values *= 86400000.0

# Specify seasons and variables
seasons = ['DJF', 'MAM', 'JJA', 'SON']
res_names = {
    'SWCF': 'SWCF',
    'LWCF': 'LWCF',
    'PRECT': 'PRECT',
    'TREFHT': 'TREFHT',
    'PSL': 'PSL',
    'Z500': 'Z500',
    'U850': 'U850',
    'U200': 'U200',
    'RELHUM': 'RELHUM',
    'T': 'T', 
    'U': 'U'
}

S = {}
for s in seasons:
    for obs_name,std_name in res_names.items():
        if 'plev' in field_std[std_name].coords:
            field_std[std_name] = field_std[std_name].rename({'plev': 'lev'})
        S[s + '_' + obs_name] = field_std[std_name].loc[s]
            
        
glb_std = xr.open_mfdataset('/global/cfs/cdirs/e3sm/emulate/sigma_temporal/glb_annual_sigma_RESTOM_180x360_aave.nc')

S['RESTOM'] = glb_std.RESTOM

S['dnet_cld_dir'] = xr.DataArray([1.0])

with open('/global/cfs/cdirs/e3sm/emulate/data/pi_temporal_std/pi_temporal_std.pkl', 'wb') as f:
    pickle.dump(S, f)
