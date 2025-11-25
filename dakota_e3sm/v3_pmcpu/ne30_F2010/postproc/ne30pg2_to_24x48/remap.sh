# Get an example of the ne30pg2 grid for nco
ln -sf /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ctrl/20230802.v3alpha02.F2010.pmcpu.intel.8N/archive/atm/hist/20230802.v3alpha02.F2010.pmcpu.intel.8N.eam.h0.0001-11.nc data_on_ne30pg2.nc

# Get the source file for ne30pg2
ln -sf /global/homes/z/zender/data/grids/ne30pg2.g 

 
# Do a remap, creating the map file.
ncremap -a conserve  -s ne30pg2.nc -g scrip/24x48_SCRIP.nc  -m map_ne30pg2_to_24x48.nc  data_on_ne30pg2.nc data_on_24x48.nc

