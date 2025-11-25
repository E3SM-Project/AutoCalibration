#!/bin/bash

obs_dir=/project/projectdirs/e3sm/acme_diags/obs_for_e3sm_diags/climatology/  #NERSC


ncks -O  -C -v plev ${obs_dir}/ERA-Interim/ERA-Interim_01_197901_201601_climo.nc ERAI_L37.nc
#ncap2 -O -s "plev=plev*0.01" ERAI_L37.nc ERAI_L37.nc
#ncatted -O -a units,plev,m,c,"hPa" ERAI_L37.nc

