#!/bin/bash

# To count failed runs
# python call_st_archive.py
# This will even list the failed runs. 18 failed for F2010 and 9 for F2010+4k. However, this info is inaccurate. Runs that failed due to node failure were not detected by this script, nor were timeouts. 

# Clean up temporary dirs
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/ -type d -name 'tmp.*' -exec rm -rf {} \;

# Count number of F2010 cases where the 24x48 climatology has successfully computed.

find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/ -type f -wholename  *24x48*20230802.v3alpha02.F2010.pmcpu.intel.8N_08_000208_000608_climo.nc | wc -l
# 331 as of 20231031

u# 342 as of 20231106.

# Count the number of 5-yr simulations that produced the last history file
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??*/**/archive/atm/hist -name 20230802.v3alpha02.F2010.pmcpu.intel.8N.eam.h0.0006-12.nc | wc -l # 93

# Count the number of 5-yr simulations that produced climo files (zppy ran)
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??*/**/post/atm/24x48_aave/clim/5yr -name 20230802.v3alpha02.F2010.pmcpu.intel.8N_DJF_000201_000612_climo.nc | wc -l # 82
# I will-rerun zppy to try to get 93 instead of 82.
# 20240411 Now I have 93 5yr (above) and 92 1yr (below)
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??*/**/post/atm/24x48_aave/clim/1yr -name 20230802.v3alpha02.F2010.pmcpu.intel.8N.p4k_DJF_000201_000212_climo.nc | wc -l # 82


# Count the number of simulations in workdir.3XX with merged 24x48 files
# There are 82, and 85 for 180x360
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*DJF*merged.nc  | wc -l
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*MAM*merged.nc  | wc -l
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*JJA*merged.nc  | wc -l
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*SON*merged.nc  | wc -l
# Count the number of simulations in workdir.3XX with feedback files. 
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*DJF*feedbacks*.nc  | wc -l  # Only 78. Update 20240411: 87
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*MAM*feedbacks*.nc  | wc -l
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*JJA*feedbacks*.nc  | wc -l
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/workdir.3??* -type f -wholename  *24x48*5yr*20230802*SON*feedbacks*.nc  | wc -l

# Count history matching zppy jobs are stuck in "running", then delete them.  
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*climo*aave*status' -exec grep -Hi "RUNNING" {} \;  | wc -l
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*climo*aave*status' -exec grep -Hi "RUNNING" {} \;  -exec rm -i {} \;
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*climo*aave*status' -exec grep -Hi "ERROR" {} \;  | wc -l
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*climo*aave*status' -exec grep -Hi "ERROR" {} \;  -exec rm {} \;



 

# Find out which sims died from node failure
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -name 'e3sm.log*' -exec grep -H "DUE TO NODE FAILURE" {} \; # This affected simulations 201-250
# Count the sims that died from node failure.
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -name 'e3sm.log*' -exec grep -H "DUE TO NODE FAILURE" {} \; | wc -l # 49

# Find out which p4k sims died from the old SNOWP error.
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*p4k*e3sm.log*' -exec grep -H "Non blocking write for variable (SNOWDP, varid=157) failed" {} \; 

# Find out which p4k simulations timed out.
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*p4k*e3sm.log*' -exec grep -Hi "timeout" {} \;  # 13 in total. 

######### Postprocessing ############
# Find out which p4k zppy climo jobs timed out. Then delete their status files and resubmit them. This tends to be a machine issue and they will likely run the 2nd time. 
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*p4k*climo*aave*' -exec grep -Hi "DUE TO TIME LIMIT" {} \; | wc -l  # 24.
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*p4k*climo*aave*status' -exec grep -Hi "RUNNING" {} \;  | wc -l # Also 24. Let's delete the status file, then run call_zppy again. I'll increase the wallclock time.
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/  -wholename '*p4k*climo*aave*status' -exec grep -Hi "RUNNING" {} \; -exec rm -i {} \;


###### History matching ########

# Clean up temporary dirs
find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/hm/ -type d -name 'tmp.*' -exec rm -rf {} \;


# Count history matching zppy jobs are stuck in "running" 
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/hm/  -wholename '*climo*aave*status' -exec grep -Hi "RUNNING" {} \;  | wc -l

 # Delete history matching zppy jobs that are stuck in running so I can resubmit.
 find /pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/hm/  -wholename '*climo*aave*status' -exec grep -Hi "RUNNING" {} \;  -exec rm -i {} \;
 





