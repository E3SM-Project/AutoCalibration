mkdir v3_merged_targets
rsync -av   --files-from=merged_list.txt wagmanbe@perlmutter.nersc.gov:/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ v3_merged_targets/
