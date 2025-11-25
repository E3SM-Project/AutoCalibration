# Inspired by http://nco.sourceforge.net/nco.txt
# Also read on confluence that one should not use the FV CAP (eg. 25x48) grid because it has cells centered on poles. So Im making a gaussian grid instead. I think this is appropriate for ne4np4 regridding because it has 1152 cells whereas np4 has 866 columns. 

ncremap -G ttl="lat-lon-gaus-grid"#latlon=24,48#lat_typ=gss#lon_typ=grn_ctr -g 24x48_SCRIP.nc
