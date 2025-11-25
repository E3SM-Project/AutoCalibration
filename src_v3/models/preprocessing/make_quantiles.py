import os, pdb, xarray as xr, numpy as np, glob, copy, pandas as pd
home = os.getenv("HOME")
print()
#---------------------------------------------------------------------------------------------------
case, case_root, case_sub_in, case_sub_out = [],[],[],[]
def add_case( case_in, root_in, sub_in, sub_out=None ):
   global case_list, case_root, case_sub_in, case_sub_out
   case.append(case_in)
   case_root.append(root_in)
   case_sub_in.append(sub_in)
   case_sub_out.append(sub_out)
#---------------------------------------------------------------------------------------------------
# NERSC paths

# ds_regions = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')

# obs_root_all = '/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/obs'
# obs_root_lwp = '/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune'

# # New DY1 ne256 PPE - no nudging
# tmp_root      = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-s15-mar7/setupC1'
# tmp_sub       = 'SCREAM.2024-autocal-00.ne256pg2/run'; out_sub   = tmp_sub.replace('run','pdf')
# file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2016-08-*-00000.nc'
# sim_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/sims-s15-mar7/setupC1'
# file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.PDF.nc'
# campaign      = 'DY1'

# # New DY1 ne256 PPE - w/ nudging
# tmp_root      = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-s15-mar7/setupD1'
# tmp_sub       = 'SCREAM.2024-autocal-00.ne256pg2/run'; out_sub   = tmp_sub.replace('run','pdf')
# file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2016-08-*-00000.nc'
# sim_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/sims-s15-mar7/setupD1'
# file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.PDF.nc'
# campaign      = 'DY1'


# # DY2 ne1024 PPE - 2-day - no nudging
tmp_root      = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2'
tmp_sub       = 'SCREAM.2024-autocal-00.ne1024pg2/run'; out_sub   = tmp_sub.replace('run','pdf')
file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc'
# sim_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/ne1024pg2_DY2'
sim_root_out  = '/global/cfs/cdirs/m3312/whannah/ecp-autotune/ne1024pg2_DY2'
file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.PDF.nc'
campaign      = 'DY2'
# obs_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/obs/DY2'
obs_root_out  = '/global/cfs/cdirs/e3sm/whannah/ecp-autotune/obs/DY2'
case_to_skip  = ['m0024','m0107','m0230']


# # DY1 ne1024 PPE - 5-day - no nudging
# tmp_root      = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-s15-mar7/setupA1'
# tmp_sub       = 'SCREAM.2024-autocal-00.ne1024pg2/run'; out_sub   = tmp_sub.replace('run','pdf')
# file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2016-08-*-00000.nc'
# # sim_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/ne1024pg2_DY1'
# sim_root_out  = '/global/cfs/cdirs/m3312/whannah/ecp-autotune/ne1024pg2_DY1'
# file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.PDF.nc'
# campaign      = 'DY1'
# # obs_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/obs/DY1'
# obs_root_out  = '/global/cfs/cdirs/e3sm/whannah/ecp-autotune/obs/DY1'
# case_to_skip  = ['m0024','m0025','m0061','m0237','m0289','m0290','m0262','m0263','m0264','m0266','m0267','m0270','m0272','m0274','m0275','m0279','m0292','m0293','m0294','m0295','m0296','m0299','m0300']

#---------------------------------------------------------------------------------------------------
# OLCF paths


ds_regions = xr.open_dataset('/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/regions.nc')


# obs_root_all = '/lustre/orion/proj-shared/cli115/hannah6/SCREAM.2024-autocal-00.ne1024pg2/obs'
# obs_root_lwp = '/lustre/orion/proj-shared/cli115/hannah6/SCREAM.2024-autocal-00.ne1024pg2/obs'

# DY2 ne1024 PPE - 2-day - no nudging
tmp_root      = '/lustre/orion/proj-shared/cli115/noel/e3sm_scratch/s10-feb7/dd1024'
tmp_sub       = 'SCREAM.2024-autocal-00.ne1024pg2/run'; #out_sub = tmp_sub.replace('run','pdf')
file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc'
sim_root_out  = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/ne1024pg2_DY2'
file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.quantiles.nc'
campaign      = 'DY2'
obs_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/obs/DY2'
obs_root_out  = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/obs/DY2'
obs_root_all  = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/obs/DY2/'; obs_root_lwp = obs_root_all
case_to_skip  = ['m0107','m0230']


# # DY1 ne1024 PPE - 5-day - no nudging
# tmp_root      = '/lustre/orion/proj-shared/cli115/noel/e3sm_scratch/s15-mar7/setupA1'
# tmp_sub       = 'SCREAM.2024-autocal-00.ne1024pg2/run'; #out_sub = tmp_sub.replace('run','pdf')
# file_name_in  = 'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2016-08-07-00000.nc'
# sim_root_out  = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/ne1024pg2_DY1'
# file_name_out = 'output.scream.AutoCal.daily_avg_ne30pg2.quantiles.nc'
# campaign      = 'DY1'
# # obs_root_out  = '/pscratch/sd/w/whannah/scream_scratch/ecp-autotune/obs/DY1'
# obs_root_out  = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/obs/DY1'
# obs_root_all = '/lustre/orion/proj-shared/cli115/hannah6/ecp-autotune/obs/DY1/'; obs_root_lwp = obs_root_all
# case_to_skip  = [ 'm0024','m0025','m0061','m0237','m0289','m0290','m0262','m0263',
#                   'm0264','m0266','m0267','m0270','m0272','m0274','m0275','m0279',
#                   'm0292','m0293','m0294','m0295','m0296','m0299','m0300']

#---------------------------------------------------------------------------------------------------
run_obs_only = False

print_stats = True


# # set nskip to ignore first day if needed
# if campaign == 'DY1': nskip = 0#1
# if campaign == 'DY2': nskip = 0

if run_obs_only:
   add_case(f'obs', None, sub_in='' )
   # add_case( 'm0000', tmp_root, sub_in=tmp_sub )
   # add_case( 'm0001', tmp_root, sub_in=tmp_sub )
   # add_case( 'm0002', tmp_root, sub_in=tmp_sub )
   # add_case( 'm0003', tmp_root, sub_in=tmp_sub )
else:
   # add_case(f'obs', None, sub_in='' )

   ### optimal parameter runs
   # add_case( 'optmar22ga', tmp_root, sub_in=tmp_sub )
   # add_case( 'optmar22gb', tmp_root, sub_in=tmp_sub )
   # add_case( 'optmar22ha', tmp_root, sub_in=tmp_sub )
   # add_case( 'optmar22hb', tmp_root, sub_in=tmp_sub )
   # add_case( 'optmar22hc', tmp_root, sub_in=tmp_sub )
   # add_case( 'optmar22hd', tmp_root, sub_in=tmp_sub )
   # add_case( 'optfeb20', tmp_root, sub_in=tmp_sub )  
   # add_case( 'optmar5', tmp_root, sub_in=tmp_sub )  
   # add_case( 'optmar15', tmp_root, sub_in=tmp_sub )  
   
   add_case( 'optmar26a', tmp_root, sub_in=tmp_sub )  
   add_case( 'optmar26b', tmp_root, sub_in=tmp_sub )  
   add_case( 'optmar26c', tmp_root, sub_in=tmp_sub )  
   add_case( 'optmar26d', tmp_root, sub_in=tmp_sub )  
   add_case( 'optmar26e', tmp_root, sub_in=tmp_sub )   
   add_case( 'optmar26f', tmp_root, sub_in=tmp_sub )   

   # #case_path_list = sorted(glob.glob(f'{tmp_root}/m[0-9][0-9][0-9][0-9]'))
   # case_path_list = sorted(glob.glob(f'{tmp_root}/opt*'))
   # for c in range(len(case_path_list)):
   # # for c in range(8):
   #    case_tmp = case_path_list[c].replace(f'{tmp_root}/','')
   #    # skip bad cases
   #    if case_tmp in case_to_skip: continue
   #    # if case_tmp=='m0107': continue # negative ice_sed_knob
   #    # if case_tmp=='m0230': continue # negative ice_sed_knob
   #    ### special logic to continue after failure to finish all cases
   #    # if int(case_tmp[1:])<230 : continue 
   #    # if int(case_tmp[1:])<237 : continue 
   #    ### special logic to redo first cases
   #    # if int(case_tmp[1:])>25 : continue 
   #    add_case( case_tmp, tmp_root, sub_in=tmp_sub )

# print(); print(case); exit()

# build list of variables to consider
var,var_str = [],[]
var.append('precip_total_surf_mass_flux'); var_str.append('precip')
var.append('LiqWaterPath')               ; var_str.append('LWP')
var.append('LW_flux_up_at_model_top')    ; var_str.append('TOA LW')
var.append('SW_flux_up_at_model_top')    ; var_str.append('TOA SW')

#---------------------------------------------------------------------------------------------------
num_case = len(case)
num_var  = len(var)
#---------------------------------------------------------------------------------------------------
# load mask data
ds_regions = ds_regions.isel(time=0,drop=True)
mask_pole_all = ds_regions['poles']
mask_extr_ocn = ds_regions['extratropical_ocean']
mask_extr_lnd = ds_regions['extratropical_land']
mask_trop_lnd = ds_regions['tropical_land']
mask_trop_oup = ds_regions['ascending_tropical_ocean']
mask_trop_odn = ds_regions['descending_tropical_ocean']

region_name, region_data = [],[]
region_name.append('pole_all'); region_data.append(mask_pole_all)
region_name.append('extr_ocn'); region_data.append(mask_extr_ocn)
region_name.append('extr_lnd'); region_data.append(mask_extr_lnd)
region_name.append('trop_lnd'); region_data.append(mask_trop_lnd)
region_name.append('trop_oup'); region_data.append(mask_trop_oup)
region_name.append('trop_odn'); region_data.append(mask_trop_odn)

num_regions = len(region_name)
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
case_skip_list = []
for c in range(num_case):
   print(); print(' '*2+f'case: {case[c]}')
   #----------------------------------------------------------------------------
   # open input file
   #if case[c]!='obs':
   
   # if not os.path.isfile(f'{case_root[c]}/{case[c]}/{case_sub_in[c]}/{file_name_in}'):
   #     casecheck = case[c]
   #     print(f' no file for case {casecheck}')
   
   if (case[c]!='obs') and (os.path.isfile(f'{case_root[c]}/{case[c]}/{case_sub_in[c]}/{file_name_in}')):
      # print(f'{case_root[c]}/{case[c]}/{case_sub_in[c]}/{file_name_in}')
      # exit()

      ds = xr.open_mfdataset( f'{case_root[c]}/{case[c]}/{case_sub_in[c]}/{file_name_in}' )

      ### alternate way of loading the files that skips cases where the input file is not readable
      ### this is disabled since we don't expect this to happen and we want the script to  
      ### fail when this happens so we're forced to figure out what's wrong
      # casecheck = case_sub_in[c]
      # try:
      #     ds = xr.open_mfdataset( f'{case_root[c]}/{case[c]}/{case_sub_in[c]}/{file_name_in}' )
      #     print(f'opened data for case {casecheck}')
      # except:
      #     print(f'cant open file. skipping case {casecheck}')
      #     continue
      # casecheck = case_sub_in[c]
      # print( casecheck ) # I hope this is not saving the last case to open....

      # # skip first "nskip" daily averages
      # if nskip>0: ds = ds.isel(time=slice(nskip,len(ds['time'])))
      area = ds['area']
      #-------------------------------------------------------------------------
      if not os.path.exists(sim_root_out): os.makedirs(sim_root_out)
      # define output file
      # out_path_tmp = f'{sim_root_out}/{case[c]}/{case_sub_out[c]}'
      out_path_tmp = f'{sim_root_out}/{case[c]}'
      if not os.path.exists(out_path_tmp): os.mkdir(out_path_tmp)
      out_file_tmp = f'{out_path_tmp}/{file_name_out}'
      out_ds = xr.Dataset()
   #----------------------------------------------------------------------------
   # loop over variables
   for v in range(num_var):
      if print_stats: print()
      # print(' '*4+f'var: {var[v]}')
      #-------------------------------------------------------------------------
      def obs_fix_time(ds):
         date_str = ds.encoding['source'].split('/')[-1].split('.')[-3]
         yr = int(date_str[0:4])
         mn = int(date_str[4:6])
         dy = int(date_str[6:8])
         ds['time'] = pd.DatetimeIndex([f'{yr}-{mn:02d}-{dy:02d} 12:00:00'])
         ds = ds.drop('lat_vertices')
         ds = ds.drop('lon_vertices')
         return ds
      #-------------------------------------------------------------------------
      # special logic is needed to load the obs data
      if case[c]=='obs':
         if campaign == 'DY1':
            # if var[v]=='precip_total_surf_mass_flux': obs_file_path = f'{obs_root_all}/IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
            # if var[v]=='LiqWaterPath'               : obs_file_path = f'{obs_root_lwp}/mac-data/mac.clwp-tlwp-wvp.201608*.ne30pg2.nc'
            # if var[v]=='LW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.LW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
            # if var[v]=='SW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.SW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
            if var[v]=='precip_total_surf_mass_flux': obs_file_path = f'{obs_root_all}/IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
            if var[v]=='LiqWaterPath'               : obs_file_path = f'{obs_root_lwp}/mac.clwp-tlwp-wvp.20160807.ne30pg2.nc'
            if var[v]=='LW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.LW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
            if var[v]=='SW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.SW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807-20160810.nc'
         if campaign == 'DY2':
            if var[v]=='precip_total_surf_mass_flux': obs_file_path = f'{obs_root_all}/IMERG.precip_total_surf_mass_flux.AVERAGE.ne30pg2.20200126.nc'
            if var[v]=='LiqWaterPath'               : obs_file_path = f'{obs_root_lwp}/mac.clwp-tlwp-wvp.20200126.ne30pg2.nc'
            if var[v]=='LW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.LW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
            if var[v]=='SW_flux_up_at_model_top'    : obs_file_path = f'{obs_root_all}/CERES.SW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
         #----------------------------------------------------------------------
         file_chk = glob.glob(obs_file_path)
         # if not os.path.exists(obs_file_path):
         if file_chk==[]:
            print(); print('ERROR: obs data file path is incorrect?')
            print(); print(f'  obs_file_path: {obs_file_path}')
            print()
            exit()
         #----------------------------------------------------------------------
         if var[v]=='LiqWaterPath'               : 
            if campaign == 'DY1': ds = xr.open_mfdataset( obs_file_path, preprocess=obs_fix_time )
            if campaign == 'DY2': ds = xr.open_mfdataset( obs_file_path )
            # # skip first "nskip" daily averages
            # ds = ds.isel(time=slice(nskip,len(ds['time'])))
            ds = ds.isel(time=0)
         else:
            ds = xr.open_dataset( obs_file_path )
         area = ds['area']
         #----------------------------------------------------------------------
         if var[v]=='precip_total_surf_mass_flux': data = ds['precip_total_surf_mass_flux']*86400*1e3
         if var[v]=='LiqWaterPath'               : data = ds['tlwp']
         if var[v]=='LW_flux_up_at_model_top'    : data = ds['LW_flux_up_at_model_top']
         if var[v]=='SW_flux_up_at_model_top'    : data = ds['SW_flux_up_at_model_top']
         #----------------------------------------------------------------------
         if not os.path.exists(obs_root_out): os.makedirs(obs_root_out)
         # define output file
         out_file_tmp = f'{obs_root_out}/{file_name_out}'
         if 'out_ds' not in locals(): out_ds = xr.Dataset()
      #-------------------------------------------------------------------------
      # else => load the model simulation data
      else:
         if var[v]=='precip_total_surf_mass_flux': data = ds[var[v]]*86400*1e3 # m/s => mm/day
         if var[v]=='LiqWaterPath'               : data = ( ds['LiqWaterPath'] + ds['RainWaterPath'] )*1e3 # kg/m2 => g/m2
         if var[v]=='LW_flux_up_at_model_top'    : data = ds[var[v]]
         if var[v]=='SW_flux_up_at_model_top'    : data = ds[var[v]]
      #-------------------------------------------------------------------------
      data.load()
      #-------------------------------------------------------------------------
      # loop over regions
      for r in range(num_regions):
         # get regional subset of data and area
         data_reg = data.where( region_data[r]>0, drop=True )
         area_reg = area.where( region_data[r]>0, drop=True )
         #----------------------------------------------------------------------
         # print some basic stats for a sanity check
         if print_stats:
            msg  = ' '*8+f'{var[v]:30}  reg: {region_name[r]:10}'
            msg += f'   min: {np.nanmin(data_reg.values) :8.1f}'
            msg += f'   avg: {np.mean(data_reg.values)   :8.1f}'
            msg += f'   max: {np.nanmax(data_reg.values) :8.1f}'
            msg += f'   shp: {data_reg.shape}'
            if r==0: print()
            print(msg)
         #----------------------------------------------------------------------
         # calculate area weighted global mean for sanity check
         # wgt, *__ = xr.broadcast(area_reg, data_reg) 
         # gbl_mean = ( (data_reg*wgt).sum() / wgt.sum() ).values 
         #----------------------------------------------------------------------
         # if var[v] in ['precip_total_surf_mass_flux','LiqWaterPath']:
         #    data_reg = np.sqrt( data_reg )
         #----------------------------------------------------------------------
         # quant = xr.DataArray( np.arange(0.05,1,0.05) )
         # quant = xr.DataArray( np.arange(0.025,1,0.025) )
         quant = xr.DataArray( np.arange(0.0,1+0.025,0.025) )
         # qvals = xr.DataArray( np.quantile(np.ma.masked_invalid(data_reg.values),quant) )
         qvals = xr.DataArray( np.nanquantile(data_reg.values,quant) )
         #----------------------------------------------------------------------
         # msg  = ' '*8+f'{var[v]:30}  reg: {region_name[r]:10}'
         # msg += f'   min: {np.nanmin(qvals.values) :8.1f}'
         # msg += f'   avg: {np.mean(qvals.values)   :8.1f}'
         # msg += f'   max: {np.nanmax(qvals.values) :8.1f}'
         # if r==0: print()
         # print(msg)
         # exit()
         #----------------------------------------------------------------------
         # print(); print(data_reg)
         # print(); print(quant)
         # print(); print(qvals)
         # exit()
         #----------------------------------------------------------------------
         quant_coord_name = f'quant_{var[v]}'
         quant = quant.rename({'dim_0':quant_coord_name})
         qvals = qvals.rename({'dim_0':quant_coord_name})
         #----------------------------------------------------------------------
         # add to output dataset
         var_suffix = f'{region_name[r]}_{var[v]}'
         out_ds.coords[quant_coord_name] = (quant_coord_name,quant.values)
         out_ds[f'qvals_{var_suffix}'] = qvals

   #----------------------------------------------------------------------------
   # write final dataset to file
   if print_stats: print()
   print(' '*2+f'writing to file: {out_file_tmp}')
   out_ds.to_netcdf(path=out_file_tmp,mode='w')
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



















