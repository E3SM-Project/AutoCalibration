import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from cftime import num2date
from datetime import date, timedelta, datetime
from dateutil.relativedelta import *
import pandas as pd
import pdb
import xarray as xr
import glob
import shutil
import sys
import fnmatch
import random
import matplotlib.colors as colors
from cartopy import crs
import copy
from csv_to_xarray_fns import return_annual_cost, cost_only, ds_to_array
from   zonal_means_native import zonal_means_native

         
# Cases that are named differently  described here. Otherise, name in all_cases matches casename. 
# Name in cost file                         E3SM Case name                                     Bryce's  case name
# workdir.293                                                                                  v3alt.LR.lowECS001 
# dnet-2.1_RESTOM-0.5_rw2                   validate.dnet-2.1_RESTOM-0.5_rw2_20240531132555    v3alt.LR.lowECS002
# dnet-2.1_RESTOM-0.5_highR2                validate.dnet-2.1_RESTOM-0.5_highR2_20240531093356 v3alt.LR.lowECS003
# validate.dnet-1.5_RESTOM2.5               validate.dnet-1.5_RESTOM2.5_20240530123051         v3alt.LR.highECS03

# Full table here: https://acme-climate.atlassian.net/wiki/spaces/NGDSA/pages/4414341121/2024-06-24+Meeting+notes
# High 02 used validate.opt_params_dnet-1.5_reweight_mincdnc12.5e6_20240423132337 . The name is misleading bec. value of p3_mincdnc is not 12.5e6. 
# Choose input CSV and assign labels to optimal high and low cases. 
csv_dic={'H003':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-1.5_RESTOM2.5_10pc_20240529210909/validation/validation_H003_bugfix_20241021162937/output/',
          'root_local':'csv/',
          'root':'',
         'cost':'val_modsim_H003_bugfix.csv',
          'cost_wgts': 'cost_weights_H003_bugfix.csv',
          'pred_cost':'val_surrogate_H003_bugfix.csv'},
         'L002':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-2.1_RESTOM-0.5_10pc_20240530133621/validation/validation_L002_bugfix_20241021161335/output/',
          'root_local':'csv/',
          'root':'',
          'cost':'val_modsim_L002_bugfix.csv',
          'cost_wgts':'cost_weights_L002_bugfix.csv',
          'pred_cost':'val_surrogate_L002_bugfix.csv'}
         }
hdic = {'01':'H001','02':'H002','03':'H003'}
ldic = {'01':'L001','02':'L002','03':'L003'}


###############################################################
###### Derived data used in multiple figures.################## 
###############################################################
# Decide from where to load the data. If on pelrmutter, load it from the original source.
# If you're not on perlmutter, it looks locally in a csv dir that is currently not part of the repo. You can put the csv data there yourself. 
for d in csv_dic.keys():
    if os.path.isdir(csv_dic[d]['root_pmcpu']):
        csv_dic[d]['root']=csv_dic[d]['root_pmcpu']
    else:
        if os.path.isfile(os.path.join(csv_dic[d]['root_local'],csv_dic[d]['cost'])):
            csv_dic[d]['root']=csv_dic[d]['root_local']
        else:
            print('could not find data locally or on pm-cpu')

cost      = pd.read_csv(os.path.join( csv_dic['H003']['root'], csv_dic['H003']['cost'] ))
cost_wgts = pd.read_csv(os.path.join( csv_dic['H003']['root'], csv_dic['H003']['cost_wgts'] ))

pred_cost = pd.read_csv(os.path.join( csv_dic['H003']['root'], csv_dic['H003']['pred_cost'] ))
cost, cost_wgts, pred_cost = cost.to_xarray(), cost_wgts.to_xarray(), pred_cost.to_xarray()

h_costs = {'actual': {'data':cost, 'sets':{} },
             'pred': {'data': pred_cost, 'sets':{}}
             }
# Create categories for sims of interest. 
for k,v in h_costs.items():            
    data = v['data']
    v['sets']['ens']=data.where( data.id.str.contains('ens'), drop=True)
    v['sets']['hm']=data.where( data.id.str.contains('hm'), drop=True)
    v['sets']['valid']=data.where( data.id.str.contains('valid') | data.id.str.contains('dnet'), drop=True)
    #v['sets']['valid_hi'] =  data.where(data.id.str.match(hdic['03']), drop=True)
    v['sets']['valid_hi'] = data.where((data.id.str.match(hdic['01']) |
                             data.id.str.match(hdic['02']) |
                             data.id.str.match(hdic['03'])), drop=True)
    v['sets']['valid_lo'] = data.where((data.id.str.match(ldic['01'])  |
                             data.id.str.match(ldic['02']) |
                             data.id.str.contains(ldic['03'])), drop=True)
    v['sets']['p3_lo'] = data.where( data.p3_mincdnc < 5e6 , drop=True)
    v['sets']['ctrl']  = data.where( data.id.str.contains('ctrl'), drop=True)

pnames = ['clubb_c1', 'clubb_gamma_coef', 'zmconv_tau', 'zmconv_dmpdz', 'zmconv_micro_dcs',
          'zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','nucleate_ice_subgrid', 'p3_nc_autocon_expon',
          'p3_qc_accret_expon','cldfrc_dp1', 'p3_embryonic_rain_size','p3_mincdnc' ]

# Construct array of weighted and unweighted costs. 
ctrl =  h_costs['actual']['sets']['ctrl'] 
ens =  h_costs['actual']['sets']['ens'] 
hi  = h_costs['actual']['sets']['valid_hi']
lo  = h_costs['actual']['sets']['valid_lo']
ar_cost_names = np.array(list(ctrl.keys()))
ar_cost_names_plot = np.array([ w.replace('dnet_cld_dir','$ \lambda $') for w in ar_cost_names])

#ar_ctrl, ar_ens, ar_hi = ds_to_array(ctrl), ds_to_array(ens), ds_to_array(hi)
#ar_ctrl_wgtd, ar_ens_wgtd, ar_hi_wgtd = ds_to_array(ctrl, wgt=cost_wgts ), ds_to_array(ens, wgt=cost_wgts ), ds_to_array(hi, wgt=cost_wgts )
# i_argsort , i_argsort_wgtd = np.argsort( ar_ctrl), np.argsort( ar_ctrl_wgtd)
# i_argsort_wgtd_hi_vs_ctrl = np.argsort(ar_hi_wgtd.squeeze() - ar_ctrl_wgtd)
# sort_wgtd_hi_vs_ctrl = np.sort(ar_hi_wgtd.squeeze() - ar_ctrl_wgtd)



# # Consolidate the seasons into annual sum. 
# ctrl_annual_ds = return_annual_cost( ctrl )
# ctrl_annual_ds_wgt =  return_annual_cost( ctrl, wgt = cost_wgts)
# ens_annual_ds, ens_annual_ds_wgt = return_annual_cost( ens ), return_annual_cost( ens , wgt=cost_wgts) 
# hi_annual_ds, hi_annual_ds_wgt = return_annual_cost( hi ), return_annual_cost( hi, wgt=cost_wgts )
# ar_cost_names_an = np.array(list(ctrl_annual_ds.keys()))
# ar_cost_names_plot_an = np.array([ w.replace('dnet_cld_dir','$ \lambda $') for w in ar_cost_names_an])


# ar_ctrl_an, ar_ens_an, ar_hi_an = ds_to_array(ctrl_annual_ds), ds_to_array(ens_annual_ds),ds_to_array(hi_annual_ds)
# # For annual weighted arrays, just input the weighted annual dataset with no additional weighting. 
# ar_ctrl_wgtd_an, ar_ens_wgtd_an, ar_hi_wgtd_an = ds_to_array(ctrl_annual_ds_wgt), ds_to_array(ens_annual_ds_wgt),ds_to_array(hi_annual_ds_wgt)
# i_argsort_an, i_argsort_wgtd_an = np.argsort( ar_ctrl_an), np.argsort( ar_ctrl_wgtd_an)
# i_argsort_wgtd_hi_vs_ctrl_an = np.argsort(ar_hi_wgtd_an.squeeze() - ar_ctrl_wgtd_an)
# sort_wgtd_hi_vs_ctrl_an = np.sort(ar_hi_wgtd_an.squeeze() - ar_ctrl_wgtd_an)



###############################################################
###### End derived data used in multiple figures.##############
###############################################################


# Parameter radar plot. Examples
# https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
# https://www.pythoncharts.com/matplotlib/radar-charts/
# https://python-graph-gallery.com/web-circular-barplot-with-matplotlib/

# For now, a simple dot plot scaled by the min and max for each parameter.  
do_plot = True
if do_plot:
    fig, axes = plt.subplots(figsize=(10,4.5))
    ticks = np.arange(0, len( pnames))
    boxticks = ticks+1
    pdata = ens[pnames].to_array().transpose()
    pdata_max, pdata_min = ens[pnames].to_array().transpose().max(dim='index'),ens[pnames].to_array().transpose().min(dim='index')
    pdata_range = pdata_max - pdata_min
    pdata_scaled = (pdata - pdata_min) / pdata_range
    lodata_scaled = (lo[pnames].to_array().transpose() - pdata_min) / pdata_range
    hidata_scaled = (hi[pnames].to_array().transpose() - pdata_min) / pdata_range
    plt.scatter(np.broadcast_to( boxticks, pdata_scaled.shape) ,pdata_scaled,c='grey',alpha=0.5,marker='.')
    #plt.scatter(np.broadcast_to( boxticks +  0.15, lodata_scaled.shape) ,lodata_scaled,
    #            edgecolors='blue', facecolors='none', marker='<')
    #plt.scatter(np.broadcast_to( boxticks - 0.15, hidata_scaled.shape) ,hidata_scaled,
    #            edgecolors='red', facecolors='none', marker='>')
    for ind in v['sets']['valid_lo'].index: # Loop over set of low/hi runs.
        x_ind = 0        # Loop over parameters cause plotting text cant handle arrays.
        for p in pnames: # Loop over parameters cause plotting text cant handle arrays. 
            scaled_pdata = (lo[p].sel(index=ind)  - pdata_min.sel(variable=p)) / pdata_range.sel(variable=p)
            if not lo.sel(index=ind).id == 'L001':
                plt.text( x = boxticks[x_ind] - 0.2, y=scaled_pdata, s= str(lo.sel(index=ind).id.values).replace("0", "").replace("L", "") , color='blue' )
            else:
                plt.text( x = boxticks[x_ind] - 0.2, y=scaled_pdata, s= str(lo.sel(index=ind).id.values).replace("0", "").replace("L", "") , color='blue', bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 3} )
            x_ind = x_ind + 1
    for ind in v['sets']['valid_hi'].index: # Loop over set of low/hi runs.
        x_ind = 0        # Loop over parameters cause plotting text cant handle arrays.
        for p in pnames: # Loop over parameters cause plotting text cant handle arrays. 
            scaled_pdata = (hi[p].sel(index=ind)  - pdata_min.sel(variable=p)) / pdata_range.sel(variable=p)
            if not hi.sel(index=ind).id == 'H003':
                plt.text( x = boxticks[x_ind] + 0.1, y=scaled_pdata, s= str(hi.sel(index=ind).id.values).replace("0", "").replace("H", ""), color='red' )
            else:
                plt.text( x = boxticks[x_ind] + 0.1, y=scaled_pdata, s= str(hi.sel(index=ind).id.values).replace("0", "").replace("H", ""), color='red', bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 3} )
            x_ind = x_ind + 1
    plt.scatter(boxticks, (ctrl[pnames].to_array().transpose() - pdata_min) / pdata_range, c='k')
    plt.xticks(ticks=boxticks, labels = pnames,rotation=60, fontsize=7)
    plt.ylabel('Parameter value (norm. by bounds)')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('png/param_opt.png')
    pdb.set_trace()





# Old load data
# Before Gavin re-generated with new names H001, H002, H003
# hdic = {'01':'validate/validate.v3alpha02.2023102',
#         '02':'validate/validate.opt_params_dnet-1.5_reweight_mincdnc12.5e6_20240423132337',
#         '03':'dnet-1.5_RESTOM2.5'
#         }
# ldic = {'01':'ens/workdir.293/20230802.v3alpha02.F2010.pmcpu.intel.8N',
#         '02':'dnet-2.1_RESTOM-0.5_rw2',
#         '03':'dnet-2.1_RESTOM-0.5_highR2'}
