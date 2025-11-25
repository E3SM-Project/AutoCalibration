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
from fns.autotuning_fns_module import load_reshape_save, fix_lon
         
# Cases that are named differently  described here. Otherise, name in all_cases matches casename. 
# Name in cost file                         E3SM Case name                                     Bryce's  case name
# workdir.293                                                                                  v3alt.LR.lowECS001 
# dnet-2.1_RESTOM-0.5_rw2                   validate.dnet-2.1_RESTOM-0.5_rw2_20240531132555    v3alt.LR.lowECS002
# dnet-2.1_RESTOM-0.5_highR2                validate.dnet-2.1_RESTOM-0.5_highR2_20240531093356 v3alt.LR.lowECS003
# validate.dnet-1.5_RESTOM2.5               validate.dnet-1.5_RESTOM2.5_20240530123051         v3alt.LR.highECS03

# Full table here: https://acme-climate.atlassian.net/wiki/spaces/NGDSA/pages/4414341121/2024-06-24+Meeting+notes
# High 02 used validate.opt_params_dnet-1.5_reweight_mincdnc12.5e6_20240423132337 . The name is misleading bec. value of p3_mincdnc is not 12.5e6. 
# Choose input CSV and assign labels to optimal high and low cases. 

csv_dic={'H001':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_archive/pce/surrogate_max_order_4/',
          'root_local':''},
         'H002':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-1.5_20240416194142/validation/validation_H002_bugfix_20241021162536/output/',
          'root_local':'csv/validation_H003_bugfix_20241021162937/output/'},
         'H003':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-1.5_RESTOM2.5_10pc_20240529210909/validation/validation_H003_bugfix_20241021162937/output/',
          'root_local':'csv/validation_H003_bugfix_20241021162937/output/'},
         'L002':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-2.1_RESTOM-0.5_10pc_20240530133621/validation/validation_L002_bugfix_20241021161335/output/',
          'root_local':'csv/validation_L002_bugfix_20241021161335/output/'},
         'L003':
         {'root_pmcpu':'/global/cfs/cdirs/e3sm/emulate/surrogate_models/v3/surrogate_provenance_dnet-2.1_RESTOM-0.5_highR2_20240531093356/validation/validation_L003_bugfix_20241021162100/output/',
          'root_local':'csv/validation_L003_bugfix_20241021162100/output/'}

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
        if os.path.isdir(csv_dic[d]['root_local']):
            csv_dic[d]['root']=csv_dic[d]['root_local']
        else:
            csv_dic[d]['root']=''
            print(f'could not find data for {d} validation locally or on pm-cpu')


for opt in csv_dic.keys():
    csvs = glob.glob(os.path.join( csv_dic[opt]['root'],'*.csv') )
    csv_dic[opt]['cost'] =      [pd.read_csv(f) for f in csvs if 'val_modsim' in f]
    csv_dic[opt]['pred_cost'] = [pd.read_csv(f) for f in csvs if 'val_surrogate' in f]
    csv_dic[opt]['cost_wgts'] = [pd.read_csv(f) for f in csvs if 'cost_weights' in f]
            
opt_key='H003' # Costs and predictions vary depending on the optimization that produced each optimum. Must choose 1 optimization to rule them all.  

cost, cost_wgts, pred_cost = csv_dic[opt_key]['cost'][0].to_xarray(), csv_dic[opt_key]['cost_wgts'][0].to_xarray(), csv_dic[opt_key]['pred_cost'][0].to_xarray()

h_costs = {'actual': {'data':cost, 'sets':{} },
             'pred': {'data':pred_cost, 'sets':{}}
             }

# Create categories for sims of interest. 
for k,v in h_costs.items():            
    data = v['data']
    v['sets']['ens']=data.where( data.id.str.contains('ens'), drop=True)
    v['sets']['hm']=data.where( data.id.str.contains('hm'), drop=True)
    v['sets']['valid']=data.where( data.id.str.contains('valid') | data.id.str.contains('dnet'), drop=True)
    v['sets']['valid_hi'] =  data.where(data.id.str.match(hdic['03']), drop=True)
    v['sets']['valid_h01'] =  data.where(data.id.str.match(hdic['01']), drop=True)
    v['sets']['valid_h02'] =  data.where(data.id.str.match(hdic['02']), drop=True)
    v['sets']['valid_h03'] =  data.where(data.id.str.match(hdic['03']), drop=True)
    v['sets']['valid_l01'] =  data.where(data.id.str.match(ldic['01']), drop=True)
    v['sets']['valid_l02'] =  data.where(data.id.str.match(ldic['02']), drop=True)
    v['sets']['valid_l03'] =  data.where(data.id.str.match(ldic['03']), drop=True)
    v['sets']['all_valid_hi'] = data.where((data.id.str.match(hdic['01']) |
                             data.id.str.match(hdic['02']) |
                             data.id.str.match(hdic['03'])), drop=True)
    v['sets']['all_valid_lo'] = data.where((data.id.str.match(ldic['01'])  |
                             data.id.str.match(ldic['02']) |
                             data.id.str.contains(ldic['03'])), drop=True)
    v['sets']['p3_lo'] = data.where( data.p3_mincdnc < 5e6 , drop=True)
    v['sets']['ctrl']  = data.where( data.id.str.contains('ctrl'), drop=True)

# Construct array of weighted and unweighted costs. 
ctrl = cost_only( h_costs['actual']['sets']['ctrl'] ).drop_vars('total_cost')
ens = cost_only( h_costs['actual']['sets']['ens'] ).drop_vars('total_cost')
hi  = cost_only( h_costs['actual']['sets']['valid_hi'] ).drop_vars('total_cost')
all_hi = cost_only( h_costs['actual']['sets']['all_valid_hi'] ).drop_vars('total_cost')
all_lo = cost_only( h_costs['actual']['sets']['all_valid_lo'] ).drop_vars('total_cost')

ar_cost_names = np.array(list(ctrl.keys()))
ar_cost_names_plot = np.array([ w.replace('dnet_cld_dir','$ \lambda $') for w in ar_cost_names])

ar_ctrl, ar_ens, ar_hi, ar_all_hi, ar_all_lo = ds_to_array(ctrl), ds_to_array(ens), ds_to_array(hi), ds_to_array(all_hi),ds_to_array(all_lo)
ar_ctrl_wgtd, ar_ens_wgtd, ar_hi_wgtd = ds_to_array(ctrl, wgt=cost_wgts ), ds_to_array(ens, wgt=cost_wgts ), ds_to_array(hi, wgt=cost_wgts )
i_argsort , i_argsort_wgtd = np.argsort( ar_ctrl), np.argsort( ar_ctrl_wgtd)
i_argsort_wgtd_hi_vs_ctrl = np.argsort(ar_hi_wgtd.squeeze() - ar_ctrl_wgtd)
sort_wgtd_hi_vs_ctrl = np.sort(ar_hi_wgtd.squeeze() - ar_ctrl_wgtd)


# Consolidate the seasons into annual sum. 
ctrl_annual_ds = return_annual_cost( ctrl )
ctrl_annual_ds_wgt =  return_annual_cost( ctrl, wgt = cost_wgts)
ens_annual_ds, ens_annual_ds_wgt = return_annual_cost( ens ), return_annual_cost( ens , wgt=cost_wgts) 
hi_annual_ds, hi_annual_ds_wgt = return_annual_cost( hi ), return_annual_cost( hi, wgt=cost_wgts )
ar_cost_names_an = np.array(list(ctrl_annual_ds.keys()))
ar_cost_names_plot_an = np.array([ w.replace('dnet_cld_dir','$ \lambda $') for w in ar_cost_names_an])

ar_cost_names_an_heatmap = [a for a in ar_cost_names_an if not 'RESTOM' in a and not  'dnet_cld_dir' in a]

ar_ctrl_an, ar_ens_an, ar_hi_an = ds_to_array(ctrl_annual_ds), ds_to_array(ens_annual_ds),ds_to_array(hi_annual_ds)

# For annual weighted arrays, just input the weighted annual dataset with no additional weighting. 
ar_ctrl_wgtd_an, ar_ens_wgtd_an, ar_hi_wgtd_an = ds_to_array(ctrl_annual_ds_wgt), ds_to_array(ens_annual_ds_wgt),ds_to_array(hi_annual_ds_wgt)
i_argsort_an, i_argsort_wgtd_an = np.argsort( ar_ctrl_an), np.argsort( ar_ctrl_wgtd_an)
i_argsort_wgtd_hi_vs_ctrl_an = np.argsort(ar_hi_wgtd_an.squeeze() - ar_ctrl_wgtd_an)
sort_wgtd_hi_vs_ctrl_an = np.sort(ar_hi_wgtd_an.squeeze() - ar_ctrl_wgtd_an)

sn_list = ['DJF','MAM','JJA','SON']

# Load the simulation data as well.
#### LOAD DATA ##########
# Returns
# all_cases: dataset with obs, surrogate, and model on the same grid
# all_cases_orig: dataset with model, surrogate on the original surrogate vector but with nans to ensure equal lengths
# obs: A dictionary of obs for each season, in case working with the obs in all_cases breaks.

# Behavior:
# Will save netcdf files if do not exist, or if clobber=True.
# Will load from netcdf if exist and clobber=False

all_cases, all_cases_orig, obs = load_reshape_save('H003', clobber=False) 
#### END LOAD DATA ##########

###############################################################
###### End derived data used in multiple figures.##############
###############################################################

# Unweighted cost by component. High-ECS 03 target for RESTOM and dnet. 
do_plot = False
if do_plot:
    ticks = np.arange(0, len( ar_ctrl))
    plt.plot( ar_ens[i_argsort],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi[i_argsort],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl[i_argsort],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names[i_argsort], rotation=90, fontsize=6)
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_no_wgt.png')

# Same but WEIGHTED cost. 
do_plot = False
if do_plot:
    plt.figure()
    ticks = np.arange(0, len( ar_ctrl))
    plt.plot( ar_ens_wgtd[i_argsort_wgtd],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi_wgtd[i_argsort_wgtd],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl_wgtd[i_argsort_wgtd],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names[i_argsort_wgtd], rotation=90, fontsize=6)
    plt.ylabel('Weighted Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_wgt.png')


# Same but WEIGHTED annual cost. 
do_plot = False
if do_plot:
    plt.figure()
    ticks = np.arange(0, len( ar_ctrl_wgtd_an))
    plt.plot( ar_ens_wgtd_an[i_argsort_wgtd_an],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi_wgtd_an[i_argsort_wgtd_an],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl_wgtd_an[i_argsort_wgtd_an],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names_an[i_argsort_wgtd_an], rotation=90, fontsize=6)
    plt.ylabel('Weighted Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_wgt_an.png')



# WEIGHTED annual cost sorted by improvement over control.  
do_plot = False
if do_plot:
    plt.figure()
    ticks = np.arange(0, len( ar_ctrl_wgtd_an))
    plt.plot( ar_ens_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names_an[i_argsort_wgtd_hi_vs_ctrl_an], rotation=90, fontsize=6)
    plt.ylabel('Weighted Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_wgt_srt_by_improv_vs_ctrl_an.png')



    

do_plot = False
if do_plot:
    fig, axes = plt.subplots(figsize=(10,4.5))
    ticks = np.arange(0, len( ar_ctrl_an))
    boxticks = ticks+1
    Hi03total, ctrltotal = round(ar_hi_wgtd_an.sum(),2), round(ar_ctrl_wgtd_an.sum(),2)
    #plt.plot( boxticks, np.ones( boxticks.shape)) 
    plt.boxplot(ar_ens_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an].transpose(),showfliers=False)
    plt.xticks( ticks = boxticks, labels = ar_cost_names_plot_an[i_argsort_wgtd_hi_vs_ctrl_an], rotation=90, fontsize=7)
    plt.plot( boxticks, ar_hi_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an], 'ro',label=f'Hi03 {Hi03total}')
    plt.plot( boxticks, ar_ctrl_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an], 'ko',label=f'Ctrl {ctrltotal}')
    breakeven_i = np.sum(sort_wgtd_hi_vs_ctrl_an < 0) - 0.5
    plt.plot([breakeven_i, breakeven_i], [-1,20],'k--' ) 
    plt.ylabel('Weighted Cost')
    plt.gca().set_ylim([0,20])
    plt.legend()
    plt.tight_layout()
    plt.savefig('png/cost_component_boxplot_hi_wgt_srt_by_improv_vs_ctrl_an.png')


# Weighted annual, all 6 optimal cases, field values only.
do_plot = False
if do_plot:
    fig, axes = plt.subplots(figsize=(10,4.5))
    ticks = np.arange(0, len( ar_ctrl_an))
    boxticks = ticks+1
    Hi03total, ctrltotal = round(ar_hi_wgtd_an.sum(),2), round(ar_ctrl_wgtd_an.sum(),2)
    #plt.plot( boxticks, np.ones( boxticks.shape)) 
    plt.boxplot(ar_ens_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an].transpose(),showfliers=False)
    plt.xticks( ticks = boxticks, labels = ar_cost_names_plot_an[i_argsort_wgtd_hi_vs_ctrl_an], rotation=90, fontsize=7)
    plt.plot( boxticks, ar_hi_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an], 'ro',label=f'Hi03 {Hi03total}')
    plt.plot( boxticks, ar_ctrl_wgtd_an[i_argsort_wgtd_hi_vs_ctrl_an], 'ko',label=f'Ctrl {ctrltotal}')
    breakeven_i = np.sum(sort_wgtd_hi_vs_ctrl_an < 0) - 0.5
    plt.plot([breakeven_i, breakeven_i], [-1,20],'k--' ) 
    plt.ylabel('Weighted Cost')
    plt.gca().set_ylim([0,20])
    plt.legend()
    plt.tight_layout()
    pdb.set_trace()
    plt.savefig('png/cost_component_boxplot_hi_wgt_an_all_opt.png')

    
    
# WEIGHTED seasonal cost sorted by total improvement over control.  
do_plot = False
if do_plot:
    plt.figure()
    ticks = np.arange(0, len( ar_ctrl))
    plt.plot( ar_ens_wgtd[i_argsort_wgtd_hi_vs_ctrl],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi_wgtd[i_argsort_wgtd_hi_vs_ctrl],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl_wgtd[i_argsort_wgtd_hi_vs_ctrl],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names[i_argsort_wgtd_hi_vs_ctrl], rotation=90, fontsize=6)
    plt.ylabel('Weighted Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_wgt_srt_by_improv_vs_ctrl.png')

# Same but boxplot.  
do_plot = False
if do_plot:
    fig, axes = plt.subplots(figsize=(10,4.5))
    ticks = np.arange(0, len( ar_ctrl))
    boxticks = ticks+1
    #plt.plot( boxticks, np.ones( boxticks.shape)) 
    plt.boxplot(ar_ens_wgtd[i_argsort_wgtd_hi_vs_ctrl].transpose(),showfliers=False)
    plt.xticks( ticks = boxticks, labels = ar_cost_names_plot[i_argsort_wgtd_hi_vs_ctrl], rotation=90, fontsize=7)
    plt.plot( boxticks, ar_hi_wgtd[i_argsort_wgtd_hi_vs_ctrl], 'ro',label='Hi03')
    plt.plot( boxticks, ar_ctrl_wgtd[i_argsort_wgtd_hi_vs_ctrl], 'ko',label='Ctrl')
    breakeven_i = np.sum(sort_wgtd_hi_vs_ctrl < 0) - 0.5
    plt.plot([breakeven_i, breakeven_i], [-1,10],'k--' ) 
    plt.ylabel('Weighted Cost')
    plt.gca().set_ylim([0,7])
    plt.legend()
    plt.tight_layout()
    plt.savefig('png/cost_component_boxplot_hi_srt_by_improv_vs_ctrl.png')
    pdb.set_trace()




#Figure. Normalized cost boxplots, High ECS optimization. 
# For this plot, it makes sense to show all high-ECS, since weighting cost is not applied.
do_plot=False
if do_plot:
    plt.figure()
    boxticks = ticks+1
    ar_ens_norm = ar_ens / ar_ctrl[:,None]
    ar_hi_norm = ar_hi / ar_ctrl[:,None]
    plt.plot( boxticks, np.ones( boxticks.shape)) 
    plt.boxplot(ar_ens_norm[i_argsort].transpose(),showfliers=False)
    plt.xticks( ticks = boxticks, labels = ar_cost_names[i_argsort], rotation=90, fontsize=6)
    plt.ylabel('Cost')
    plt.gca().set_ylim([0,8])
    plt.plot( boxticks, ar_hi_norm[i_argsort], 'r.')
    plt.tight_layout()
    plt.savefig('png/cost_component_boxplot_norm_hi.png')
    plt.gca().set_ylim([0,2])
    plt.savefig('png/cost_component_boxplot_norm_hi_ylimzoom.png')
    plt.tight_layout()


#Figure. Normalized cost boxplots, High ECS optimization, ANNUAL sum of seasons. 
# For this plot, it makes sense to show all high-ECS, since weighting cost is not applied.
do_plot=False
if do_plot:
    plt.figure()
    ticks = np.arange(0, len( ar_ctrl))
    plt.plot( ar_ens[i_argsort],label='Ens', color='lightgrey',alpha=0.5)
    plt.plot( ar_hi[i_argsort],label='Hi ECS', color='red',alpha=0.5)
    plt.plot( ar_ctrl[i_argsort],label='Ctrl',color='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xticks( ticks = ticks, labels = ar_cost_names[i_argsort], rotation=90, fontsize=6)
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig('png/cost_component_hi_no_wgt.png')
    i_argsort = np.argsort( ar_ctrl_an)
    pdb.set_trace()
    
    

#### BEGIN HEATMAPS #################

    
# Function to remove dnet_cld_dir and RESTOM from data and array of names.
def remove_annual(ar_names, ar_data ):
    ar_data_no_restom = ar_data[ar_names != 'RESTOM']
    ar_names_no_restom = ar_names[ar_names != 'RESTOM']
    ar_data_no_dnet  = ar_data_no_restom[ar_names_no_restom != 'dnet_cld_dir']
    ar_names_no_dnet  = ar_names_no_restom[ar_names_no_restom != 'dnet_cld_dir']
    seasonal_only_data, seasonal_only_names = ar_data_no_dnet.squeeze(), ar_names_no_dnet.squeeze()
    return( seasonal_only_names, seasonal_only_data)
    
# Heat maps.
# Because these use unweighted cost and do not display RESTOM or dnet, it is fair to plot for all L1-3, H1-3.
# Could also plot surrogate side-by-side with model. 
do_plot = False
if do_plot:
    plt.ion()
    # 1 remove dnet and RESTOM from the data because they are annual.
    ar_names_heatmap,ar_hi_heatmap = remove_annual(  ar_cost_names, ar_hi ) 
    ar_names_heatmap,ar_ctrl_heatmap  = remove_annual( ar_cost_names, ar_ctrl )
    ar_ctrl_heatmap,ar_hi_heatmap = np.reshape(ar_ctrl_heatmap, (4,-1), order='F'), np.reshape(ar_hi_heatmap, (4,-1), order='F')
    pct_improv_heatmap  = 100 * ( ar_hi_heatmap  - ar_ctrl_heatmap ) / ar_ctrl_heatmap
    cmap = plt.get_cmap('coolwarm',9)
    fig, ax = plt.subplots()
    im = plt.imshow(pct_improv_heatmap.transpose(), vmin=-45, vmax=45, cmap=cmap, alpha=0.7 )
    # Loop over data dimensions and create text annotations.
    for i in range(len(ar_cost_names_an_heatmap)):
        for j in range(len(sn_list)):
            text = ax.text(j, i,  pct_improv_heatmap[j, i].astype(int),
                           ha="center", va="center", color="k")
    ax.set_yticks(np.arange(len( ar_cost_names_an_heatmap)))
    ax.set_xticks(np.arange(len( sn_list)))
    ax.set_yticklabels(ar_cost_names_an_heatmap )
    ax.set_xticklabels(sn_list)
    ax.set_title(f'H03, {np.round(np.average(pct_improv_heatmap),2)}$\%$')
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig('png/H03_heatmap')


# Multiple heatmaps.
ar_lo_and_hi = np.concatenate( (ar_all_lo, ar_all_hi),axis=-1)
lo_and_hi_labels = ['L1','L2','L3','H1','H2','H3']
lo_and_hi_labels_lambda = ['L1 $\lambda = -2.11 $','L2 $\lambda = -1.97 $','L3 $\lambda = -2.05 $',
                           'H1 $\lambda = -1.41 $','H2 $\lambda = -1.43 $','H3 $\lambda = -1.36 $']

do_plot = True
if do_plot:
    plt.ion()
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(8,9))
    col = 0
    for a in ax.ravel():
        optimal_data = ar_lo_and_hi[:,col]
        # 1 remove dnet and RESTOM from the data because they are annual.
        ar_names_heatmap,ar_data_heatmap = remove_annual(  ar_cost_names, optimal_data ) 
        ar_names_heatmap,ar_ctrl_heatmap  = remove_annual( ar_cost_names, ar_ctrl )
        ar_ctrl_heatmap,ar_data_heatmap = np.reshape(ar_ctrl_heatmap, (4,-1),order='F'), np.reshape(ar_data_heatmap, (4,-1), order='F')
        pct_improv_heatmap  = 100 * ( ar_data_heatmap  - ar_ctrl_heatmap ) / ar_ctrl_heatmap
        cmap = plt.get_cmap('coolwarm',9)
        im = a.imshow(pct_improv_heatmap.transpose(), vmin=-45, vmax=45, cmap=cmap, alpha=0.7 )
        # Loop over data dimensions and create text annotations.
        for i in range(len(ar_cost_names_an_heatmap)):
            for j in range(len(sn_list)):
                text = a.text(j, i,  pct_improv_heatmap[j, i].astype(int),
                               ha="center", va="center", color="k")
        a.set_yticks(np.arange(len( ar_cost_names_an_heatmap)))
        a.set_xticks(np.arange(len( sn_list)))
        a.set_yticklabels(ar_cost_names_an_heatmap )
        a.set_xticklabels(sn_list)
        # Include lambda values 2025-09-16. Remove percent_improv data from title.
        if col < 3: 
            a.set_title(lo_and_hi_labels_lambda[col], color='blue')
        else:
            a.set_title(lo_and_hi_labels_lambda[col], color='red')
        #a.set_title(f'{lo_and_hi_labels[col]}, {np.round(np.average(pct_improv_heatmap),2)}$\%$')
        col+=1
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig('png/every_heatmap')







# Fig. Low ECS validation. Cost are indicative of a Low-ECS target RESTOM and dnet.
pdb.set_trace()
plt.figure()
cost = pd.read_csv(os.path.join( csv_dic['L002']['root'], csv_dic['L002']['cost'] ))
pred_cost = pd.read_csv(os.path.join( csv_dic['L002']['root'], csv_dic['L002']['pred_cost'] ))
cost, pred_cost = cost.to_xarray(), pred_cost.to_xarray()

l_costs = {'actual': {'data':cost, 'sets':{} },
             'pred': {'data': pred_cost, 'sets':{}}
             }
# Create categories for sims of interest. 
for k,v in l_costs.items():            
    data = v['data']
    v['sets']['ens']=data.where( data.id.str.contains('ens'), drop=True)
    v['sets']['hm']=data.where( data.id.str.contains('hm'), drop=True)
    v['sets']['valid']=data.where( data.id.str.contains('valid') | data.id.str.contains('dnet'), drop=True)
    v['sets']['valid_hi'] = data.where((data.id.str.match(hdic['01']) |
                             data.id.str.match(hdic['02']) |
                             data.id.str.match(hdic['03'])), drop=True)
    v['sets']['valid_lo'] = data.where((data.id.str.match(ldic['01']) |
                             data.id.str.match(hdic['02']) |
                             data.id.str.contains(hdic['03'])), drop=True)
    v['sets']['p3_lo'] = data.where( data.p3_mincdnc < 5e6 , drop=True)
    v['sets']['ctrl']  = data.where( data.id.str.contains('ctrl'), drop=True)

# Append different variables into an array, then sort by ctrl cost and plot.
ctrl = cost_only( l_costs['actual']['sets']['ctrl'] ).drop_vars('total_cost')
ens = cost_only( l_costs['actual']['sets']['ens'] ).drop_vars('total_cost')
lo  = cost_only( l_costs['actual']['sets']['valid_lo'] ).drop_vars('total_cost')
ar_cost_names = np.array(list(ctrl.keys()))
ar_ctrl, ar_ens, ar_hi, ar_lo = [],[],[],[]
for v in ar_cost_names:
    ar_ctrl.append( ctrl[v].data[0])
    ar_ens.append( ens[v].data )
    ar_lo.append( lo[v].data )
ar_ctrl, ar_ens, ar_lo  = np.array(ar_ctrl), np.array(ar_ens), np.array(ar_lo)
i_argsort = np.argsort( ar_ctrl)
ticks = np.arange(0, len( ar_ctrl))
plt.plot( ar_ens[i_argsort],label='Ens', color='lightgrey',alpha=0.5)
plt.plot( ar_lo[i_argsort],label='Lo ECS', color='blue',alpha=0.5)
plt.plot( ar_ctrl[i_argsort],label='Ctrl',color='black')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xticks( ticks = ticks, labels = ar_cost_names[i_argsort], rotation=90, fontsize=6)
plt.ylabel('Cost')
plt.tight_layout()
plt.savefig('png/cost_component_lo.png')









#Normalized cost boxplots, Low ECS optimization. 





# Old load data
# Before Gavin re-generated with new names H001, H002, H003
# hdic = {'01':'validate/validate.v3alpha02.2023102',
#         '02':'validate/validate.opt_params_dnet-1.5_reweight_mincdnc12.5e6_20240423132337',
#         '03':'dnet-1.5_RESTOM2.5'
#         }
# ldic = {'01':'ens/workdir.293/20230802.v3alpha02.F2010.pmcpu.intel.8N',
#         '02':'dnet-2.1_RESTOM-0.5_rw2',
#         '03':'dnet-2.1_RESTOM-0.5_highR2'}
