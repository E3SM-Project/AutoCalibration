import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
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
from cartopy.util import add_cyclic_point
import cmocean
from fns.autotuning_fns_module import zonal_means_native
from fns.autotuning_fns_module import reshape
from fns.autotuning_fns_module import load_reshape_save, fix_lon
from csv_to_xarray_fns import ds_to_array

def fix_lon(lon):
    return np.where(lon > 180, lon - 360, lon)


def plot_zm( ens, obs, season, saveroot, var_list = ['SWCF','LWCF','PRECT','TREFHT','PSL','Z500','U200','U850'], hm = None, valid_lo=None, valid_hi=None, ctrl=None, lo_p3=None, hi_p3=None ):

    # Zonal means.

    # quickly add an annual mean to obs. The others have it. 
    obs['ANN'] = xr.concat( [obs['DJF'],obs['MAM'],obs['JJA'],obs['SON']],dim='time').mean(dim='time',skipna=True)
                            
    plt.ion() 
    ncolumns_plot = 3
    if len( var_list) < 3 :
        ncolumns_plot =  len( var_list) 
    fig_z, axs_z = plt.subplots(nrows= int( np.ceil( len(var_list)/ ncolumns_plot)) , ncols=ncolumns_plot,figsize=(10,5))
    axs_z = axs_z.flatten()
    for ax in axs_z:
        ax.set_axis_off()
    
    for i in range(len(var_list)):
        v = var_list[i]
        alpha = 0.5
        axs_z[i].set_axis_on()
        pl_m = axs_z[i].plot( obs[season][v].lat, np.nanmean( ens[season][v]['mod'], axis=-1).squeeze().transpose(),'lightgrey',label='E3SMv3 PPE')
        if valid_lo:
            pl_lo = axs_z[i].plot( obs[season][v].lat, np.nanmean( valid_lo[season][v]['mod'], axis=-1).squeeze().transpose(),'c', label='Low ECS',alpha=alpha)
        if valid_hi:
            pl_hi = axs_z[i].plot( obs[season][v].lat, np.nanmean( valid_hi[season][v]['mod'], axis=-1).squeeze().transpose(),'r', label='High ECS',alpha=alpha)
        if lo_p3:
            p3_lo = axs_z[i].plot( obs[season][v].lat, np.nanmean( lo_p3[season][v]['mod'], axis=-1).squeeze().transpose(),'c', label='Low mincdnc',alpha=alpha)
        if hi_p3:
            p3_hi = axs_z[i].plot( obs[season][v].lat, np.nanmean( hi_p3[season][v]['mod'], axis=-1).squeeze().transpose(),'r', label='High mincdnc',alpha=alpha)
        if ctrl:
            pl_c = axs_z[i].plot( obs[season][v].lat, np.nanmean( ctrl[season][v]['mod'], axis=-1).squeeze().transpose(),'purple',label='E3SMv3',linewidth=3)
            
        #pl_s = axs_z[row].plot( obs[season][v].lat, np.nanmean( data_rshp[season][v]['sur'], axis=-1).squeeze().transpose(),'r',label='surrogate',alpha=alpha)
        pl_o = axs_z[i].plot( obs[season][v].lat, np.nanmean( obs[season][v], axis=-1).squeeze(),'k-',label='Obs',linewidth=2)


        props = dict( facecolor='white', alpha=0.5)
        # place a text box in upper left in axes coords
        axs_z[i].text(0.04, 0.96,v , transform=axs_z[i].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        axs_z[i].set_xlabel('Latitude')
        #axs_z[i].yaxis.tick_right()
        #axs_z[i].label_outer()

    handles, labels = axs_z[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_z.legend(by_label.values(), by_label.keys(),ncol=1, loc='right')

    plt.suptitle(f'{season}')
    plt.subplots_adjust( right=0.85, wspace = 0.3)
    #fig_z.tight_layout()
    plt.savefig(f'png/{saveroot}_{season}.png')


def plot_contour( case_data_list, case_names_list, v, subtract_control=True ):
    # Figure 2. Full fields. Could plot surrogate predicted vs modeled for two vastly different cases, like in the E3SMv2 paper. Will choose SWCF for the high and low ECS cases. Will subtract ctrl to make the differences more apparent. Looks good for SWCF. COuld potentially choose two members at the extremes of the SWCF range to better show it. 
    # Top Row: L1
    # Bottom row: H3
    # Left: E3SM prediction
    # Right: Surrogate prediction. 

    # quickly add an annual mean to obs. The others have it. Not using it at the moment but could add as a 3rd column.
    # Currently only using obs to pluck its lat, lon, and lev, which are missing from my reshape fn. 

    lat, lon, lev  = all_cases.lat, all_cases.lon, all_cases.lev
    lev_dic = {'SWCF': {'cm': 'cmo.ice','un':'Wm$^{-2}$'},
               'LWCF': {'cm': 'hot_r','un':'Wm$^{-2}$'},
               'PRECT': {'cm': 'WhiteBlueGreenYellowRed','un':'mm day$^{-1}$'}}
    ncolumns_plot = 2
    proj = crs.Robinson()
    fig, axs = plt.subplots(nrows=len(case_data_list),ncols=ncolumns_plot,
                    subplot_kw={'projection': proj},
                    figsize=(6.4,3.8))

    if lev_dic[v]['cm']:
        if v=='PRECT':
            rgb = np.loadtxt('WhiteBlueGreenYellowRed.rgb', skiprows=2)
            wbgyr = colors.LinearSegmentedColormap.from_list('wbgyr', rgb/256, N=256)
            cmap = wbgyr
        else:
            cmap = plt.get_cmap(lev_dic[v]['cm'])
    else:
        cmap = plt.get_cmap('magma')
    cmap.set_bad(color='grey')
    # Establish common vmin and vmax for all subplots
    vmin,vmax = np.nan, np.nan
    for case in case_data_list:
        sur, mod, obs  = case.sel(product='sur'), case.sel(product='mod'), case.sel(product='obs')
        sur_c, mod_c = all_cases[v].sel(ens_idx='ctrl').sel(product='sur').sel(time='ANN'), all_cases[v].sel(ens_idx='ctrl').sel(product='mod').sel(time='ANN')
        if not subtract_control:
            sur_c, mod_c = sur_c * 0, mod_c * 0
        low,high = np.nanmin( [sur - sur_c, mod - mod_c]), np.nanmax( [sur - sur_c, mod - mod_c]) 
        vmin,vmax = np.nanmin( [vmin, low]), np.nanmax( [vmax, high])

    if subtract_control:
        vmin = -1 * np.max( np.abs( [vmin, vmax]))
        vmax = np.max( np.abs( [vmin, vmax]))
        levels = np.linspace(vmin, vmax, 8)
        cmap  = 'cmo.balance'
        if v=='PRECT':
            cmap = 'BrBG'
    else:
        levels = np.linspace(vmin, vmax, 20)
    obs, cyclic_lons =  add_cyclic_point(obs, coord=lon)
    sur_c, cyclic_lons =  add_cyclic_point(sur_c, coord=lon)
    mod_c, cyclic_lons =  add_cyclic_point(mod_c, coord=lon)
    for row in range(len(case_data_list)):
        case = case_data_list[row]
        sur, cyclic_lons  = add_cyclic_point(case.sel(product='sur'), coord=lon)
        mod, cyclic_lons =  add_cyclic_point(case.sel(product='mod'), coord=lon)
        if subtract_control:
            # Replacing contourf with pcolor to show the resolution. 
            #pl = axs[row,0].contourf(cyclic_lons , lat,  mod - mod_c, cmap=cmap, vmin = vmin, vmax = vmax, levels = levels, transform = crs.PlateCarree())
            #pl = axs[row,1].contourf(cyclic_lons , lat,  sur - sur_c, cmap=cmap, vmin = vmin, vmax = vmax, levels = levels, transform = crs.PlateCarree())
            pl = axs[row,0].pcolor(cyclic_lons , lat,  mod - mod_c, cmap=cmap, vmin = vmin, vmax = vmax,  transform = crs.PlateCarree(),shading='auto')
            pl = axs[row,1].pcolor(cyclic_lons , lat,  sur - sur_c, cmap=cmap, vmin = vmin, vmax = vmax,  transform = crs.PlateCarree(),shading='auto')
            axs[row,0].set_title(f'E3SMv3 {case_names_list[row]} - Control')
            axs[row,1].set_title(f'Surrogate {case_names_list[row]} - Control')
        else:
            #pl = axs[row,0].contourf(cyclic_lons , lat,  mod, cmap=cmap, vmin = vmin, vmax = vmax, levels = levels, transform = crs.PlateCarree())
            pl = axs[row,0].pcolor(cyclic_lons , lat,  mod, cmap=cmap, vmin = vmin, vmax = vmax, transform = crs.PlateCarree(),shading='auto')
            #axs[row,0].text(0.9,0.9,round(np.nanmean(mod),1), transform=axs[row,0].transAxes)  # uncomment to print un-weighted global mean.
            #pl = axs[row,1].contourf(cyclic_lons , lat,  sur, cmap=cmap, vmin = vmin, vmax = vmax, levels = levels, transform = crs.PlateCarree())
            pl = axs[row,1].pcolor(cyclic_lons , lat,  sur, cmap=cmap, vmin = vmin, vmax = vmax,  transform = crs.PlateCarree(),shading='auto')
            #axs[row,1].text(0.9,0.9,round(np.nanmean(sur),1), transform=axs[row,1].transAxes)
            axs[row,0].set_title(f'E3SMv3 {case_names_list[row]}')
            axs[row,1].set_title(f'Surrogate {case_names_list[row]}')

            
    for a in axs.ravel():
        a.coastlines()
        #gl = a.gridlines(linewidth=1, color='black', alpha=0.2, linestyle="--")
        # gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
        # gl.xlabels_top = False
        # gl.ylabels_right = False
        # gl.ylabels_left = True

    fig.suptitle(v + ' '+ lev_dic[v]['un'], fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(pl, cax=cbar_ax, )
    #plt.subplots_adjust(hspace=0.1)
    savename = f'png/contour_{v}.png'
    if subtract_control:
        v += '_minus_ctrl'
    if 'Min' in case_names_list[0]:
        savename = f'png/contour_{v}_ens_min_max.png'

    plt.savefig(savename)

    ## OLD BUT KEEPING SAMPLE OF PLOTTING SCREAM DATA ON IRREGULAR GRID ## 

    #pl_m = axs[row,0].tripcolor(fix_lon(mod['lon'][0,:].values.astype('float64')),mod['lat'][0,:].values.astype('float64'),mod[v][0,:].values.astype('float64'),vmin=lev_dic[v]['mm'][0], vmax=lev_dic[v]['mm'][1],  transform = crs.PlateCarree(),cmap=cmap)




# PRoduces two box plots. One with the separation of each run type. one with all run types consolidated but with L1-H3 separate
# Box plot of 1) scale global mean by obs for each var. 2) scale RMSE Relative to control
# Produce two plots: One scaled by obs for when obs are available. Another scaled by ctrl when no or multiple obs were used. 
def plot_box_figs(all_cases):
    # Data
    vars_exclude = ['params','area', 'RESTOM', 'SWCRE_ano_grd_adj','LWCRE_ano_grd_adj','dnet_cld_dir']
    ctrl_data = all_cases.where( all_cases.workdir.str.contains('ctrl'), drop=True).sel(time='ANN').isel(ens_idx=0, drop=True).sel(product='mod', drop=True)
    obs = all_cases.sel(product='obs', drop=True).sel(time='ANN', drop=True).isel(ens_idx=0, drop=True)
    obs_wgt  = obs.weighted(weights=all_cases.area)
    obs_gm = obs_wgt.mean(['lat','lon','lev'])
    all_wgt = all_cases.sel(time='ANN', drop=True).sel(product='mod').weighted(weights=all_cases.area)
    all_gm = all_wgt.mean(['lat','lon','lev'])
    all_scaled = ((all_gm - obs_gm) / obs_gm).drop_vars(vars_exclude)

    arr_all = ds_to_array(all_scaled)
    arr_ens = ds_to_array(all_scaled.where( all_scaled.workdir.str.contains('ens'), drop=True))
    arr_hm  = ds_to_array(all_scaled.where( all_scaled.workdir.str.contains('hm'), drop=True))
    arr_val  = ds_to_array(all_scaled.where( (all_scaled.workdir.str.contains('valid') | all_scaled.workdir.str.contains('dnet') ), drop=True))
    
    i_sort = np.argsort(np.mean( arr_ens, -1 ))
    arr_varnames = np.array(list(all_scaled.keys()))

    # start plots
    common_figsize=(7,8)
    fig, axs = plt.subplots(nrows= 3 , ncols=1, figsize=common_figsize)

    ax = axs[0]
    ticks = np.arange(0, len( arr_ens ))
    boxticks = ticks + 1
    showfliers=True; showcaps=False; sym='.'
    offset=0.22; alpha=0.5; wid=0.15
    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [0, 0],'--', color='grey')
    bp0 = ax.boxplot( arr_ens[i_sort].transpose(),positions=boxticks-offset,widths=wid,patch_artist=True, showfliers=showfliers, showcaps=showcaps, sym=sym)
    bp1 = ax.boxplot( arr_hm[i_sort].transpose(),positions=boxticks, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    bp2 = ax.boxplot( arr_val[i_sort].transpose(),positions=boxticks+offset, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    for box in bp0['boxes']:
        box.set(edgecolor = 'black',facecolor='none' )
    for box in bp1['boxes']:
        box.set(edgecolor = 'red' ,facecolor='none')
    for box in bp2['boxes']:
        box.set(edgecolor = 'green' ,facecolor='none')
        
        
    ax.set_xticks( boxticks)
    ax.set_xticklabels([] )
    ctrl_scaled = all_scaled.where( all_scaled.workdir.str.contains('ctrl'), drop=True).to_array()
    bpc = ax.scatter(boxticks-offset*1.5, ctrl_scaled[i_sort],facecolors='none',edgecolor='black' , marker='D', s=30)
    ax.set_ylabel('Standardized global mean bias')
    #ax.legend([bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0]], ['LHS', 'HM', 'Validate'])
    ax.legend([bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0], bpc], ['LHS', 'HM', 'Validate', 'Ctrl'])

    ax = axs[1]

    se = ((all_cases.sel(time='ANN', drop=True).sel(product='mod') - obs).drop_vars(vars_exclude)**2)
    se_wgt = se.weighted(weights=all_cases.area)
    rmse = se_wgt.mean(['lat','lon','lev'] )**(1/2)
    rmse_ctrl = rmse.where( all_scaled.workdir.str.contains('ctrl'), drop=True).to_array()
    rmse_scaled_by_ctrl = (rmse.to_array() / rmse_ctrl.values).to_dataset(name='whatever')
    arr_rmse_scaled_all = ds_to_array(rmse_scaled_by_ctrl)
    arr_ens = ds_to_array(rmse_scaled_by_ctrl.where( rmse_scaled_by_ctrl.workdir.str.contains('ens'), drop=True))
    arr_hm  = ds_to_array(rmse_scaled_by_ctrl.where( rmse_scaled_by_ctrl.workdir.str.contains('hm'), drop=True))
    arr_val = ds_to_array(rmse_scaled_by_ctrl.where( (rmse_scaled_by_ctrl.workdir.str.contains('valid') | rmse_scaled_by_ctrl.workdir.str.contains('dnet') ), drop=True))
    
    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [1, 1],'--', color='grey')
    bp0 = ax.boxplot( arr_ens[i_sort].transpose(),positions=boxticks-offset,widths=wid,patch_artist=True, showfliers=showfliers,showcaps=showcaps, sym=sym)
    bp1 = ax.boxplot( arr_hm[i_sort].transpose(),positions=boxticks, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    bp2 = ax.boxplot( arr_val[i_sort].transpose(),positions=boxticks+offset, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    for box in bp0['boxes']:
        box.set(edgecolor = 'black',facecolor='none' )
    for box in bp1['boxes']:
        box.set(edgecolor = 'red' ,facecolor='none')
    for box in bp2['boxes']:
        box.set(edgecolor = 'green' ,facecolor='none')

    ax.set_xticks( boxticks)
    ax.set_xticklabels([] )
    rmse_ctrl_scaled_by_ctrl = rmse_ctrl / rmse_ctrl  # all 1 by design. 
    #bpc = ax.scatter(boxticks-offset*1.5, rmse_ctrl_scaled_by_ctrl[i_sort],facecolors='none',edgecolor='black' , marker='D', s=30)
    ax.set_ylabel('Standardized RMSE')
    #ax.legend([bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0]], ['LHS', 'HM', 'Validate'])
    #ax.tight_layout()

    ax = axs[2]

    all_gridpoint = all_cases.sel(time='ANN', drop=True).sel(product='mod')
    corr_ds = xr.Dataset()
    print('computing spatial correlation')
    for v in all_gridpoint:
        if ('lat' in all_gridpoint[v].dims) & ('ens_idx' in all_gridpoint[v].dims):
            case_corrs = []
            for case in all_gridpoint.ens_idx:
                moddat = all_gridpoint[v].sel(ens_idx=case).data.flatten()
                obsdat = obs[v].data.flatten()
                mask = (~np.isnan(moddat)) & (~np.isnan(obsdat))
                cc = np.corrcoef(moddat[mask], obsdat[mask])[0,-1]
                case_corrs.append(cc)
            all_cases_c = xr.DataArray(case_corrs, dims=['ens_idx'])
            corr_ds[v]=all_cases_c
            del(all_cases_c)
            
    arr_all_corr = ds_to_array(corr_ds)
    arr_ens = ds_to_array(corr_ds.where( all_scaled.workdir.str.contains('ens'), drop=True))
    arr_hm  = ds_to_array(corr_ds.where( all_scaled.workdir.str.contains('hm'), drop=True))
    arr_val  = ds_to_array(corr_ds.where( (all_scaled.workdir.str.contains('valid') | all_scaled.workdir.str.contains('dnet') ), drop=True))

    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [1, 1],'--', color='grey')
    bp0 = ax.boxplot( arr_ens[i_sort].transpose(),positions=boxticks-offset,widths=wid,patch_artist=True, showfliers=showfliers,showcaps=showcaps, sym=sym)
    bp1 = ax.boxplot( arr_hm[i_sort].transpose(),positions=boxticks, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    bp2 = ax.boxplot( arr_val[i_sort].transpose(),positions=boxticks+offset, widths=wid,patch_artist=True,showfliers=showfliers,showcaps=showcaps, sym=sym)
    for box in bp0['boxes']:
        box.set(edgecolor = 'black',facecolor='none' )
    for box in bp1['boxes']:
        box.set(edgecolor = 'red' ,facecolor='none')
    for box in bp2['boxes']:
        box.set(edgecolor = 'green' ,facecolor='none')

    ax.set_xticks( boxticks)
    ax.set_xticklabels(  arr_varnames[i_sort], rotation=45, fontsize=8)
    corr_ctrl = corr_ds.where( all_scaled.workdir.str.contains('ctrl'), drop=True).to_array()
    bpc = ax.scatter(boxticks-offset*1.5, corr_ctrl[i_sort],facecolors='none',edgecolor='black' , marker='D', s=30)
    ax.set_ylabel('Spatial Correlation')
    #ax.legend([bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0]], ['LHS', 'HM', 'Validate'])
    fig.tight_layout()
    plt.savefig('pdf/gm_bias_rmse_corr_boxplot.pdf')


    # Figure 2. 
    # Consolidate all simulations, Add L1 to H3 and save as a different filename. 
    fig2, axs2 = plt.subplots(nrows= 3 , ncols=1, figsize=common_figsize)


    # Keeping it down to L1 and H3 only for clutter. 
    d_opt = {'L1': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('L001'),drop=True)), 
    #         'L2': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('L002'),drop=True)),
    #         'L3': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('L003'),drop=True)), 
    #         'H1': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('H001'),drop=True)), 
    #         'H2': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('H002'),drop=True)), 
             'H3': ds_to_array( all_scaled.where(all_cases.workdir.str.contains('H003'),drop=True))}

    ax = axs2[0]
    offset_orig = 0.15
    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [0, 0],'--', color='grey')
    bp0 = ax.boxplot( arr_all[i_sort].transpose(),positions=boxticks,widths=wid,patch_artist=True, showfliers=showfliers, showcaps=showcaps, sym=sym)
    for box in bp0['boxes']:
        box.set(edgecolor = 'grey',facecolor='none' )
    bpc = ax.scatter(boxticks, ctrl_scaled[i_sort],facecolors='none',edgecolor='black' , marker='D', s=30)
    # for opt in [arrl1, arrl2, arrl3, arrh1, arrh2, arrh3]:
    #     o = ax.scatter(boxticks+offset, opt[i_sort],facecolors='none',edgecolor='k' , marker='>', s=30)
    for opt in d_opt.keys():
        color = 'blue'
        offset = offset_orig
        if 'H' in opt:
            color='red'; offset = -1.75*offset_orig;
        for i in range(len(boxticks)):
            ax.text( boxticks[i]+offset, d_opt[opt][i_sort][i], s=opt.replace("L", "").replace("H", ""), color=color, verticalalignment='center')
    ax.set_ylabel('Standardized global mean bias')

    ax = axs2[1]
    d_rmse = {'L1': ds_to_array(rmse_scaled_by_ctrl.where( rmse_scaled_by_ctrl.workdir.str.contains('L001'), drop=True)),
              'H3': ds_to_array(rmse_scaled_by_ctrl.where( rmse_scaled_by_ctrl.workdir.str.contains('H003'), drop=True))}
    
    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [1, 1],'--', color='grey')
    bp = ax.boxplot( arr_rmse_scaled_all [i_sort].transpose(),positions=boxticks,widths=wid,patch_artist=True, showfliers=showfliers,showcaps=showcaps, sym=sym)
    for box in bp['boxes']:
        box.set(edgecolor = 'grey',facecolor='none' )
    for opt in d_rmse.keys():
        color = 'blue'
        offset = offset_orig
        if 'H' in opt:
            color='red'; offset = -1.75*offset_orig;
        for i in range(len(boxticks)):
            ax.text( boxticks[i]+offset, d_rmse[opt][i_sort][i], s=opt.replace("L", "").replace("H", ""), color=color, verticalalignment='center')
    ax.set_ylabel('Standardized RMSE')

    ax = axs2[2]
    ax.plot([boxticks[0]-0.5, boxticks[-1]+0.5], [1, 1],'--', color='grey')
    
    d_corr = {'L1':ds_to_array(corr_ds.where(all_cases.workdir=='L001', drop=True)),
              'H3':ds_to_array(corr_ds.where(all_cases.workdir=='H003', drop=True ))}
    

    bp = ax.boxplot( arr_all_corr[i_sort].transpose(),positions=boxticks,widths=wid,patch_artist=True, showfliers=showfliers,showcaps=showcaps, sym=sym)
    for box in bp['boxes']:
        box.set(edgecolor = 'grey',facecolor='none' )
    for opt in d_corr.keys():
        color = 'blue'
        offset = offset_orig
        if 'H' in opt:
            color='red'; offset = -1.75*offset_orig;
        for i in range(len(boxticks)):
            ax.text( boxticks[i]+offset, d_corr[opt][i_sort][i], s=opt.replace("L", "").replace("H", ""), color=color, verticalalignment='center')
    bpc = ax.scatter(boxticks, corr_ctrl[i_sort],facecolors='none',edgecolor='black' , marker='D', s=30)
    ax.set_ylabel('Spatial Correlation')
    ax.set_xticks( boxticks)
    ax.set_xticklabels(  arr_varnames[i_sort], rotation=45, fontsize=8)

    plt.savefig('pdf/gm_bias_rmse_corr_boxplot_optcases.pdf')
    pdb.set_trace()
    
        
    

    
# Load simulations ( in all_cases )
# Load surrogate ( in all_cases )
# Load obs (obs)

# Full table here: https://acme-climate.atlassian.net/wiki/spaces/NGDSA/pages/4414341121/2024-06-24+Meeting+notes

all_cases, all_cases_orig, obs = load_reshape_save('H003', clobber=False)

ens = all_cases.where( all_cases.workdir.str.contains('ens'), drop=True) 
hm  = all_cases.where( all_cases.workdir.str.contains('hm'), drop=True)
valid= all_cases.where( (all_cases.workdir.str.contains('valid')   | all_cases.workdir.str.contains('dnet') ), drop=True)
valid_hi = all_cases.where( (all_cases.workdir.str.match('H001') |
                             all_cases.workdir.str.match('H002') |
                             all_cases.workdir.str.match('H003')), drop=True)
valid_lo = all_cases.where( (all_cases.workdir.str.match('L001') |
                             all_cases.workdir.str.match('L002') |
                             all_cases.workdir.str.match('L003')), drop=True)
l1 = all_cases.where( (all_cases.workdir.str.match('L001')), drop=True)
h3 = all_cases.where( (all_cases.workdir.str.match('H003')), drop=True)
lo_p3 = all_cases.where( all_cases.params[:,13] < 5e6 , drop=True)
hi_p3 = all_cases.where( all_cases.params[:,13] > 5e6 , drop=True)
ctrl = all_cases.where( all_cases.workdir.str.contains('ctrl'), drop=True)


# Find the extreme members for precip, swcf
extreme_d = {'SWCF':{},'LWCF':{},'PRECT':{}}
for v in extreme_d:
    max_i =  np.argmax( all_cases[v].sel(time='ANN').sel(product='mod').mean(dim=('lat','lon'),skipna=True) )
    min_i =  np.argmax( all_cases[v].sel(time='ANN').sel(product='mod').mean(dim=('lat','lon'),skipna=True) )
    extreme_d[v]['max_data'] = all_cases[v].sel(time='ANN').isel(ens_idx=max_i.data)
    extreme_d[v]['min_data'] = all_cases[v].sel(time='ANN').isel(ens_idx=min_i.data)
    # working. max swcf = -28
    
#var_list = ['PSL','TREFHT','Z500','U200','U850','RELHUM','T','U']
for sn in ['ANN']:
    plot_box_figs(all_cases)
    plot_contour( [extreme_d['PRECT']['min_data'], extreme_d['PRECT']['max_data'] ], ['Ens. Min','Ens. Max'], 'PRECT', subtract_control=True) # Need to finish remaking these since changed load data. 
    plot_contour( [extreme_d['SWCF']['min_data'], extreme_d['SWCF']['max_data'] ], ['Ens. Min','Ens. Max'],   'SWCF', subtract_control=True)
    plot_contour( [extreme_d['LWCF']['min_data'], extreme_d['LWCF']['max_data'] ], ['Ens. Min','Ens. Max'],   'LWCF', subtract_control=True)
    plot_contour( [extreme_d['SWCF']['min_data'], extreme_d['SWCF']['max_data'] ], ['Ens. Min','Ens. Max'],   'SWCF', subtract_control=False)
    plot_contour( [extreme_d['LWCF']['min_data'], extreme_d['LWCF']['max_data'] ], ['Ens. Min','Ens. Max'],   'LWCF', subtract_control=False)
    plot_contour( [extreme_d['PRECT']['min_data'], extreme_d['PRECT']['max_data'] ], ['Ens. Min','Ens. Max'], 'PRECT', subtract_control=False)
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],  'LWCF')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],  'SWCF')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],  'PRECT')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],  'RELHUM')
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl_p3sort', ctrl = ctrl_rshp_data, lo_p3 = lo_p3_rshp_data,  var_list = ['SWCF','LWCF'])
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl', ctrl = ctrl_rshp_data )
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl_lo_hi', var_list = ['SWCF','LWCF'], valid_lo = valid_lo_rshp_data, valid_hi=valid_hi_rshp_data, ctrl = ctrl_rshp_data )



#plot_zm( ens_rshp_data, obs , 'MAM', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )
#plot_zm( ens_rshp_data, obs , 'JJA', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )
#plot_zm( ens_rshp_data, obs , 'SON', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )



