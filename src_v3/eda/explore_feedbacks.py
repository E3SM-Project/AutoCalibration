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
from fns.autotuning_fns_module import load_reshape_save, fix_lon


# This code reads spatial fields of SWCF and LWCF from the merged files that are updated to include feedbacks.
# The merged files are on perlmutter
# -within each target dir (until purge)
# -or in /global/cfs/cdirs/e3sm/emulate/postproc_e3sm/v3_merged_targets/ ~500 MB
# I also rsynced to my local machine
# Autotuning-NGD/src/eda/v3/rsync_e3sm_data/v3_merged_targets 

# The surrogate only predicted dnet; surrogate's model output data alone insufficient for this script.
# Therefore, additionally load 24x48 feedback files. 

def zm( all_cases, valid_lo, valid_hi ):
    # Zonal means.
    p3_lo=False
    p3_hi=False
    ctrl = True
    ylim=[-4,3]
    plt.ion()
    nrows_plot = 1
    fig_z, axs_z = plt.subplots(nrows= nrows_plot , ncols=2 ,figsize=(7,3.25))
    axs_z = axs_z.flatten()
    i=0
    label_dic = {'SWCRE_ano_grd_adj':'$\lambda_{SWcld}$  Wm$^{-2}$K$^{-1}$',
                 'LWCRE_ano_grd_adj':'$\lambda_{LWcld}$  Wm$^{-2}$K$^{-1}$'}
    for v in ['SWCRE_ano_grd_adj', 'LWCRE_ano_grd_adj']:
        alpha = 0.3
        axs_z[i].set_axis_on()
        pl_m = axs_z[i].plot( all_cases[v].lat, all_cases[v].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'lightgrey',label='E3SMv3 PPE')
        if valid_lo:
            pl_lo = axs_z[i].plot( all_cases[v].lat, valid_lo[v].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'b--', label='Low ECS',alpha=alpha)
            pl_l1 = axs_z[i].plot( all_cases[v].lat, valid_lo.sel(ens_idx='L001')[v].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'b', label='Low ECS')
            
        if valid_hi:
            pl_hi = axs_z[i].plot( all_cases[v].lat, valid_hi[v].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'r--', label='High ECS',alpha=alpha)
            pl_l1 = axs_z[i].plot( all_cases[v].lat, valid_hi.sel(ens_idx='H003')[v].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'r', label='High ECS')

        #if lo_p3:
            #p3_lo = axs_z[i].plot( obs[season][v].lat, np.nanmean( lo_p3[season][v]['mod'], axis=-1).squeeze().transpose(),'c', label='Low mincdnc',alpha=alpha)
        #if hi_p3:
            #p3_hi = axs_z[i].plot( obs[season][v].lat, np.nanmean( hi_p3[season][v]['mod'], axis=-1).squeeze().transpose(),'r', label='High mincdnc',alpha=alpha)
        if ctrl:
            ctrl = all_cases.where(all_cases.ens_idx.str.match( 'ctrl' )).sel(time='ANN').sel(product='mod')
            pl_c =axs_z[i].plot( all_cases[v].lat, ctrl[v].weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean("lon").transpose(),'k-',label='Ctrl',linewidth=2) 

        axs_z[i].set_xlabel('Latitude')
        axs_z[i].set_xticks(np.linspace(-90,90,7))

        axs_z[i].set_ylim(ylim)
        props = dict( facecolor='white', alpha=0.5)
        # place a text box in upper left in axes coords
        axs_z[i].text(0.04, 0.96,label_dic[v] , transform=axs_z[i].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        i+=1
    handles, labels = axs_z[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs_z[1].legend(by_label.values(), by_label.keys(),ncol=2, loc='lower right')
    #plt.subplots_adjust( right=0.85, wspace = 0.3)
    fig_z.tight_layout()
    plt.savefig(f'pdf/zm_feedbacks.pdf')


def plot_contour( case_data_list, case_names_list, obs, ctrl, v, subtract_control=True ):
    # Figure 2. Full fields. Could plot surrogate predicted vs modeled for two vastly different cases, like in the E3SMv2 paper. Will choose SWCF for the high and low ECS cases. Will subtract ctrl to make the differences more apparent. Looks good for SWCF. COuld potentially choose two members at the extremes of the SWCF range to better show it. 
    # Top Row: L1
    # Bottom row: H3
    # Left: E3SM prediction
    # Right: Surrogate prediction. 

    # quickly add an annual mean to obs. The others have it. Not using it at the moment but could add as a 3rd column.
    # Currently only using obs to pluck its lat, lon, and lev, which are missing from my reshape fn. 
    obs['ANN'] = xr.concat( [obs['DJF'],obs['MAM'],obs['JJA'],obs['SON']],dim='time').mean(dim='time',skipna=True)
    lat, lon, lev  = obs['ANN'].lat, obs['ANN'].lon, obs['ANN'].lev
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
        sur, mod, ob, sur_c, mod_c = case['ANN'][v]['sur'][0], case['ANN'][v]['mod'][0], obs['ANN'][v], ctrl['ANN'][v]['sur'][0], ctrl['ANN'][v]['mod'][0]
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
    obs, cyclic_lons =  add_cyclic_point(obs['ANN'][v], coord=lon)
    sur_c, cyclic_lons =  add_cyclic_point(ctrl['ANN'][v]['sur'][0], coord=lon)
    mod_c, cyclic_lons =  add_cyclic_point(ctrl['ANN'][v]['mod'][0], coord=lon)
    for row in range(len(case_data_list)):
        case = case_data_list[row]
        sur, cyclic_lons  = add_cyclic_point(case['ANN'][v]['sur'][0], coord=lon)
        mod, cyclic_lons =  add_cyclic_point(case['ANN'][v]['mod'][0], coord=lon)
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


def scat(swfd_mean, lwfd_mean, all_cases, valid_lo, valid_hi):
    fig, axs = plt.subplots(nrows= 1 , ncols=3,figsize=(9,3.25))
    ax=axs[0]
    ax.axis('equal')
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean + lwfd_mean, alpha=0.5, facecolors='grey', edgecolors='grey')
    for case in xr.merge( [valid_lo, valid_hi]).workdir:
        dnet = all_cases.where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        cf   = (swfd_mean + lwfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red'
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        ax.text(dnet.dnet_cld_dir.sel(product='mod').sel(time='ANN'), cf, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center', fontweight='bold')
        
        
        
    ctrl = all_cases.where(all_cases.ens_idx.str.match('ctrl'), drop=True)
    ax.scatter(ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), (swfd_mean + lwfd_mean).where(all_cases.ens_idx.str.match('ctrl'), drop=True), color='k', marker='D', s=100)
    ax.set_xlabel('$\lambda$ Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel('$\lambda_{cld}$ Wm$^{-2}$K$^{-1}$')
    
    ax=axs[1]
    ax.axis('equal')
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean, label='Shortwave', alpha=0.5, facecolors='grey', edgecolor='grey' )
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), lwfd_mean, label='Longwave', marker='.', color='k')
    ax.scatter( ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean.where(all_cases.ens_idx.str.match('ctrl'), drop=True),facecolors='grey',edgecolor='white', marker='D' , s=100)
    ax.scatter( ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), lwfd_mean.where(all_cases.ens_idx.str.match('ctrl'), drop=True),facecolors='black',edgecolor='white' , marker='D', s=100)
    
    ax.set_xlabel('$\lambda$ Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel('$\lambda_{cld}$ Wm$^{-2}$K$^{-1}$')
    ax.legend()

    ax=axs[2]
    ax.axis('equal')
    ## Draw diagonal lines
    for pos in np.linspace(-2, 2, 11):
        plt.axline((pos, 0), slope=-1, color='grey', alpha=0.3)
    ax.scatter(swfd_mean, lwfd_mean, c=all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), alpha=0.5, label='$\lambda_{Net}$')
    for case in xr.merge( [valid_lo, valid_hi]).workdir:
        swfd   = (swfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        lwfd   = (lwfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red';
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        ax.text(swfd, lwfd, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center',fontweight='bold')
    ax.scatter(swfd_mean.where(swfd_mean.ens_idx.str.match( 'ctrl' )), lwfd_mean.where(lwfd_mean.ens_idx.str.match( 'ctrl' )), color='k', marker='D', s=100)
    ax.set(xlim=(-0.75, 0.75),ylim=(-0.75, 0.75))
    ax.set_xlabel('$\lambda_{SWcld}$  Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel( '$\lambda_{LWcld}$  Wm$^{-2}$K$^{-1}$' )
    fig.tight_layout()
    plt.savefig('pdf/scat_feedbacks.pdf')


def scat_swcf_vs_swfeed(all_cases, valid_lo, valid_hi):
    plt.figure(figsize=(4.75,3.25))
    #plt.scatter( all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean(['lat','lon']), all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean(['lat','lon']) , color='grey',alpha=0.4  )
    restom = all_cases['RESTOM'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon'])
    #cmap  = 'cmo.balance'
    #cmap = plt.get_cmap('coolwarm',3)
    cmap = plt.get_cmap('coolwarm',8)

    plt.scatter( all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']), all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']) ,alpha=0.6 , c = restom, edgecolor='grey', cmap=cmap, vmin=-12, vmax=12)
    cbar = plt.colorbar(extend='max')
    #cbar.set_ticks([-12, -8,-4, 0, 4, 8, 12])
    cbar.set_ticks([-12, -9, -6, -3, 0, 3, 6, 9, 12])
    

    cbar.ax.set_title('RESTOM  Wm$^{-2}$')        
    sw   = all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)

    for case in xr.merge( [valid_lo, valid_hi]).workdir:
        swfd = all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red'
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        plt.text(sw, swfd, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                 color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center', fontsize=15,fontweight='bold')
    sw   = all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( 'ctrl'), drop=True)
    swfd = all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( 'ctrl' ), drop=True)
    plt.scatter( sw, swfd,facecolors='black',edgecolor='white' , marker='D', s=100)
    lims=plt.gca().get_ylim()
    obs_swcf = all_cases['SWCF'].sel(time='ANN').sel(product='obs').weighted(all_cases['area']).mean(['lat','lon']).isel(ens_idx=0)
    plt.plot([obs_swcf, obs_swcf], [lims[0],lims[1]],'k--', label='CERES')
    plt.ylim(lims)
    plt.legend(loc='lower right')
    plt.ylabel( '$\lambda_{SWcld}$  Wm$^{-2}$K$^{-1}$' )
    plt.xlabel( 'SWCF Wm$^{-2}$' )
    plt.tight_layout()

    plt.savefig('pdf/scat_swcf_vs_swfeed.pdf')
    pdb.set_trace()
    

    
# Load simulations ( in all_cases )
# Load surrogate ( in all_cases )
# Load obs (obs)


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



# Scatter, 1-panel, SWCFeedback vs dnet and LWCF vs dnet.
swfd_mean = (all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'])).mean(('lat','lon'))
lwfd_mean = (all_cases['LWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'])).mean(('lat','lon'))
plot_scat = True
if plot_scat:
    scat( swfd_mean, lwfd_mean, all_cases, valid_lo, valid_hi)
# Zonal mean of swfd, lwfd
plot_zm = False
if plot_zm:
    zm( all_cases, valid_lo, valid_hi)

# Plot swhortwave cloud forcing vs shortwave cloud feedback
plot_scat_swcf_vs_swfeed = True
if plot_scat_swcf_vs_swfeed:
    scat_swcf_vs_swfeed( all_cases, valid_lo, valid_hi )
    
pdb.set_trace()

# Find the extreme members for precip, swcf
extreme_d = {'SWCF':{},'LWCF':{},'PRECT':{}, 'SWCRE_ano_grd_adj':{},'LWCRE_ano_grd_adj':{}}
for v in extreme_d:
    pdb.set_trace()
    max_i = np.argmax( np.nanmean(np.nanmean( ens_rshp_data['ANN'][v]['mod'], axis= -1 ),axis=-1))
    min_i = np.argmin( np.nanmean(np.nanmean( ens_rshp_data['ANN'][v]['mod'], axis= -1 ),axis=-1))
    extreme_d[v]['min_data'], min_data = reshape( ens.where( ens['ens_idx'] == min_i, drop=True), all_var_dic, mask)
    extreme_d[v]['max_data'], max_data = reshape( ens.where( ens['ens_idx'] == max_i, drop=True), all_var_dic, mask)
    # working. max swcf = -28

#var_list = ['PSL','TREFHT','Z500','U200','U850','RELHUM','T','U']
for sn in ['ANN']:
    pdb.set_trace()
    plot_contour( [extreme_d['PRECT']['min_data'], extreme_d['PRECT']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'PRECT', subtract_control=True)
    plot_contour( [extreme_d['SWCF']['min_data'], extreme_d['SWCF']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'SWCF', subtract_control=True)
    plot_contour( [extreme_d['LWCF']['min_data'], extreme_d['LWCF']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'LWCF', subtract_control=True)
    plot_contour( [extreme_d['SWCF']['min_data'], extreme_d['SWCF']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'SWCF', subtract_control=False)
    plot_contour( [extreme_d['LWCF']['min_data'], extreme_d['LWCF']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'LWCF', subtract_control=False)
    plot_contour( [extreme_d['PRECT']['min_data'], extreme_d['PRECT']['max_data'] ], ['Ens. Min','Ens. Max'], obs, ctrl_rshp_data, 'PRECT', subtract_control=False)
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],obs, ctrl_rshp_data,  'LWCF')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],obs, ctrl_rshp_data,  'SWCF')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],obs, ctrl_rshp_data,  'PRECT')
    plot_contour( [l1_rshp_data, h3_rshp_data],['L01','H03'],obs, ctrl_rshp_data,  'RELHUM')
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl_p3sort', ctrl = ctrl_rshp_data, lo_p3 = lo_p3_rshp_data,  var_list = ['SWCF','LWCF'])
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl', ctrl = ctrl_rshp_data )
    plot_zm( ens_rshp_data, obs , sn, 'zm_ens_obs_ctrl_lo_hi', var_list = ['SWCF','LWCF'], valid_lo = valid_lo_rshp_data, valid_hi=valid_hi_rshp_data, ctrl = ctrl_rshp_data )



#plot_zm( ens_rshp_data, obs , 'MAM', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )
#plot_zm( ens_rshp_data, obs , 'JJA', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )
#plot_zm( ens_rshp_data, obs , 'SON', 'zm_ens_obs_ctrl', hm = hm_rshp_data, valid = valid_rshp_data, ctrl = ctrl_rshp_data )



