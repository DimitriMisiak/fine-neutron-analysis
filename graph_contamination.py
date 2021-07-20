#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:04:43 2020

@author: misiak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

cartoon = [
        pe.Stroke(linewidth=3, foreground='k'),
        pe.Normal(),
]
cartoon_light = [
        pe.Stroke(linewidth=2, foreground='k'),
        pe.Normal(),
]

from plot_addon import lighten_color
from stats_addon import cdf_calc
from pipeline_data_science import sigma_function

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'

### DATA
h5_data_path = '/'.join([analysis_dir, 'data_science.h5'])   
df_data = pd.read_hdf(
    h5_data_path,
    key='df',
    where=(
        'quality_cut = True'
        '& charge_conservation_cut = True'
        '& bulk_cut = True'
    )
)

### SIMU
h5_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])

source_list = ['Background', 'Calibration']
simulation_list =  ['NR', 'ER']
simulation_er_list = ['flat_ER', 'line_1keV', 'line_10keV' ]

ntot = dict()
for source in source_list:
    
    ntot[source] = dict()
    for simu in simulation_er_list:
        
        df_simu = pd.read_hdf(
            h5_simu_path,
            key='df',
            where=(
                'source = {}'
                '& simulation = {}'
            ).format(source, simu)
        )
        
        ntot[source][simu] = df_simu.shape[0]

df_simu = pd.read_hdf(
    h5_simu_path,
    key='df',
    where=(
        'trigger_cut = True'
        '& quality_cut = True'
        '& charge_conservation_cut = True'
        '& bulk_cut = True'
    )
)

# # =============================================================================
# # BINNING
# # =============================================================================
# std0 = df_data.std_energy_heat.unique().max()
# std10 = df_data.std_calib_energy_heat.unique().max()

# nsigma = 2

# bins_list = [0,]
# while ( bins_list[-1] < 50 ):
#     last_bin = bins_list[-1]
#     dx = sigma_function(last_bin, std0*nsigma, std10*nsigma)
#     bins_list.append(last_bin + dx)

# print('Bins OK')

# bins = np.array(bins_list)
# bins_width = (bins[1:] - bins[:-1])
# # bins_array = bins[:-1]
# bins_array = bins[:-1] + (bins_width) / 2

# hack_cut = ~(
#     ( (bins_array > 8) & (bins_array < 9) )
#     | ( (bins_array > 20) & (bins_array < 22) )
# )

eff_path = '/home/misiak/Analysis/NEUTRON/efficiency_dict.npy'
eff_dict = np.load(eff_path, allow_pickle=True).item()


bins = eff_dict['bins']
bins_width = (bins[1:] - bins[:-1])
bins_array = bins[:-1] + (bins_width) / 2


#%%

fig, axes = plt.subplots(
    figsize=(6.3, 5),
    nrows=2,
    ncols=2,
    sharex='col',
    sharey='row'
)

# x_background = [14133, 25089, 271422]
# x_calibration = [36610, 65312, 660798]

x_background = [9808, 25289, 214850]
x_calibration = [65943, 65158, 535453]

param_dict = dict()
param_dict['Background'] = x_background
param_dict['Calibration'] = x_calibration


data_band_cut = (
    df_data.gamma_cut 
    & ~df_data.neutron_cut 
    & ~df_data.HO_cut
)

simu_band_cut = (
    df_simu.gamma_cut 
    & ~df_simu.neutron_cut 
    & ~df_simu.HO_cut
)

contamination_result = dict()

for i, source in enumerate(source_list):
    
    contamination_result[source] = dict()
    
    df_local = df_data[
        (df_data.source == source)
        & data_band_cut
    ]

    # hist_data = np.histogram(df_local.recoil_energy_bulk, bins=bins)[0]
    hist_data = np.histogram(df_local.energy_heat, bins=bins)[0]

    axes[i,0].plot(
        bins_array,
        hist_data,
        ls='none',
        marker='.',
        color='k',
        zorder=10
    )

    model_er_gamma = 0
    color_list = ['gold',
    'forestgreen',
    'slateblue'
    ]
    
    zorder_list = [-5, -4, -3]
    
    # for simu,x in zip(simulation_er_list, param_dict[source]):
    for j in range(len(simulation_er_list)):
        
        x = param_dict[source][j]
        simu = simulation_er_list[j]
        
        df_local = df_simu[
            (df_simu.source == source)
            & (df_simu.simulation == simu)
            & simu_band_cut
        ]

        # hist_simu = np.histogram(df_local.recoil_energy_bulk, bins=bins)[0]
        hist_simu = np.histogram(df_local.energy_heat, bins=bins)[0]

        model_comp = x * hist_simu / ntot[source][simu]
        model_er_gamma += model_comp

        axes[i,0].fill_between(
            bins_array,
            model_comp,
            # color=lighten_color('coral'),
            color=color_list[j],
            zorder=zorder_list[j],
            step='mid',
            # edgecolor='orangered',
        )
        
        if j ==0:

            axes[i,0].plot(
                bins_array,
                model_comp,
                color=color_list[j],
                drawstyle='steps-mid',
            )            

    axes[i,0].errorbar(
        bins_array,
        model_er_gamma,
        yerr = (2*model_er_gamma)**0.5,
        drawstyle='steps-mid',
        color='r'
    )

    model_er_neutron = 0
    for simu,x in zip(simulation_er_list, param_dict[source]):
        
        df_local = df_simu[
            (df_simu.source == source)
            & (df_simu.simulation == simu)
            & df_simu.neutron_cut
        ]
        
        # ### check
        # plt.figure(num=source)
        
        # plt.plot(
        #     df_local.recoil_energy_bulk,
        #     df_local.quenching_bulk,
        #     ls='none',
        #     marker='.',
        #     color='b'
        # )
        # plt.plot(
        #     df_local.input_energy,
        #     df_local.quenching_bulk,
        #     ls='none',
        #     marker='.',
        #     color='r'
        # )        
        
        

        # hist_simu = np.histogram(df_local.recoil_energy_bulk, bins=bins)[0]
        hist_simu = np.histogram(df_local.energy_heat, bins=bins)[0]

        model_er_neutron += x * hist_simu / ntot[source][simu]


    # model_er_total = model_er_gamma + model_er_neutron
    model_er_total = hist_data + model_er_neutron

    contamination_result[source]['ER_background'] = model_er_total
    contamination_result[source]['NR_correction'] = model_er_neutron
    # axes[i,1].errorbar(
    #     bins_array,
    #     model_er_total,
    #     yerr = (2*model_er_total)**0.5,
    #     drawstyle='steps-mid'
    # )

    # axes[i,1].errorbar(
    #     bins_array,
    #     model_er_gamma,
    #     yerr = (2*model_er_gamma)**0.5,
    #     drawstyle='steps-mid'
    # )
    
    # axes[i,1].errorbar(
    #     bins_array,
    #     model_er_neutron,
    #     yerr = (2*model_er_neutron)**0.5,
    #     drawstyle='steps-mid'
    # )

    # axes[i,1].plot(
    #     bins_array,
    #     model_er_total,
    #     drawstyle='steps-mid',
    #     color='coral'
    # )

    # axes[i,1].plot(
    #     bins_array,
    #     model_er_gamma,
    #     drawstyle='steps-mid',
    #     color='yellow'
    # )
    
    # axes[i,1].plot(
    #     bins_array,
    #     model_er_neutron,
    #     drawstyle='steps-mid',
    #     color='deepskyblue'
    # )
    
    axes[i,1].fill_between(
        bins_array,
        model_er_total,
        color=lighten_color('coral'),
        step='mid',
        edgecolor='orangered',
    )

    axes[i,1].fill_between(
        bins_array,
        model_er_neutron,
        color=lighten_color('deepskyblue'),
        step='mid',
        edgecolor='cornflowerblue'
    )

    axes[i,0].axvspan(0, 0.5, color='grey', alpha=0.5)

    ax = axes[i,0]
    ax.text(0.2, 0.90, '{}'.format(source),
                 fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=1)
    )

    ax = axes[i,1]
    ax.text(0.2, 0.90, '{}'.format(source),
                 fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=1)
    )

axes[0,0].set_ylabel(r'Counts')
axes[1,0].set_ylabel(r'Counts')
axes[1,0].set_xlabel(r'$E_R$ / keV')
axes[1,1].set_xlabel(r'$E_R$ / keV')

# ## GLOBAL FIGURE PARAMETERS
for ax in np.ravel(axes):
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.set_yscale('log')
    ax.set_xscale('log')
#     ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
#     ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
#     ax.set_ylim(5e-4, 2)
    ax.set_xlim(0.5, 49.9)

axes[0, 0].spines['bottom'].set_linewidth(2)
axes[0, 1].spines['bottom'].set_linewidth(2)
axes[0, 0].spines['right'].set_linewidth(2)
axes[1, 0].spines['right'].set_linewidth(2)


leg = axes[0,0].legend(
    handles=[
        plt.Line2D([], [], ls='none', marker='.', color='k'),
        plt.Line2D([], [], color='r', marker='+'),
        plt.fill_between([], [], color='gold'),
        plt.fill_between([], [], color='slateblue'),
        plt.fill_between([], [], color='forestgreen'),
        
    ],
    labels=[
        r'Experimental Bulk Events, ER band only', #,\n$(ER \& \neg NR \& \neg HO)$',
        'Adjusted ER Background from Pulse Simulation',
        'Flat ER component',
        '10.37 keV line component',
        '1.3 keV line component',
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=1
)

leg = axes[0,1].legend(
    handles=[
        plt.fill_between([], [], color='orangered', edgecolor=lighten_color('coral')),
        plt.fill_between([], [], color='cornflowerblue', edgecolor=lighten_color('deepskyblue'))
        
    ],
    labels=[
        'Corrected ER Background',
        'NR band contamination',
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=1
)

# # plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/neutron_contamination.pdf')

corr_path = '/'.join([analysis_dir, 'contamination_dict.npy']) 
np.save(corr_path, contamination_result)