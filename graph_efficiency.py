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

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'

h5_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])   
df_simu = pd.read_hdf(
    h5_simu_path,
    key='df',
)

all_cut = np.ones(shape=df_simu.shape[0], dtype=bool)
trigger_cut = all_cut & df_simu.trigger_cut 
quality_cut = trigger_cut & df_simu.quality_cut
charge_cut = quality_cut & df_simu.charge_conservation_cut
bulk_cut = charge_cut & df_simu.bulk_cut

# gamma_cut = bulk_cut & df_simu.gamma_cut
# neutron_cut = bulk_cut & df_simu.neutron_cut

common_cut_dict = {
    'All': all_cut,
    'Trigger': trigger_cut,
    'Quality': quality_cut,
    'Charge': charge_cut,
    'Bulk': bulk_cut,
}
# 'ER': gamma_cut,
# 'NR': neutron_cut

source_list = ['Background', 'Calibration']
simulation_list =  ['NR', 'ER']


# =============================================================================
# BINNING
# =============================================================================
# dx = 1
# bins = np.arange(0, 50 + dx, dx)
# bins_width = bins[1] - bins[0]
# bins_array = bins[:-1] + (bins_width) / 2
# eff_x_array = bins_array

# bins = np.logspace(np.log(0.2), np.log(50), 100, base=np.exp(1))
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]


std0 = df_simu.std_energy_heat.unique().max()
std10 = df_simu.std_calib_energy_heat.unique().max()
from pipeline_data_science import sigma_function

nsigma = 2

bins_list = [0,]
while ( bins_list[-1] < 50 ):
    last_bin = bins_list[-1]
    dx = sigma_function(last_bin, std0*nsigma, std10*nsigma)
    bins_list.append(last_bin + dx)

print('Bins OK')

bins = np.array(bins_list)
bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]
bins_array = bins[:-1] + (bins_width) / 2

# =============================================================================
# CALCULATION
# =============================================================================

num_dict = dict()
for source in source_list:
    source_cut = (df_simu.source == source)
    
    num_dict[source] = dict()
    for recoil in simulation_list:
        
        if recoil == 'NR':
            recoil_cut = (df_simu.simulation == 'flat_NR')
            band_cut = df_simu.neutron_cut & bulk_cut
        if recoil == 'ER':
            recoil_cut = (df_simu.simulation == 'flat_ER')
            band_cut = df_simu.gamma_cut & bulk_cut
            
        num_dict[source][recoil] = dict()
        for key in common_cut_dict.keys():
            
            local_cut = source_cut & recoil_cut & common_cut_dict[key]
            num_dict[source][recoil][key] = np.histogram(
                df_simu[local_cut].input_energy,
                bins=bins                
            )[0]

        local_cut = source_cut & recoil_cut & band_cut
        num_dict[source][recoil]['Band'] = np.histogram(
            df_simu[local_cut].input_energy,
            bins=bins                
        )[0]

eff_dict = dict()
for source in num_dict.keys():
    
    eff_dict[source] = dict()
    for recoil in num_dict[source].keys():
        
        eff_dict[source][recoil] = dict()
        for cut in num_dict[source][recoil].keys():
            
            eff_dict[source][recoil][cut] = (
                num_dict[source][recoil][cut] / num_dict[source][recoil]['All']
            )

eff_dict['bins'] = bins


color_list = {
    'All': 'lightgrey',
    'Quality': 'yellow',
    'Charge': 'slateblue',
    'Bulk': 'forestgreen',
    'Band': 'deepskyblue',
    'Trigger': 'coral',    
}

#%%
# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(
    figsize=(6.3, 6),
    nrows=2,
    ncols=2,
    sharex='col',
    sharey='row'
)

for im, source in enumerate(source_list):

    ax = axes[im]

    for js, simulation in enumerate(simulation_list):

            a = ax[js]
            
            for key, eff in eff_dict[source][simulation].items():
            
                # line, = a.plot(
                #     bins_array,
                #     eff,
                #     drawstyle='steps-mid',
                #     # marker='.',
                #     # markersize=2.5,
                #     label=key,
                #     color=color_list[key]
                # )
                a.fill_between(
                    bins_array,
                    eff,
                    color=lighten_color(color_list[key]),
                    edgecolor=color_list[key],
                    step='mid',
                    label=key,
                )

                # a.fill_between(
                #     bins_array,
                #     eff,
                #     st
                
            a.grid()
            a.set_yscale('log')


            a.text(0.2, 0.94, '{} {}'.format(source, simulation),
                         fontsize=10,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=a.transAxes,
                         bbox=dict(facecolor='white', alpha=1)
            )


# axes[0,0].set_ylabel(r'$f^{eff.}$')
# axes[1,0].set_ylabel(r'$f^{eff.}$')
axes[0,0].set_ylabel(r'$\mathcal{F}$')
axes[1,0].set_ylabel(r'$\mathcal{F}$')
axes[1,0].set_xlabel(r'$E_R^{input}$ / keV')
axes[1,1].set_xlabel(r'$E_R^{input}$ / keV')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

## GLOBAL FIGURE PARAMETERS
for ax in np.ravel(axes):
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_xlim(0, 49.9)

axes[0,0].set_ylim(2e-2, 2)
axes[1,0].set_ylim(2e-3, 3)

axes[0, 0].spines['bottom'].set_linewidth(2)
axes[0, 1].spines['bottom'].set_linewidth(2)
axes[0, 0].spines['right'].set_linewidth(2)
axes[1, 0].spines['right'].set_linewidth(2)


leg = axes[0,0].legend(
    # handles=[
    #     plt.Line2D(
    #         [], [],
    #         color=color_list[3], lw=5, alpha=0.5
    #     ),        
    # ],
    labels=[
        'No Cut',
        'Trigger and \nLive Time Cuts',
        'Quality Cut',
        'Charge Conservation Cut',
        'Bulk Cut',
        'ER Band Cut for Simulated ER Events\nNR Band Cut for Simulated NR Events',
    ],
    loc='lower center',
    bbox_to_anchor=(0.9, 1),
    frameon=False,
    ncol=3
)

# plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/cut_efficiency.pdf')

eff_path = '/'.join([analysis_dir, 'efficiency_dict.npy']) 
np.save(eff_path, eff_dict)