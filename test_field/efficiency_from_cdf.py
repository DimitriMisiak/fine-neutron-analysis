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

# num_dict = dict()
# for source in ['Background', 'Calibration']:
#     source_cut = (df_simu.source == source)
    
#     num_dict[source] = dict()
#     for recoil in ['NR', 'ER']:
        
#         if recoil == 'NR':
#             band_cut = df_simu.neutron_cut
#         if recoil == 'ER':
#             band_cut = df_simu.gamma_cut
        
#         num_dict[source][recoil] = dict()
#         for key in common_cut_dict.keys():
            
#             num_dict[source][recoil][key] = sum(
#                 source_cut & band_cut & common_cut_dict[key]
#             )

# eff_dict = dict()
# for source in num_dict.keys():
    
#     eff_dict[source] = dict()
#     for recoil in num_dict[source].keys():
        
#         eff_dict[source][recoil] = dict()
#         for cut in num_dict[source][recoil].keys():
            
#             eff_dict[source][recoil][cut] = (
#                 num_dict[source][recoil][cut] / num_dict[source][recoil]['All']
#             )
    

# =============================================================================
# PLOT
# =============================================================================
# fig, axes = plt.subplots(
#     figsize=(6.3, 6.3),
#     nrows=2,
#     ncols=2,
#     sharex='col',
#     sharey='row'
# )

#%%
plt.close('all')
fig, axes = plt.subplots(nrows=2, ncols=2)

bins = np.arange(0, 51, 1)
bins_width = bins[1] - bins[0]
bins_array = bins[:-1] + (bins_width) / 2
eff_x_array = bins_array


cut_0 = (
    (df_simu.source == 'Background')
    & (df_simu.simulation == 'flat_NR')
    & common_cut_dict['All']
)
data_0 = df_simu[cut_0].input_energy
xdata_0, cdf_0 = cdf_calc(data_0)


axes[0,0].plot(
    xdata_0,
    # cdf_0,
    np.gradient(cdf_0, xdata_0),
    drawstyle='steps-mid',
    label='All',
)

n0 = axes[1,0].hist(
    data_0,
    bins=bins,
    label='All'
)[0]

cut_1 = (
    (df_simu.source == 'Background')
    & (df_simu.simulation == 'flat_NR')
    & common_cut_dict['Bulk']
)
data_1 = df_simu[cut_1].input_energy
xdata_1, cdf_1 = cdf_calc(data_1)

axes[0,0].plot(
    xdata_1,
    # cdf_1,
    np.gradient(cdf_1, xdata_1),
    drawstyle='steps-mid',
    label='Bulk',
)

n1 = axes[1,0].hist(
    data_1,
    bins=bins,
    label='Bulk'
)[0]



axes[0, 1].plot(
    xdata_1,
    np.gradient(cdf_1, xdata_1) / np.gradient(np.interp(xdata_1, xdata_0, cdf_0), xdata_1) * data_1.size / data_0.size,
    drawstyle='steps-mid',
    label='cdf'
)

axes[0, 1].plot(
    bins_array,
    n1/n0,
    drawstyle='steps-mid',
    label='hist'
)


for ax in np.ravel(axes):     
    ax.legend()
    
fig.tight_layout()
#%%
       
# # ### PLOT number of events passing the cuts
# # fig, axes = plt.subplots(nrows=2, ncols=2, num='Histogramm events passing cuts',
# #                           figsize=(10, 7), sharex='col', sharey='row')

# # for im, source in enumerate(source_list):

# #     ax = axes[im]

# #     for js, simulation in enumerate(simulation_list):

# #             a = ax[js]
            
# #             for key, num in num_dict[source][simulation].items():
            
# #                 line, = a.plot(
# #                     bins_array,
# #                     num,
# #                     drawstyle='steps-mid',
# #                     marker='.',
# #                     label=key
# #                 )
# #                 a.fill_between(
# #                     bins_array,
# #                     num,
# #                     color=lighten_color(line.get_color()),
# #                     step='mid',
# #                 )
                
# #             msg = '{} {} Events'.format(source, simulation).replace('_', ' ')
# #             a.text(
# #                 0.5, 0.1,
# #                 msg,
# #                 horizontalalignment='center',
# #                 verticalalignment='center',
# #                 transform=a.transAxes
# #             )
# #             a.grid()
# #             a.set_yscale('log')

# # a.legend(
# #     loc='center left',
# #     bbox_to_anchor=(1.05, 1.05)
# # )
# # axes[0,0].set_ylabel('Counts')
# # axes[1,0].set_ylabel('Counts')
# # axes[1,0].set_xlabel('Recoil Energy [keV]')
# # axes[1,1].set_xlabel('Recoil Energy [keV]')

# fig.tight_layout()
# fig.subplots_adjust(hspace=0, wspace=0)


# ### PLOT number of events passing the cuts
# fig, axes = plt.subplots(nrows=2, ncols=2, num='Histogramm cut efficiency',
#                           figsize=(10, 7), sharex='col', sharey='row')

# for im, source in enumerate(source_list):

#     ax = axes[im]

#     for js, simulation in enumerate(simulation_list):

#             a = ax[js]
            
#             for key, eff in eff_dict[source][simulation].items():
            
#                 line, = a.plot(
#                     bins_array,
#                     eff,
#                     drawstyle='steps-mid',
#                     marker='.',
#                     label=key
#                 )
#                 a.fill_between(
#                     bins_array,
#                     eff,
#                     color=lighten_color(line.get_color()),
#                     step='mid',
#                 )
                
#             msg = '{} {} Efficiency'.format(source, simulation).replace('_', ' ')
#             a.text(
#                 0.5, 0.1,
#                 msg,
#                 horizontalalignment='center',
#                 verticalalignment='center',
#                 transform=a.transAxes
#             )
#             a.grid()
#             a.set_yscale('log')






















axes[0,0].set_ylabel('f^{eff.}')
axes[1,0].set_ylabel('f^{eff}')
axes[1,0].set_xlabel('$E_R^{input}$ / keV')
axes[1,1].set_xlabel('$E_R^{input}$ / keV')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


# ### ECEI
# ax = axes[0]
# ax.set_xlabel('$E_{heat}$ / keV$_{ee}$')
# ax.set_ylabel('$E_{Ion.}^{bulk}$ / keV')
# ax.set_ylim(-2, 15)
# ax.set_xlim(0, 15)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

# ### QUENCHING
# ax = axes[1]
# ax.set_xlabel('$E_{R}$ / keV')
# ax.set_ylabel('$Q$')

# ax.set_ylim(-0.25, 1.5)
# ax.set_xlim(0, 15)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
                          

## GLOBAL FIGURE PARAMETERS
for ax in axes:
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2.5))


# leg = axes[0].legend(
#     title='RED80, Calibration streams\nAll events passing Quality, Charge Conservation and Bulk Cuts',
#     handles=[
#         plt.Line2D(
#             [], [],
#             color=color_list[2], marker='o', ls='none',
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[1], marker='o', ls='none',
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[3], marker='o', ls='none',
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[0], marker='o', ls='none',
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[2], path_effects=cartoon_light
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[1], path_effects=cartoon_light
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[3], path_effects=cartoon_light
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[2], lw=5, alpha=0.5
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[1], lw=5, alpha=0.5
#         ),
#         plt.Line2D(
#             [], [],
#             color=color_list[3], lw=5, alpha=0.5
#         ),        
#     ],
#     labels=[
#         'Passing ER Cut',
#         'Passing NR Cut',
#         'Passing HO Cut',
#         'Discarded events',
#         'Theory ER line',
#         'Theory NR line',
#         'Theory NR line',
#         'ER Band Cut',
#         'NR Band Cut',
#         'HO Band Cut',      

#     ],
#     loc='lower center',
#     bbox_to_anchor=(0.5, 1),
#     frameon=False,
#     ncol=3
# )

# plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.subplots_adjust(hspace=.0)

# fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/cut_histogram.pdf')
