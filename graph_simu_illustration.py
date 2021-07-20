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


from plot_addon import (
    ax_hist_v2,
)

from pipeline_data_heat_calib import (
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
    lindhard,
    quenching,
    energy_recoil,
)

from pipeline_data_science import (
    sigma_function,
)


plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'
output_dir = '/'.join([analysis_dir, 'analysis_plots'])
extension='pdf'

h5type = 'simu'

h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
source = 'Background'

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    # where=(
    #     'source = "{}"'
    # ).format(source)
)

all_cut = (
    df_analysis['quality_cut']
    & df_analysis['charge_conservation_cut']
    & df_analysis['bulk_cut']
)

if h5type == 'simu':
    all_cut = all_cut & df_analysis['trigger_cut']

# neutron_cut = df_analysis['neutron_cut'] & all_cut
# gamma_cut = df_analysis['gamma_cut'] & all_cut
# ho_cut = df_analysis['HO_cut'] & all_cut

# df_all = df_analysis[all_cut]
# df_neutron = df_analysis[neutron_cut]
# df_gamma = df_analysis[gamma_cut]
# df_ho = df_analysis[ho_cut]


simu_list = [
    'flat_ER',
    'flat_NR',
    'line_1keV',
    'line_10keV'
]

color_list = [
    'gold',
    'coral',
    'forestgreen',
    'slateblue',    
]

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(
    figsize=(6.3, 6.3),
    nrows=2,
)

for i in range(len(simu_list)):
    simu_cut = ( df_analysis.simulation == simu_list[i] )
    
    df = df_analysis[all_cut & simu_cut]
    col = color_list[i]

    axes[0].plot(
        df.energy_heat,
        df.energy_ion_bulk,
        ls='none', marker='.', color=col, alpha=0.33, markersize=2.5,
    )

    axes[1].plot(
        df.recoil_energy_bulk,
        df.quenching_bulk,
        ls='none', marker='.', color=col, alpha=0.33, markersize=2.5,
    )



# ### Band illustration
# h5type = 'data'

# h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
# source = 'Calibration'

# df_analysis = pd.read_hdf(
#     h5_path,
#     key='df',
#     where=(
#         'source = "{}"'
#     ).format(source)
# )

# all_cut = (
#     df_analysis['quality_cut']
#     & df_analysis['charge_conservation_cut']
#     & df_analysis['bulk_cut']
# )

# if h5type == 'simu':
#     all_cut = all_cut & df_analysis['trigger_cut']

# neutron_cut = df_analysis['neutron_cut'] & all_cut
# gamma_cut = df_analysis['gamma_cut'] & all_cut
# ho_cut = df_analysis['HO_cut'] & all_cut

# df_all = df_analysis[all_cut]
# df_neutron = df_analysis[neutron_cut]
# df_gamma = df_analysis[gamma_cut]
# df_ho = df_analysis[ho_cut]


# nsigma = 2

# x_range = np.linspace(0, 25, int(1e4))

# sig0 = df_all['std_energy_ion_bulk'].unique().mean()
# sig10 = df_all['std_calib_energy_ion_bulk'].unique().mean()

# V = 2

# er_theory = np.linspace(0, 100, int(1e4))

# qu_neutron = lindhard(er_theory)
# qu_gamma = np.ones(er_theory.shape)
# qu_ho = np.zeros(er_theory.shape)

# color_list = [
#     'grey',
#     'coral',
#     'forestgreen',
#     'slateblue',    
# ]

# quenching_list = [
#     qu_neutron,
#     qu_gamma,
#     qu_ho
# ]


# for i in range(3):
#     col = color_list[i+1]
#     quen = quenching_list[i]
#     V = 2
#     ec_array = energy_heat_from_er_and_quenching(er_theory, quen, V)
#     ei_array = energy_ion_from_er_and_quenching(er_theory, quen)


#     sigma_ion = sigma_function(ec_array, sig0, sig10)
 
#     ei_sup = ei_array + nsigma * sigma_ion
#     ei_inf = ei_array - nsigma * sigma_ion
    
#     er_sup = energy_recoil(ec_array, ei_sup, V)
#     er_inf = energy_recoil(ec_array, ei_inf, V)
    
#     qu_sup = quenching(ec_array, ei_sup, V)
#     qu_inf = quenching(ec_array, ei_inf, V)
    
    
#     ax = axes[0]
#     ax.plot(
#         ec_array,
#         ei_array,
#         color=col,
#         path_effects = cartoon_light,
#     )
#     ax.plot(
#         ec_array,
#         ei_sup,
#         color=col,
#         lw=0.5        
#     )
#     ax.plot(
#         ec_array,
#         ei_inf,
#         color=col,
#         lw=0.5  
#     )

#     ax.fill_between(
#         ec_array,
#         ei_array - nsigma * sigma_ion,
#         ei_array + nsigma * sigma_ion,
#         color=col,
#         alpha=0.1
#     )

#     ax = axes[1]
#     ax.plot(
#         er_theory,
#         quen,
#         color=col,
#         path_effects = cartoon_light,
#     )
#     ax.plot(
#         er_sup,
#         qu_sup,
#         color=col,
#         lw=0.5        
#     )
#     ax.plot(
#         er_inf,
#         qu_inf,
#         color=col,
#         lw=0.5  
#     )

#     ax.fill_between(
#         er_theory,
#         np.interp(er_theory, er_sup, qu_sup),
#         np.interp(er_theory, er_inf, qu_inf),
#         color=col,
#         alpha=0.1
#     )



### ECEI
ax = axes[0]
ax.set_xlabel('$E_{heat}$ / keV$_{ee}$')
ax.set_ylabel('$E_{Ion.}^{bulk}$ / keV')
ax.set_ylim(-2, 15)
ax.set_xlim(0, 15)

ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

### QUENCHING
ax = axes[1]
ax.set_xlabel('$E_{R}$ / keV')
ax.set_ylabel('$Q$')

ax.set_ylim(-0.25, 1.5)
ax.set_xlim(0, 15)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
                          

## GLOBAL FIGURE PARAMETERS
for ax in axes:
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))


leg = axes[0].legend(
    title='RED80, Pulse Simulation \nAll events passing Quality, Charge Conservation and Bulk Cuts',
    handles=[
        plt.Line2D(
            [], [],
            color=color_list[0], marker='o', ls='none',
        ),          
        plt.Line2D(
            [], [],
            color=color_list[1], marker='o', ls='none',
        ),        
        plt.Line2D(
            [], [],
            color=color_list[2], marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color=color_list[3], marker='o', ls='none',
        ),
    ],
    labels = [
        'Flat ER',
        'Flat NR',
        'Line 1.3keV',
        'Line 10.37keV',
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=4
)

plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.subplots_adjust(hspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/pulse_simulation.pdf')
fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/pulse_simulation.png',
            dpi=600)