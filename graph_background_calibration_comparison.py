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

from plot_addon import (
#     LegendTitle,
#     custom_autoscale,
    ax_hist_v2,
#     basic_corner,
#     save_figure_dict
)


# from pipeline_data_calibrated import (
#     quenching,
#     lindhard,
#     energy_recoil,
#     energy_heat_from_er_and_quenching,
#     energy_ion_from_er_and_quenching,
# )


# from pipeline_data_science import (
# #     ionization_baseline_resolution,
# #     std_energy_ion,
#     charge_conservation_threshold
# )

    
plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm
debug = True

analysis_dir = '/home/misiak/Analysis/NEUTRON'
output_dir = '/'.join([analysis_dir, 'analysis_plots'])
extension='pdf'

h5type = 'data'

h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
df_analysis = pd.read_hdf(
    h5_path,
    key='df'
)

df_analysis = df_analysis[(
    df_analysis.quality_cut 
    & df_analysis.bulk_cut
    & df_analysis.charge_conservation_cut
)]

df_back = df_analysis[df_analysis.source == 'Background']
df_calib = df_analysis[df_analysis.source == 'Calibration']

# energy_heat = df_analysis['energy_heat'][quality_cut]
# ion_conservation = df_analysis['energy_nodecor_ion_conservation'][quality_cut]

# x_array = np.linspace(
#     energy_heat.min(),
#     energy_heat.max(),
#     int(1e4)
# )

fig, axes = plt.subplots(
    figsize=(6.3, 6.3),
    nrows=2,
)


ax = axes[0]
ax.plot(
    df_calib.energy_heat,
    df_calib.energy_ion_bulk,
    ls='none', marker='.', color='deepskyblue', alpha=0.5, markersize=2.5,
    label='Calibration',
)
ax.plot(
    df_back.energy_heat,
    df_back.energy_ion_bulk,
    ls='none', marker='.', color='k', alpha=0.5, markersize=2.5,
    label='Background'
)
ax.set_xlabel('$E_{heat}$ / keV$_{ee}$')
ax.set_ylabel('$E_{Ion.}^{bulk}$ / keV')
ax.set_ylim(-2, 15)
ax.set_xlim(0, 15)

ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))


ax = axes[1]
ax.plot(
    df_calib.recoil_energy_bulk,
    df_calib.quenching_bulk,
    ls='none', marker='.', color='deepskyblue', alpha=0.5, markersize=2.5,
    label='Calibration',
)
ax.plot(
    df_back.recoil_energy_bulk,
    df_back.quenching_bulk,
    ls='none', marker='.', color='k', alpha=0.5, markersize=2.5,
    label='Background'
)
ax.set_xlabel('$E_{R}$ / keV')
ax.set_ylabel('$Q$')

ax.set_ylim(-0.25, 1.5)
ax.set_xlim(0, 15)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
                          

for ax in axes:
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))


axes[0].legend(
    handles=[
        plt.Line2D(
            [], [],
            color='deepskyblue', marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color='k', marker='o', ls='none',
        ),
    ],
    labels=[
        'Calibration streams: {} bulk events'.format( df_calib.shape[0] ),
        'Background streams: {} bulk events'.format( df_back.shape[0] ),
        'Charge conservation cut'
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.subplots_adjust(hspace=.0)

# fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/background_calibration_comparison.pdf')
fig.savefig(
    '/home/misiak/Analysis/NEUTRON/thesis_plots/background_calibration_comparison.png',
    dpi=600
)
