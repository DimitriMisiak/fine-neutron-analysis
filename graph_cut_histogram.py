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


plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'

h5_path = '/'.join([analysis_dir, 'data_science.h5'])   
df_data = pd.read_hdf(
    h5_path,
    key='df',
)

all_cut = np.ones(shape=df_data.shape[0], dtype=bool)
quality_cut = all_cut & df_data.quality_cut
charge_cut = quality_cut & df_data.charge_conservation_cut
bulk_cut = charge_cut & df_data.bulk_cut

gamma_cut = bulk_cut & df_data.gamma_cut
neutron_cut = bulk_cut & df_data.neutron_cut

cut_list = [
    all_cut,
    quality_cut,
    charge_cut,
    bulk_cut,
    neutron_cut,
    gamma_cut,   
]

color_list = [
    'lightgrey',
    'yellow',
    'slateblue',
    'forestgreen',
    'deepskyblue',
    'coral',    
]

label_list = [
    'All events',
    'Events passing Quality cuts',
    'Events passing Charge Conservation Cut',
    'Events passing Bulk Cut',
    'Events passing NR Band Cut',   
    'Events passing ER Band Cut',
]


bins = np.arange(0, 51, 1)
bins_width = bins[1] - bins[0]
bins_array = bins[:-1] + (bins_width) / 2
eff_x_array = bins_array

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(
    figsize=(6.3, 6.3),
    nrows=2,
    sharex=True,
)

for im, mode in enumerate(['Background', 'Calibration']):

    source_cut = ( df_data.source == mode )

    ax = axes[im]

    for i, local_cut in enumerate(cut_list):

        df = df_data[source_cut & local_cut]
        
        color = color_list[i]
        edgecolor = 'k'
        lw = 0.5
        if color == 'coral':
            edgecolor='orangered'
            lw = 2
        if color == 'deepskyblue':
            edgecolor='cornflowerblue'
            lw = 2
            
        ax.hist(
            df.recoil_energy_bulk,
            bins=bins,
            color=color,
            histtype='stepfilled',
            edgecolor=edgecolor,
            linewidth=lw,
            label=label_list[i]
        )
        
        
    n_gamma = np.histogram(
        df_data[source_cut & gamma_cut].recoil_energy_bulk,
        bins=bins
    )[0]
                    
    ax.plot(
        bins_array,
        n_gamma,
        drawstyle = 'steps-mid',
        color='orangered',
        lw=2
    )

    n_neutron = np.histogram(
        df_data[source_cut & neutron_cut].recoil_energy_bulk,
        bins=bins
    )[0]
            
    ax.plot(
        bins_array,
        n_neutron,
        drawstyle = 'steps-mid',
        color='cornflowerblue',
        lw=2
    )
 
## GLOBAL FIGURE PARAMETERS
for ax in axes:
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    ax.set_ylabel('Counts')
    ax.set_xlim(0, 50)
    
axes[0].spines['bottom'].set_linewidth(2)
axes[1].set_xlabel('$E_R$ / keV')

axes[0].text(0.8, 0.9, 'Background streams',
             fontsize=12,
             horizontalalignment='center',
             verticalalignment='center',
             transform=axes[0].transAxes,
             bbox=dict(facecolor='white', alpha=1)
)

axes[1].text(0.8, 0.9, 'Calibration streams',
             fontsize=12,
             horizontalalignment='center',
             verticalalignment='center',
             transform=axes[1].transAxes,
             bbox=dict(facecolor='white', alpha=1)
)

leg = axes[0].legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=2
)

plt.setp(leg.get_title(), multialignment='center')



### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/cut_histogram.pdf')
