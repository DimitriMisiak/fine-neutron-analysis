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
    LegendTitle,
    custom_autoscale,
    ax_hist,
    basic_corner,
    save_figure_dict
)

from pipeline_data_quality import (
    ion_chi2_threshold_function,
    heat_chi2_threshold_function,
    quality_parameters,
)

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1

analysis_dir = '/home/misiak/Analysis/NEUTRON'
h5_path = '/'.join([analysis_dir, 'data_quality.h5'])

stream = 'tg18l005'

df = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

### PLOt
    
quality_cut = df['quality_cut']

offset_ion = abs(df['offset_ionD'])
energy_ion = df['energy_adu_ionD']

# Init figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.3, 3.9))

ax.plot(
    offset_ion,
    energy_ion,
    ls='none',
    marker='.',
    color='k',
    alpha=0.3,
    label='All events'
)

ax.plot(
    offset_ion[quality_cut],
    energy_ion[quality_cut],
    ls='none',
    marker='.',
    color='slateblue',
    alpha=0.3,
    label='Quality events'
)

ax.axvspan(
    quality_parameters["offset_ion_threshold"],
    35000,
    label='Region discarded by the Offset Cut',
    color='r',
    alpha=0.3
)


### plot formatting

ax.set_ylabel('Amplitude Ionization D / ADU')
ax.set_xlabel('Absolute Value of the Offset Ionization D / ADU')

ax.set_ylim(-10, 80)
ax.set_xlim(0, 34000)


ax.xaxis.set_major_locator(mticker.MultipleLocator(5000))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1000))

ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))

# for i in range(ndim):
#     for j in range(len(ind_list)-1):
#         axes[i,j].spines['right'].set_linewidth(2)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

ax.grid(True, alpha=0.5, which='major')
ax.grid(True, alpha=0.1, which='minor')

ax.legend(
    handles=[
        plt.Line2D([], [], ls='none', marker='o', color='k'),
        plt.Line2D([], [], ls='none', marker='o', color='slateblue'),
        plt.axvspan(0, 0, color='r', alpha=0.3),
    ],
    labels=[
        'All events',
        'Quality events',
        'Region discarded by the Offset Cut',
    ],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, 1,),
    frameon=False
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=0.)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/offset_cut.png', dpi=600)