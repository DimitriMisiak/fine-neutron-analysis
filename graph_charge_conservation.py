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
    ax_hist_v2,
)

from pipeline_data_science import (
    sigma_function
)

    
plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm
debug = True

analysis_dir = '/home/misiak/Analysis/NEUTRON'
output_dir = '/'.join([analysis_dir, 'analysis_plots'])
extension='pdf'



source = 'Background'
h5type = 'data'
stream = 'tg18l005'


h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'source = "{0}"'
        '& stream = "{1}"'
    ).format(source, stream)
)

quality_cut = df_analysis['quality_cut']
charge_cut = df_analysis['charge_conservation_cut']   

energy_heat = df_analysis['energy_heat'][quality_cut]
ion_conservation = df_analysis['energy_nodecor_ion_conservation'][quality_cut]

sig0 = df_analysis.std_energy_nodecor_ion_conservation.unique()[0]
sig10 = df_analysis.std_calib_energy_nodecor_ion_conservation.unique()[0]

x_array = np.linspace(
    energy_heat.min(),
    energy_heat.max(),
    int(1e4)
)

threshold_array = 2 * sigma_function(x_array, sig0, sig10)

### PLOT
fig, ax = plt.subplots(
    figsize=(6.3, 3.9),
)

ax.plot(
    energy_heat[quality_cut & charge_cut],
    ion_conservation[quality_cut & charge_cut],
    ls='none',
    marker='.',
    markersize=2.5,
    alpha=0.3,    
    # marker=',',
    color='slateblue',

    label='Passing events'
)

ax.plot(
    energy_heat[quality_cut & ~charge_cut],
    ion_conservation[quality_cut & ~charge_cut],
    ls='none',
    marker='.',
    markersize=2.5,
    alpha=0.3,
    # marker=',',
    color='coral',
    label='Discarded events'
)

ax.plot(
    x_array,
    threshold_array,
    color='k'
)

ax.plot(
    x_array,
    -threshold_array,
    color='k'
)

ax.grid()
ax.set_xlim(5e-2, 300)
ax.set_ylim(-4, 4)
ax.set_xscale('log')

ax.set_xlabel(r'$E_{heat}$ / keV${}_{ee}$')
ax.set_ylabel(r'$E_{CC}$ / keV')

ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

ax.grid(True, alpha=0.5, which='major')
ax.grid(True, alpha=0.1, which='minor')

ax.legend(
    title='RED80, All background streams\nWith {} total quality events:'.format( sum(quality_cut) ),
    handles=[
        plt.Line2D(
            [], [],
            color='coral', marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color='slateblue', marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color='k',
        ),
    ],
    labels=[
        'Discarded: {} events'.format( sum(quality_cut & ~charge_cut) ),
        'Passing: {} events'.format( sum(quality_cut & charge_cut) ),
        'Charge conservation cut'
    ],
    loc='upper left'
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/charge_conservation.pdf')
fig.savefig(
    '/home/misiak/Analysis/NEUTRON/thesis_plots/charge_conservation.png',
    dpi=600
)

#%%
# # =============================================================================
# # bonus plot
# # =============================================================================
# fig, ax = plt.subplots()

# xdata0 = ion_conservation[quality_cut]
# xdata1 = ion_conservation[quality_cut & charge_cut]

# bin_edges = np.histogram_bin_edges(xdata0, bins=100, range=(-5, 5))

# ax.hist(xdata0, bin_edges, alpha=0.5, color='coral')
# ax.hist(xdata1, bin_edges, alpha=0.5, color='slateblue')

# ax.set_yscale('log')

# # ax_hist_v2(ax, bin_edges, x_data, 'lol')
