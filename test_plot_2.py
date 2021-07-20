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
    custom_autoscale
)

from pipeline_data_science import(
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

h5type = 'data'

h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'source = "Background"'
    )
)

df_quality = df_analysis[df_analysis.quality_cut & df_analysis.charge_conservation_cut]

df_bulk = df_quality[df_quality.bulk_cut]

    
# initializing pseudo-corner plot
ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
ax_discard = [(0, 1), (1, 2), (0, 2)]

# chan_x = np.insert(run_tree.chan_veto, 0, run_tree.chan_collect[1])
# chan_y = np.append(run_tree.chan_veto, run_tree.chan_collect[0])    
chan_x = ['D', 'A', 'C']
chan_y = ['A', 'C', 'B']

sig_a = df_quality['std_energy_ionA'].unique().mean()
sig_c = df_quality['std_energy_ionC'].unique().mean()

nsigma = 2 

thresh_a = nsigma * sig_a
thresh_c = nsigma * sig_c


fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(6.3, 6.3),
    sharex='col', sharey='row'
)

# actually plotting the data
for atupl in ax_tuples:
    
    ax = axes[atupl]
    xind = chan_x[atupl[1]]
    yind = chan_y[atupl[0]]

    # energy_x = energy[:, xind]
    # energy_y = energy[:, yind]
    energy_x = df_quality['energy_ion{}'.format(xind)]
    energy_y = df_quality['energy_ion{}'.format(yind)]

    energy_x_trapped = df_bulk['energy_ion{}'.format(xind)]
    energy_y_trapped = df_bulk['energy_ion{}'.format(yind)]

    ax.plot(
            energy_x, energy_y,
            ls='none',
            # marker=',',
            marker='o', markersize=1,
            alpha=0.1, zorder=10, color='slateblue',
            label='Selected Events'
    )
        
    
    custom_autoscale(ax, energy_x, energy_y)
    
    ax.grid(alpha=0.3)
    
    # # resolution vizualition
    # try:
    #     res_y = resolution_dict[yind] * 10
    #     ax.axhspan(-res_y/2, res_y/2, alpha=0.3)
        
    #     res_x = resolution_dict[xind] * 10 
    #     ax.axvspan(-res_x/2, res_x/2, alpha=0.3)
    # except:
    #     pass
    
    
    
    
    
    
    if atupl[0] == 2:
        ax.set_xlabel(
                r'$E_{{Ion.\ {}}}$ / keV'.format(xind)
        )
            
    if atupl[1] == 0:
        ax.set_ylabel(
                r'$E_{{Ion.\ {}}}$ / keV'.format(yind)
        )


for tupl in ax_discard:
    fig.delaxes(axes[tupl])
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=.0)

axes = fig.get_axes()
for ax in axes:
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))  

    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))  

    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')


leg = axes[0].legend(
    title="RED80, Background events\npassing quality and charge conservation cuts",
    handles=[
        plt.Line2D(
            [], [],
            color='slateblue', marker='o', ls='none',
        ),
    ],
    labels=[
        '{} events'.format( df_quality.shape[0] -  df_bulk.shape[0]),
        'Passing: {} events'.format( df_bulk.shape[0] ),
        'Bulk Cut'
    ],
    loc='lower left',
    bbox_to_anchor=(1.05, 0.05),
)

plt.setp(leg.get_title(), multialignment='center')

# ### Figure adjustments
# fig.align_ylabels(fig.axes)    
# fig.tight_layout()
# # fig.subplots_adjust(hspace=.0)

# fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/bulk_cut.pdf')
fig.savefig(
    '/home/misiak/Bureau/corner_plot.png',
    dpi=600
)