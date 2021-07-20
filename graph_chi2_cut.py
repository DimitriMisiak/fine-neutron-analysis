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


from pipeline_data_quality import (
    quality_parameters,
    ion_chi2_threshold_function,
    heat_chi2_threshold_function,
)

    
plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm
debug = True

analysis_dir = '/home/misiak/Analysis/neutron_background'
output_dir = '/'.join([analysis_dir, 'analysis_plots'])
extension='pdf'



source = 'Background'
h5type = 'data'

stream = 'tg18l005'

h5_path = '/'.join([analysis_dir, '{}_quality.h5'.format(h5type)])   
 
df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'source = "{0}"'
        '& stream = "{1}"'
    ).format(source, stream)
)

df = df_analysis

fig, axes = plt.subplots(
    ncols=2,
    figsize=(6.3, 3.9),
    sharey=True,
)

channel_suffix = [
    'heat',
    'ionD',
]

ax_titles =[
    'Heat',
    'Ion D',
]

quality_cut = df['quality_cut']


for suffix, ax, title in zip(channel_suffix, axes, ax_titles):
    
    xdata = abs( df['energy_adu_{}'.format(suffix)] )
    ydata = df['chi2_{}'.format(suffix)]
    
    nsamples = xdata.shape[0]
    
    ax.plot(xdata, ydata,
            label='All events: {}'.format(nsamples),
            c='red', marker='.', markersize=0.1, ls='none')
    
    xdata_cut = xdata[quality_cut]
    ydata_cut = ydata[quality_cut]
    
    if nsamples < 1000:
        marker = '.'
    else:
        marker = '.'

    ax.plot(xdata_cut, ydata_cut,
            label='Quality events: {}'.format(xdata_cut.shape[0]),
            c='slateblue', marker=marker, markersize=0.1, ls='none')

    ax.set_xlim(xdata_cut.min()*0.5, ax.get_xlim()[1])
    ax.set_ylim(ydata_cut.min()*0.5, ax.get_ylim()[1])

    
fig.tight_layout(rect=(0, 0, 1, 0.98))


# plotting the limit
x_data = 10**np.linspace(-2, 5, int(1e4))
cut_ion = ion_chi2_threshold_function(
    quality_parameters['ion_chi2_threshold'],
    x_data
)
cut_heat = heat_chi2_threshold_function(
    quality_parameters[stream]['heat_chi2_threshold'],
    x_data)    
for i, ax in enumerate(fig.get_axes()):
    if i == 0:
        ax.plot(x_data, cut_heat, lw=1, color='k', label='quality cut')
    else:
        ax.plot(x_data, cut_ion, lw=1, color='k', label='quality cut')


# ax.grid()
# ax.set_xlim(5e-2, 300)
# ax.set_ylim(-4, 4)
# ax.set_xscale('log')

# ax.set_xlabel(r'$E_{heat}$ / keV')
# ax.set_ylabel(r'$E_{CC}$ / keV')

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

axes[0].set_xlim(5e0, 5e4)
axes[1].set_xlim(1e-2, 5e4)

axes[0].set_ylim(10**1, 10**9)

axes[0].set_xlabel('Amplitude Heat / ADU')
axes[1].set_xlabel('Amplitude Ion. D / ADU')
axes[0].set_ylabel('$\chi^2$')

for ax in axes:
    ax.set_yscale('log')
    ax.set_xscale('log')    
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')

axes[1].legend(
    title='RED80, Stream {0}\nWith {1} total events:'.format(stream, len(quality_cut) ),
    handles=[
        plt.Line2D(
            [], [],
            color='r', marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color='b', marker='o', ls='none',
        ),
        plt.Line2D(
            [], [],
            color='k',
        ),
    ],
    labels=[
        'Discarded: {} events'.format( sum(~quality_cut) ),
        'Passing: {} events'.format( sum(quality_cut) ),
        '$\chi^2$ Cut on Heat and Ion. D channels'
    ],
    loc='lower center',
    bbox_to_anchor=(0,1),
    frameon=False,
    ncol=2
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(wspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/chi2_cut.pdf')
fig.savefig(
    '/home/misiak/Analysis/NEUTRON/thesis_plots/chi2_cut.png',
    dpi=600
)

