#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:04:43 2020

@author: misiak
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

from tqdm import tqdm

plt.rcParams['text.usetex']=True

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
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


def sigma_function(energy_heat, sigma_baseline, sigma_calib):
    
    sigma_local = (
        sigma_baseline
        + (sigma_calib - sigma_baseline) * energy_heat / 10.37
    )
    
    return sigma_local

nsigma = 2

energy_array = np.linspace(0, 100, 10000)

sigma_dict = dict()
for chan in 'ABCD':
    # sig0 = df_quality['std_energy_ion{}'.format(chan)].unique()[0]
    # sig10 = df_quality['std_calib_energy_ion{}'.format(chan)].unique()[0]

    sig0 = df_quality['std_energy_ion{}'.format(chan)].unique().max()
    sig10 = df_quality['std_calib_energy_ion{}'.format(chan)].unique().max()

    sigma_dict[chan] = sigma_function(
        energy_array,
        sig0,
        sig10
    )

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(
    nrows=2,
    figsize=(6.3, 4),
    sharex=True,
)

axes[0].plot(
    df_quality.energy_heat,
    df_quality.energy_ionA,
    ls='none',
    marker='.',
    alpha=0.3,
    color='coral',
)

axes[0].plot(
    df_bulk.energy_heat,
    df_bulk.energy_ionA,
    ls='none',
    marker='.',
    alpha=0.3,
    color='slateblue',
)

axes[0].plot(
    energy_array,
    nsigma * sigma_dict['A'],
    color='k',
    lw=2
)
axes[0].plot(
    energy_array,
    -nsigma * sigma_dict['A'],
    color='k',
    lw=2
)


axes[1].plot(
    df_quality.energy_heat,
    df_quality.energy_ionC,
    ls='none',
    marker='.',
    alpha=0.3,
    color='coral',
)

axes[1].plot(
    df_bulk.energy_heat,
    df_bulk.energy_ionC,
    ls='none',
    marker='.',
    alpha=0.3,
    color='slateblue',
)

axes[1].plot(
    energy_array,
    nsigma * sigma_dict['C'],
    color='k',
    lw=2
)
axes[1].plot(
    energy_array,
    -nsigma * sigma_dict['C'],
    color='k',
    lw=2
)
# =============================================================================
# plot formatting
# =============================================================================

axes[0].set_ylabel('$E_A$ / keV')
axes[1].set_ylabel('$E_C$ / keV')
axes[1].set_xlabel('$E_{heat}$ / keV${}_{ee}$')

axes[0].set_ylim(-5, 15)
axes[1].set_ylim(-5, 15)
axes[1].set_xlim(1e-1, 1e2)

for ax in axes:

    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2.5))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')

    ax.set_xscale('log')

# for i in range(ndim):
#     for j in range(len(ind_list)-1):
#         axes[i,j].spines['right'].set_linewidth(2)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

axes[0].legend(
    title=(
        "RED80, Background streams\n"
        "Quality events with charge conservation"
    ),
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
        'Discarded: {} events'.format( df_quality.shape[0] -  df_bulk.shape[0]),
        'Passing: {} events'.format( df_bulk.shape[0] ),
        'Bulk Cut'
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=3
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0)

# s0 = (
#       '$y = \\frac{{A_A^0}}{{A_B^0}} x + A_B^0$ \n'
#       '$A_A^0 = {0:.2f}$ ADU\n'
#       '$A_A^0 = {1:.2f}$ ADU\n'
# ).format(peakA, peakB)
# axes[0].text(x=-75, y=-75, s=s0, bbox=dict(facecolor='white', alpha=0.5))

# s1 = (
#       '$y = \\frac{{A_C^0}}{{A_D^0}} x + A_D^0$ \n'
#       '$A_C^0 = {0:.2f}$ ADU\n'
#       '$A_D^0 = {1:.2f}$ ADU\n'
# ).format(peakC, peakD)
# axes[1].text(x=35, y=47, s=s1, bbox=dict(facecolor='white', alpha=0.5))

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/bulk_cut.png', dpi=600)