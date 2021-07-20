#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:35:22 2020

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1

from plot_addon import (
    LegendTitle,
    custom_autoscale,
    ax_hist,
    basic_corner,
    save_figure_dict
)

stream = 'tg18l005'




# =============================================================================
# noise selection: baseline resolution 
# =============================================================================
h5_noise_path = '/home/misiak/Analysis/NEUTRON/noise_heat_calib.h5'.format(stream)

dfn_analysis = pd.read_hdf(
    h5_noise_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)
dfn_quality = dfn_analysis[dfn_analysis.quality_cut]
 
resolution_dict = dict()
for i, suffix in enumerate(['heat', 'ionA', 'ionB', 'ionC', 'ionD']):
    x_data = dfn_quality['energy_{}'.format(suffix)] 
    resolution_dict[suffix] = np.std(x_data)

# =============================================================================
# 10kev selection: ion energy
# =============================================================================
h5_path = '/home/misiak/Analysis/NEUTRON/data_heat_calib.h5'.format(stream)

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

df_quality = df_analysis[df_analysis.quality_cut]
df_selection = df_analysis[df_analysis.selection_cut]

# characteristic cuts

incomplete_heat = ( df_selection.energy_heat < 10 )
incomplete_ion = ( df_selection.energy_ion_total < 8 )
trapped_cut = ( incomplete_heat | incomplete_ion )
df_trapped = df_selection[incomplete_heat | incomplete_ion]

complete_cut = ~trapped_cut

ionA_sigma5 = resolution_dict['ionA']*5
ionC_sigma5 = resolution_dict['ionC']*5

bulk_cut = (
    complete_cut
    & (df_selection.energy_ionA < ionA_sigma5)
    & (df_selection.energy_ionC < ionC_sigma5)
)

df_bulk = df_selection[bulk_cut]
# df_selection = df_analysis[df_analysis.quality_cut]

#%%
# =============================================================================
# calculation
# =============================================================================

sign_vector = np.sign([df_bulk['polar_{}'.format(suffix)].unique()[0] for suffix in 'ABCD'])

recoil_energy = 10.37e3 # eV
eh_pair_energy = 3 # ev
elementary_charge = 1.602e-19 # C

# electric charge perturbation vector for 10.37keV bulk, in C
q_vector = (
    np.array([0, 1, 0, 1])
    * (- sign_vector)
    * elementary_charge * (recoil_energy / eh_pair_energy)
)

# electric potential perturbation vector for 10.37keV bulk, in V
# adu_vector = np.array([df_bulk['energy_adu_ion{}'.format(suffix)].mean() for suffix in 'ABCD'])
adu_vector = np.array([df_bulk['energy_adu_ion{}'.format(suffix)] for suffix in 'ABCD'])
gain_adu = 67.4e-9 # V/adu
v_vector = adu_vector * gain_adu

# maxwell matrix, in F
maxwell_matrix = np.loadtxt('maxwell_red80.csv', comments='%', delimiter=',')

total_matrix = maxwell_matrix + np.eye(maxwell_matrix.shape[0]) * 150e-12

# calculation
A = np.dot(maxwell_matrix, v_vector)

B = np.add(-A.T, q_vector).T

C = B / v_vector

#%%
fig, axes = plt.subplots(
    nrows=2,
    sharex=True,
    figsize=(6.3,3.9)
)

labels='ABCD'

for ax_ind,i in enumerate([1,3]):
    # ax = np.ravel(axes)[i]
    ax = axes[ax_ind]

    x_data = C[i] * 1e12 #nF

    med = np.median(x_data)
    std = np.std(x_data)
    
    range_hist = (med-5*std, med+5*std)
    
    bins_edges = np.histogram_bin_edges(
        x_data,
        bins=50,
        range=range_hist
    )

    ax.hist(x_data, bins=bins_edges, color='deepskyblue')
    ax_cdf = ax_hist(ax, bins_edges, x_data, lab=labels[i])[0]
 
    term_label ='$C_{{cabling, {0}{0} }}$'.format(i+1)
    
    leg = ax_cdf.legend(
        labels=('Histogram', 'CDF'),
        handles=(ax.patches[0], ax_cdf.lines[0]),
        title=term_label+'\nMedian={:.1f} nF \nStd.={:.1f} nF'.format(med,std),
        loc='upper left',
    )
    
    plt.setp(leg.get_title(), multialignment='center')
    
    ax_cdf.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_cdf.grid(True, alpha=0.5, which='major')
    
for ax in axes:
    ax.set_ylabel('Event Count', color='blue')
    ax.tick_params(axis='y', labelcolor='royalblue')
    ax.grid(False)
    ax.grid(True, alpha=0.5, which='major', axis='x')
    ax.grid(True, alpha=0.1, which='minor', axis='x')


plt.setp(axes[0].get_xticklabels(), visible=False)
axes[0].spines['bottom'].set_linewidth(2)
axes[1].yaxis.offsetText.set_visible(False)

# only for bottom most axis
ax.set_xlabel(r'Capacitance / nF')
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(2.5))

### Grid



### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0)


# for suffix in 'ABCD':
#     sig = df_bulk['energy_adu_ion{}'.format(suffix)].mean()
#     print("signal")
#     print(sig)
#     print("capacitance")
#     print( 3333 * 1.6e-19 / (sig * 67.4 * 1e-9 ) )

