#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:12:45 2020

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot_addon import (
    ax_hist,
)

from test_new_science import sigma_function

analysis_dir = '/home/misiak/Analysis/NEUTRON'

### h5 file path
h5_noise_path = '/'.join([
    analysis_dir,
    'noise_science.h5'
])

h5_data_path = '/'.join([
    analysis_dir,
    'data_science.h5'
])

h5_simu_path = '/'.join([
    analysis_dir,
    'simu_science.h5'
])

stream_list = [
    'tg18l005',
    'tg27l000',
    'tg28l000',
    'tg17l007',
    'tg19l010',
    'tg20l000',
    'tg21l000'
]

stream = stream_list[0]

# dfn_analysis = pd.read_hdf(
#     h5_noise_path,
#     key='df',
#     where=(
#         'stream = "{0}"'
#     ).format(stream)
# )

# dfn_quality = dfn_analysis[dfn_analysis.quality_cut]

df_data = pd.read_hdf(
    h5_data_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)


df_quality = df_data[df_data.quality_cut]


fig, ax = plt.subplots()

plt.plot(
    df_quality.energy_heat,
    df_quality.energy_nodecor_ion_conservation,
    ls='none',
    marker='.',
    alpha=0.3
)

sig0 = df_quality.std_energy_nodecor_ion_conservation.unique()[0]
sig10 = df_quality.std_calib_energy_nodecor_ion_conservation.unique()[0]

plt.plot(
    [0,0],
    [-sig0, +sig0],
    color='k',
)

plt.plot(
    [10.37, 10.37],
    [-sig10, +sig10],
    color='k',
) 

x_range = np.linspace(0, 15, 100)
plt.plot(
    x_range,
    sigma_function(x_range, sig0, sig10)
)

# def baseline_resolution(dfn_quality):
#     resolution_dict = dict()
    
#     fig, axes = plt.subplots(
#         nrows=5,
#         sharex=True,
#         figsize=(6.3, 6)
#     )
    
#     bins = 10
    
#     for i, suffix in enumerate(['heat', 'ionA', 'ionB', 'ionC', 'ionD']):
        
#         ax = axes[i]
#         x_data = dfn_quality['energy_{}'.format(suffix)] 
        
#         resolution_dict[suffix] = np.std(x_data)
        
#         bin_edges = np.histogram_bin_edges(x_data, bins=bins)
#         ax_hist(ax,
#                 bin_edges,
#                 x_data,
#                 suffix,
#         )
#         ax.legend(
#             title='$\sigma_{0}={1:.3f}$ keV'.format(suffix, resolution_dict[suffix]),
#             loc='upper left'
#         )

#     return fig, resolution_dict

# fig , resolution_dict = baseline_resolution(dfn_quality)   




# def res_main(df_selection):

#     ax_tuples = ((0, 0), (0, 1))
     
#     quant_dict = dict()
    
#     channel_suffix = [
#         'ion_total',
#         'heat',
#     ]    
    
#     ax_titles =[
#         'Ion Total',
#         'Heat',
#     ]
    
#     num = '{} : Resolution Peak'.format(stream)
#     fig, axes = plt.subplots(
#         ncols=2,
#         figsize=(11.69, 3.27),
#         num=num,
#         squeeze=False,
#     )
    
#     for suffix, tupl, label in zip(channel_suffix, ax_tuples, ax_titles):
        
#         xdata = df_selection['energy_{}'.format(suffix)]

#         if suffix == 'ion_total':
#             blob_cut = (xdata >= 10.37)
#         elif suffix == 'heat':
#             blob_cut = (xdata >= 10.37)
            
#         xdata_alt = xdata[blob_cut]
        
#         sup_alt = np.quantile(xdata_alt, [0.68,])[0]
#         quant_dict[suffix+"_alt"] = sup_alt
        
#         if suffix == 'ion_total':
#             blob_cut = (xdata > 9)
#         elif suffix == 'heat':
#             blob_cut = (xdata > 8)
            
#         xdata = xdata[blob_cut]
        
#         med, inf, sup = np.quantile(xdata, [0.5, 0.16, 0.84])
#         quant_dict[suffix] = (med, inf, sup)


        
#         ax = axes[tupl]
        
#         bin_edges = np.histogram_bin_edges(xdata, bins=100)
    
#         ax_hist(ax, bin_edges, xdata,
#                 'All events', color='coral')
    
    
#         ax.axvline(med, lw=2, ls='--', color='k')
#         ax.axvline(inf, lw=1, ls='--', color='k')
#         ax.axvline(sup, lw=1, ls='--', color='k')

#         ax.axvline(10.37, lw=2, ls='--', color='b')
#         ax.axvline(sup_alt, lw=1, ls='--', color='b')
    

#         ax.legend(
#             title="Peak=${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$".format(med, med-inf, sup-med),
#             loc=2
#         )
#         ax.set_title(label.replace('_', ' '))
        
    
#     fig.text(0.5, 0.98, num,
#               horizontalalignment='center',
#               verticalalignment='center',
#               bbox=dict(facecolor='lime', alpha=0.5))
    
    
#     # resize the plots
#     fig.get_axes()[0].set_xlim(-200, 2000)
#     for i, ax in enumerate(fig.get_axes()):
#             ax.set_xlim(-2, 12)    
     
#     # fig.delaxes(axes[0,0])    
#     fig.tight_layout()

#     return fig, quant_dict

# df_analysis = pd.read_hdf(
#     h5_data_path,
#     key='df',
#     where=(
#         'stream = "{0}"'
#     ).format(stream)
# )

# df_selection = df_analysis[df_analysis.selection_cut]

# fig, quant_dict = res_main(df_selection)