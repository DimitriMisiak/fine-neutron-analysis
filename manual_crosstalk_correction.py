#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""
 
import numpy as np
import pandas as pd
from tqdm import tqdm

from manual_fitting_addon import Manual_fitting

import matplotlib.pyplot as plt
plt.close('all')

analysis_dir = '/home/misiak/Analysis/NEUTRON'

### DATA
fine_data_path = '/'.join([analysis_dir, 'data_quality.h5'])
output_data_path = '/'.join([analysis_dir, 'data_calibrated.h5'])   

stream_list = pd.read_hdf(
    fine_data_path,
    key='df',
    columns=['stream',]
)['stream'].unique()

# initializing the HDFstore (overwriting, be careful !)
pd.DataFrame().to_hdf(
    output_data_path,
    key='df', mode='w', format='table'
)

stream = stream_list[0]
df_quality = pd.read_hdf(
    fine_data_path,
    key='df',
    where='stream = "{}"'.format(stream)
)


ion_energy = df_quality[df_quality.quality_cut][[
    'energy_adu_nodecor_ionA',
    'energy_adu_nodecor_ionB',
    'energy_adu_nodecor_ionC',
    'energy_adu_nodecor_ionD'
]]

# ion_energy = df_quality[df_quality.quality_cut][[
#     'energy_adu_ionA',
#     'energy_adu_ionB',
#     'energy_adu_ionC',
#     'energy_adu_ionD'
# ]]

maxwell_detector = np.loadtxt('maxwell_red80.csv', comments='%', delimiter=',')

def correction(cabling_capacitance):

    maxwell_total = maxwell_detector + np.eye(4) * cabling_capacitance

    crosstalk_correction_matrix = maxwell_total / maxwell_total[0, 0]
    
    corrected_energy = np.dot(crosstalk_correction_matrix, ion_energy.T).T
    
    return corrected_energy

c0 = 50e-12 #F
corrected_energy = correction(c0)

### PLOT
from plot_addon import basic_corner
title='manual crosstalk corretion' 
fig_cross, axes = basic_corner(
    corrected_energy,
    ion_energy.columns,
    num = '{}: Cross-talk Correction'.format(title),
    label='corrected',
    color='slateblue',
    alpha=0.5,
)
# basic_corner(
#     ion_energy.values,
#     ion_energy.columns,
#     axes=axes,
#     color='k',
#     zorder=-1,
#     label='raw'
# )

for ax in fig_cross.get_axes():
    # ax.axvline(0, color='r', zorder=-5)
    # ax.axhline(0, color='r', zorder=-5)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)

# ax = axes[1, 0]
lines = [ax.lines[0] for ax in fig_cross.get_axes()]

xy_list = [[3, 0], [3, 1], [0, 1], [3, 2], [0, 2], [1, 2]]

def correction_manual(x):
    corrected = correction(x[0])
    # return [corrected[:,[3,1]],]
    return [corrected[:, xy] for xy in xy_list]

M = Manual_fitting(lines, correction_manual, [c0,], autoscale=False)


