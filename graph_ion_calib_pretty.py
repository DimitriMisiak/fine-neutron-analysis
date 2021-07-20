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

from scipy.odr import ODR, Model, Data, RealData
from tqdm import tqdm

plt.rcParams['text.usetex']=True

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1

h5_path = '/home/misiak/Analysis/NEUTRON/data_xtalk.h5'

stream = 'tg18l005'

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

df_quality = df_analysis[df_analysis.quality_cut]

fig, axes = plt.subplots(
    figsize=(6.3,3.5),
    ncols=2
)

heat = df_quality.energy_adu_heat
ion_a = df_quality.energy_adu_corr_ionA
ion_b = df_quality.energy_adu_corr_ionB
ion_c = df_quality.energy_adu_corr_ionC
ion_d = df_quality.energy_adu_corr_ionD

axes[0].plot(
    ion_b,
    ion_a,
    ls='none',
    marker='.',
    color='coral',
    markersize=1,
)

axes[1].plot(
    ion_d,
    ion_c,
    ls='none',
    marker='.',
    color='coral',
    markersize=1,
)

ion_tot = ion_a + ion_b - ion_c - ion_d
ion_cons = ion_a + ion_b + ion_c + ion_d

calib_selection = (
    (df_quality.energy_adu_heat > 500)
    & (abs(ion_tot) > 100)
    & (abs(ion_tot) < 150)
)

df_calib = df_quality[calib_selection]

heat = df_calib.energy_adu_heat
ion_a = df_calib.energy_adu_corr_ionA
ion_b = df_calib.energy_adu_corr_ionB
ion_c = df_calib.energy_adu_corr_ionC
ion_d = df_calib.energy_adu_corr_ionD

axes[0].plot(
    ion_b,
    ion_a,
    ls='none',
    marker='.',
    color='slateblue',
    markersize=1,
)

axes[1].plot(
    ion_d,
    ion_c,
    ls='none',
    marker='.',
    markersize=1,
    color='slateblue',
)

def func(beta, x):
    y = (-beta[1]/beta[0])*x+beta[1]
    return y

### Calibration A & B
data = RealData(ion_b, ion_a, 5, 5)
model = Model(func)
   
odr = ODR(data, model, [1,1])

odr.set_job(fit_type=0)
output = odr.run()

peakB, peakA = output.beta

### Calibration C,D
data = RealData(ion_d, ion_c, 5, 5)
model = Model(func)
   
odr = ODR(data, model, [1,1])

odr.set_job(fit_type=0)
output = odr.run()

peakD, peakC = output.beta

axes[0].plot(
        [peakB, 0],
        [0, peakA],
        ls='-',
        marker='o',
        color='k'
)

axes[1].plot(
        [peakD, 0],
        [0, peakC],
        ls='-',
        marker='o',
        color='k'
)




# =============================================================================
# plot formatting
# =============================================================================

axes[0].set_ylabel('Amplitude Ionization A / ADU')
axes[0].set_xlabel('Amplitude Ionization B / ADU')
axes[1].set_ylabel('Amplitude Ionization C / ADU')
axes[1].set_xlabel('Amplitude Ionization D / ADU')

axes[0].set_xlim(-80, 20)
axes[0].set_ylim(-80, 20)
axes[1].set_xlim(-20, 80)
axes[1].set_ylim(-20, 80)

for ax in axes:

    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')


# for i in range(ndim):
#     for j in range(len(ind_list)-1):
#         axes[i,j].spines['right'].set_linewidth(2)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout(rect=(0, 0, 1, 0.88))
# fig.subplots_adjust(hspace=.0, wspace=0.)

axes[0].legend(
    title=(
        'Ionization Channel Calibration, RED80'
    ),
    handles=[
        plt.Line2D([], [], ls='none', marker='o', color='coral'),
        plt.Line2D([], [], ls='none', marker='o', color='slateblue'),
        plt.Line2D([], [], ls='-', marker='o', color='k'),
    ],
    labels=[
        'Quality events',
        '10.37 keV Calibration events\nwith complete charge collection',
        'Adjusted Linear law',
    ],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(1, 1),
    frameon=False
)

s0 = (
      '$y = \\frac{{A_A^0}}{{A_B^0}} x + A_B^0$ \n'
      '$A_A^0 = {0:.2f}$ ADU\n'
      '$A_A^0 = {1:.2f}$ ADU\n'
).format(peakA, peakB)
axes[0].text(x=-75, y=-75, s=s0, bbox=dict(facecolor='white', alpha=0.5))

s1 = (
      '$y = \\frac{{A_C^0}}{{A_D^0}} x + A_D^0$ \n'
      '$A_C^0 = {0:.2f}$ ADU\n'
      '$A_D^0 = {1:.2f}$ ADU\n'
).format(peakC, peakD)
axes[1].text(x=35, y=47, s=s1, bbox=dict(facecolor='white', alpha=0.5))

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/ion_calibration.png', dpi=600)