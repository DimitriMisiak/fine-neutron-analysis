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

stream = 'tg18l005'

h5_path = '/home/misiak/Analysis/NEUTRON/data_ion_calib.h5'.format(stream)
selection_path = '/home/misiak/Analysis/NEUTRON/{0}_10kev_selection.npy'.format(stream)
ind = np.load(selection_path)

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

df_quality = df_analysis[df_analysis.quality_cut]



#%%
df_reset = df_quality.reset_index(drop=True)
selection_cut = np.in1d(np.array(df_reset.index), ind, assume_unique=True)
df = df_quality[selection_cut]

heat = df.energy_adu_heat
ion = df.energy_ion_total

#%%
# =============================================================================
# Calibration A and B
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.3, 3.9)
)

ax.plot(
        df_quality.energy_adu_heat,
        df_quality.energy_ion_total,
        ls='none', marker='.', color='coral', alpha=0.3
)
ax.plot(heat, ion, ls='none', marker ='.', color='slateblue', alpha=0.3)



def func(beta, x):
    y = (x - beta[0]) * 10.37 / (beta[1] - beta[0])
    return y

### Calibration A & B
data = RealData(heat, ion, 20, 1)
model = Model(func)
   
odr = ODR(data, model, [500,1000])

odr.set_job(fit_type=0)
output = odr.run()

xn = np.linspace(
    heat.min(),
    heat.max(),
    int(1e3)
)       
yn = func(output.beta, xn)
beta = output.beta

# xinf = -beta[1]/beta[0]
# xsup = (10.37-beta[1])/beta[0]

ax.plot(
        [beta[0], beta[1]],
        [0, 10.37],
        ls='-',
        marker='o',
        color='k'
)



# =============================================================================
# plot formatting
# =============================================================================

ax.set_xlabel('Heat Amplitude $A_{heat}$ / ADU')
ax.set_ylabel('Total Ionization Energy $E_{Ion.}^{tot}$ / keV')

ax.set_xlim(0, 1400)
ax.set_ylim(-2, 13)

ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))

ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

ax.grid(True, alpha=0.5, which='major')
ax.grid(True, alpha=0.1, which='minor')


# for i in range(ndim):
#     for j in range(len(ind_list)-1):
#         axes[i,j].spines['right'].set_linewidth(2)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))


ax.legend(
    title=(
        'Heat Channel Calibration\nRED80, stream tg18l005'
    ),
    handles=[
        plt.Line2D([], [], ls='none', marker='o', color='coral'),
        plt.Line2D([], [], ls='none', marker='o', color='slateblue'),
        plt.Line2D([], [], ls='-', marker='o', color='k'),
    ],
    labels=[
        'Quality events',
        'Lasso selection of\n10.37 keV Calibration events',
        'Adjusted Linear law',
    ],
    loc='center left',
    bbox_to_anchor=(1, 0.75),
    frameon=False
)

s0 = (
      '$y = \\frac{{10.37}}{{ A_{{heat}}^{{peak}} - A_{{heat}}^{{HO}} }} (x - A_{{heat}}^{{HO}})$ \n'
      '$A_{{heat}}^{{HO}} = {0:.1f}$ ADU\n'
      '$A_{{heat}}^{{peak}} = {1:.0f}$ ADU\n'
).format(beta[0], beta[1])
ax.text(x=1450, y=1, s=s0, bbox=dict(facecolor='white', alpha=0.5))

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.tight_layout(rect=(0, 0, 1, 1))
# fig.subplots_adjust(hspace=.0, wspace=0.)


fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/heat_calibration.png', dpi=600)