#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:17:33 2020

@author: misiak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.close('all')

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


plt.figure()

plt.plot(df_quality.energy_adu_heat, df_quality.energy_ion_total, ls='none', marker='.', color='b')

#%%
df_reset = df_quality.reset_index(drop=True)
selection_cut = np.in1d(np.array(df_reset.index), ind, assume_unique=True)
df = df_quality[selection_cut]

heat = df.energy_adu_heat
ion = df.energy_ion_total

plt.plot(heat, ion, ls='none', marker='.', color='r')

#%%
# =============================================================================
# Calibration A and B
# =============================================================================

fig, ax = plt.subplots()

ax.plot(df_quality.energy_adu_heat, df_quality.energy_ion_total,
        ls='none', marker='.', color='b', alpha=0.3)
ax.plot(heat, ion, ls='none', marker ='.', color='r')
ax.grid()

from scipy.odr import ODR, Model, Data, RealData

def func(beta, x):
    y = beta[0] * x + beta[1]
    return y


### Calibration A & B
data = RealData(heat, ion, 20, 1)
model = Model(func)
   
odr = ODR(data, model, [1,1])

odr.set_job(fit_type=0)
output = odr.run()

xn = np.linspace(
    heat.min(),
    heat.max(),
    int(1e3)
)       
yn = func(output.beta, xn)
beta = output.beta

xinf = -beta[1]/beta[0]
xsup = (10.37-beta[1])/beta[0]

ax.plot(xn,yn,'r-',label='odr')
ax.errorbar(
        [xinf, xsup],
        [0, 10.37],
        yerr=0.5,
        xerr=0.5,
        ls='none',
        marker='o'
)

print("Calibration B, A")
print(output.beta)


