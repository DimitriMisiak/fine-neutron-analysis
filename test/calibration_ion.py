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

stream = 'tg21l000'

        # 'tg18l005',
        # 'tg27l000',
        # 'tg28l000',
        # 'tg17l007',
        # 'tg19l010',
        # 'tg20l000',
        # 'tg21l000'


h5_path = '/home/misiak/Analysis/NEUTRON/data_xtalk.h5'.format(stream)

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

df_quality = df_analysis[df_analysis.quality_cut]

heat = df_quality.energy_adu_heat
ion_a = df_quality.energy_adu_corr_ionA
ion_b = df_quality.energy_adu_corr_ionB
ion_c = df_quality.energy_adu_corr_ionC
ion_d = df_quality.energy_adu_corr_ionD

ion_tot = ion_a + ion_b - ion_c - ion_d
ion_cons = ion_a + ion_b + ion_c + ion_d

calib_selection = (
    (df_quality.energy_adu_heat > 500)
    & (abs(ion_tot) > 100)
    & (abs(ion_tot) < 125)
)

df_calib = df_quality[calib_selection]

heat = df_calib.energy_adu_heat
ion_a = df_calib.energy_adu_corr_ionA
ion_b = df_calib.energy_adu_corr_ionB
ion_c = df_calib.energy_adu_corr_ionC
ion_d = df_calib.energy_adu_corr_ionD

ion_tot = ion_a + ion_b - ion_c - ion_d
ion_cons = ion_a + ion_b + ion_c + ion_d
# =============================================================================
# histogramm
# =============================================================================

fig, axes = plt.subplots(
    nrows=7,
    figsize=(6.3, 8)
)

ax = axes[0]
ax.hist(heat, bins=500)

ax = axes[1]
ax.hist(ion_a, bins=250)

ax = axes[2]
ax.hist(ion_b, bins=250)

ax = axes[3]
ax.hist(ion_c, bins=250)

ax = axes[4]
ax.hist(ion_d, bins=250)

ax = axes[5]
ax.hist(ion_tot, bins=250)

ax = axes[6]
ax.hist(ion_cons, bins=100)

for ax in axes:
    ax.grid()

fig.tight_layout()
fig.subplots_adjust(hspace=.0)

#%%
# =============================================================================
# Calibration A and B
# =============================================================================

fig, ax = plt.subplots()

ax.plot(ion_b, ion_a, ls='none', marker ='.')
ax.grid()

from scipy.odr import ODR, Model, Data, RealData

def func(beta, x):
    y = (-beta[1]/beta[0])*x+beta[1]
    return y


### Calibration A & B
data = RealData(ion_b, ion_a, 5, 5)
model = Model(func)
   
odr = ODR(data, model, [1,1])

odr.set_job(fit_type=0)
output = odr.run()

xn = np.linspace(
    ion_b.min(),
    ion_a.max(),
    int(1e3)
)       
yn = func(output.beta, xn)

ax.plot(xn,yn,'r-',label='odr')
ax.errorbar(
        [output.beta[0], 0],
        [0, output.beta[1]],
        yerr=0.5,
        xerr=0.5,
        ls='none',
        marker='o'
)

print("Calibration B, A")
print(output.beta)

#%%
# =============================================================================
# Calibration C and D
# =============================================================================

fig, ax = plt.subplots()

ax.plot(ion_d, ion_c, ls='none', marker ='.')
ax.grid()

from scipy.odr import ODR, Model, Data, RealData

def func(beta, x):
    y = (-beta[1]/beta[0])*x+beta[1]
    return y

data = RealData(ion_d, ion_c, 5, 5)
model = Model(func)
   
odr = ODR(data, model, [1,1])

odr.set_job(fit_type=0)
output = odr.run()

xn = np.linspace(
    ion_d.min(),
    ion_c.max(),
    int(1e3)
)       
yn = func(output.beta, xn)

ax.plot(xn,yn,'r-',label='odr')
ax.errorbar(
        [output.beta[0], 0],
        [0, output.beta[1]],
        yerr=0.5,
        xerr=0.5,
        ls='none',
        marker='o'
)

print("Calibration D, C")
print(output.beta)
