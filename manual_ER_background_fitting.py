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
cartoon_light = [
        pe.Stroke(linewidth=2, foreground='k'),
        pe.Normal(),
]

from plot_addon import lighten_color
from stats_addon import cdf_calc
from pipeline_data_science import sigma_function


def chi2_simple(data_1, data_2, err):
    """ Simplest chi2 function comparing two sets of datas.

    Parameters
    ==========
    data_1, data_2 : array_like
        Set of data to be compared.
    err : float or array_like
        Error affecting the data. If float is given, the error is the same
        for each points. If array_like, each point is affected by its
        corresponding error.

    Returns
    =======
    x2 : float
        Chi2 value.
    """
    array_1 = np.array(data_1)
    array_2 = np.array(data_2)
    err_array = np.array(err)
    x2 = np.sum( (array_1 - array_2)**2  / err_array**2 )
    return x2


plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'

### SIMU
h5_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])   
df_simu = pd.read_hdf(
    h5_simu_path,
    key='df',
)

all_cut = np.ones(shape=df_simu.shape[0], dtype=bool)
trigger_cut = all_cut & df_simu.trigger_cut 
quality_cut = trigger_cut & df_simu.quality_cut
charge_cut = quality_cut & df_simu.charge_conservation_cut
bulk_cut = charge_cut & df_simu.bulk_cut

gamma_cut = bulk_cut & df_simu.gamma_cut
neutron_cut = bulk_cut & df_simu.neutron_cut
ho_cut = bulk_cut & df_simu.HO_cut

### DATA
h5_data_path = '/'.join([analysis_dir, 'data_science.h5'])   
df_data = pd.read_hdf(
    h5_data_path,
    key='df',
)

quality_cut_data = df_data.quality_cut
charge_cut_data = quality_cut_data & df_data.charge_conservation_cut
bulk_cut_data = charge_cut_data & df_data.bulk_cut

gamma_cut_data = bulk_cut_data & df_data.gamma_cut
neutron_cut_data = bulk_cut_data & df_data.neutron_cut
ho_cut_data = bulk_cut_data & df_data.HO_cut


source_list = ['Background', 'Calibration']
simulation_list =  ['NR', 'ER']

# dx = 0.25
# bins = np.arange(0, 50 + dx, dx)
# bins_width = bins[1] - bins[0]
# bins_array = bins[:-1] + (bins_width) / 2
# eff_x_array = bins_array

# bins = np.logspace(np.log(0.2), np.log(50), 100, base=np.exp(1))
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]



std0 = df_data.std_energy_heat.unique().max()
std10 = df_data.std_calib_energy_heat.unique().max()

nsigma = 2

bins_list = [0,]
while ( bins_list[-1] < 50 ):
    last_bin = bins_list[-1]
    dx = sigma_function(last_bin, std0*nsigma, std10*nsigma)
    bins_list.append(last_bin + dx)

print('Bins OK')

bins = np.array(bins_list)
bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]
bins_array = bins[:-1] + (bins_width) / 2

hack_cut = ~(
    (bins_array < 0.5)
    | ( (bins_array > 8) & (bins_array < 9) )
    | ( (bins_array > 20) & (bins_array < 22) )
)
# hack_cut = ~(
#     (bins_array < 0)
# )


#%%
# =============================================================================
# PLOT
# =============================================================================

# source = 'Background'
source = 'Calibration'
recoil = 'ER'

source_cut = (df_simu.source == source)
# recoil_cut = (df_simu.simulation == 'flat_ER')
band_cut = ( gamma_cut & ~neutron_cut & ~ho_cut )
# df_local = df_simu[ source_cut &  band_cut]

source_cut_data = (df_data.source == source)
band_cut_data = ( gamma_cut_data & ~neutron_cut_data & ~ho_cut_data)
df_local_data = df_data[ source_cut_data &  band_cut_data]

components_labels = ['flat_ER', 'line_1keV', 'line_10keV' ]

gamma_components = list()
for i, simu in enumerate(components_labels):
    df_all = df_simu[
        source_cut
        & (df_simu.simulation == simu)
    ]    

    df_local = df_simu[
        source_cut
        & band_cut
        & (df_simu.simulation == simu)
    ]

    # hist_array = np.histogram(df_local.recoil_energy_bulk, bins=bins)[0]
    hist_array = np.histogram(df_local.energy_heat, bins=bins)[0]
    
    ### HACK
    hist_array = hist_array[hack_cut]
    
    # gamma_components.append(hist_array / hist_array.sum())
    gamma_components.append(hist_array /  df_all.shape[0])

# hist_data = np.histogram(df_local_data.recoil_energy_bulk, bins=bins)[0]
hist_data = np.histogram(df_local_data.energy_heat, bins=bins)[0]  
hist_data = hist_data[hack_cut]

bins_array = bins_array[hack_cut]

#%%

fig, axes = plt.subplots(
    figsize=(6.3, 6),
    nrows=2,
    # ncols=2,
    sharex='col',
    # sharey='row'
)


def pure_gamma_model(x):
    a,b,c = x
    er_background = (
        a * gamma_components[0]
        + b * gamma_components[1] 
        + c * gamma_components[2]
    )
    return er_background

for comp in gamma_components:
    axes[0].plot(
        bins_array,
        comp,
        drawstyle='steps-mid'
    )
  
x0 = (1000, 1000, 10000)
mod0 = pure_gamma_model(x0)
mod0_err = mod0**0.5
mod0_err[mod0_err<1] = 1
    
import scipy.stats as st
poisson_error = st.poisson(mod0)
# poisson_error = st.norm(mod0)
mod1 = poisson_error.rvs()

x2_test = chi2_simple(mod0, mod1, mod0_err)
print("Chi2 test:")
print(x2_test)

from scipy.optimize import minimize

# x1 = (500, 56, 999)
# data_simu = pure_gamma_model(x1)

# def to_be_minimized(x):
#     data = data_simu
#     model = pure_gamma_model(x)
#     x2 = chi2_simple(data, model, model**0.5) 
#     return x2

# res = minimize(to_be_minimized, [1000, 100, 1000])

def to_be_minimized(x):
    data = hist_data
    model = pure_gamma_model(x)
    model_err = (2*model)**0.5
    model_err[model_err<1] = 1
    print(model_err)
    x2 = chi2_simple(data, model, model_err) 
    return x2

if source == 'Calibration':
    # res = minimize(to_be_minimized, [36610, 65312, 660798], method='Nelder-Mead')
    res = minimize(to_be_minimized, 
                   [  9808.93072736*3,  25284.01681568*3, 214830.52244527*3],
                   method='Nelder-Mead')
if source == 'Background':
    res = minimize(to_be_minimized, [14133, 25089, 271422], method='Nelder-Mead')
    
mod_opt = pure_gamma_model(res.x)

axes[1].errorbar(
    bins_array,
    mod_opt,
    yerr = (2*mod_opt)**0.5,
    drawstyle='steps-mid'
)

axes[1].plot(
    bins_array,
    hist_data,
    ls='none',
    marker='.',
    color='k'
)

for ax in axes:
    ax.set_yscale('log')

print(res)


