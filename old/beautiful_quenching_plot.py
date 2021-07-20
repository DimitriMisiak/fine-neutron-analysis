#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 08:36:14 2020

@author: misiak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

cartoon = [
        pe.Stroke(linewidth=3, foreground='k'),
        pe.Normal(),
]

from pipeline_data_calibrated import (
    quenching,
    energy_recoil,
    lindhard,
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
)

from pipeline_data_science import (
    charge_conservation_threshold,
    std_energy_ion,  
)


plt.close('all')
plt.rcParams['text.usetex']=True

analysis_dir = '/home/misiak/Analysis/neutron_background'
analysis_data_path = '/'.join([analysis_dir, 'data_science.h5'])

df_analysis = pd.read_hdf(
    analysis_data_path,
    key='df',
    where='source="Calibration"'
)

nsigma = 2
title = 'Calibration'

# =============================================================================
# ACTUAL PLOT
# =============================================================================

quality_cut = df_analysis['quality_cut']
bulk_cut = df_analysis['bulk_cut']

all_cut = quality_cut & bulk_cut
try:
    # for simulation
    all_cut = all_cut & df_analysis['trigger_cut']
except:
    pass    

neutron_cut = df_analysis['neutron_cut'] & all_cut
gamma_cut = df_analysis['gamma_cut'] & all_cut
ho_cut = df_analysis['HO_cut'] & all_cut

#%%
   
    
fig_band_ecei, ax_ecei = plt.subplots(
    num = '{} : band cut ecei'.format(title),
    # figsize=(10,7),
)
ax_ecei.set_title('{} : band cut ecei'.format(title))
fig_band_quenching, ax_qu = plt.subplots(
    num = '{} : band cut quenching'.format(title),
    # figsize=(10,7),
)
# ax_qu.set_title('{} : band cut quenching'.format(title))

ec_array = df_analysis[all_cut]['energy_heat']
ei_array = df_analysis[all_cut]['energy_ion_bulk']
er_array = energy_recoil(ec_array, ei_array, 2)
qu_array = quenching(ec_array, ei_array, 2)

# from scipy.stats import binned_statistic
# A = binned_statistic(
#     er_array,
#     ei_array,
#     bins=100
# )
# plt.figure()
# plt.plot(er_array, ei_array, ls='none', marker='.')

ax_ecei.plot(
    ec_array,
    ei_array,
    ls='none',
    marker='.',
    color='k',
    alpha=0.3,
    label='data',
)

line_data,= ax_qu.plot(
    er_array,
    qu_array,
    ls='none',
    marker='.',
    color='k',
    alpha=0.3,
    label='data'
)

line_data.set_markeredgewidth(0)


er_theory = np.linspace(0, 100, int(1e4))

# gamma
qu_gamma = np.ones(int(1e4))
ec_gamma = energy_heat_from_er_and_quenching(er_theory, qu_gamma, 2)
ei_gamma = energy_ion_from_er_and_quenching(er_theory, qu_gamma)
ei_err_gamma = nsigma*std_energy_ion(ec_gamma)

qu_gamma_sup_aux = quenching(ec_gamma, ei_gamma + ei_err_gamma, 2)
er_gamma_sup = energy_recoil(ec_gamma, ei_gamma + ei_err_gamma, 2)
qu_gamma_sup = np.interp(er_theory, er_gamma_sup, qu_gamma_sup_aux)

qu_gamma_inf_aux = quenching(ec_gamma, ei_gamma - ei_err_gamma, 2)
er_gamma_inf = energy_recoil(ec_gamma, ei_gamma - ei_err_gamma, 2)
qu_gamma_inf = np.interp(er_theory, er_gamma_inf, qu_gamma_inf_aux)

# neutron
qu_neutron = lindhard(er_theory)
ec_neutron = energy_heat_from_er_and_quenching(er_theory, qu_neutron, 2)
ei_neutron = energy_ion_from_er_and_quenching(er_theory, qu_neutron)
ei_err_neutron = nsigma*std_energy_ion(ec_neutron)

qu_neutron_sup_aux = quenching(ec_neutron, ei_neutron + ei_err_neutron, 2)
er_neutron_sup = energy_recoil(ec_neutron, ei_neutron + ei_err_neutron, 2)
qu_neutron_sup = np.interp(er_theory, er_neutron_sup, qu_neutron_sup_aux)

qu_neutron_inf_aux = quenching(ec_neutron, ei_neutron - ei_err_neutron, 2)
er_neutron_inf = energy_recoil(ec_neutron, ei_neutron - ei_err_neutron, 2)
qu_neutron_inf = np.interp(er_theory, er_neutron_inf, qu_neutron_inf_aux)

# # heat only
# qu_ho = np.zeros(int(1e4))
# ec_ho = energy_heat_from_er_and_quenching(er_theory, qu_ho, 2)    
# ei_ho = energy_ion_from_er_and_quenching(er_theory, qu_ho)
# ei_err_ho = nsigma*std_energy_ion(ec_ho)

# qu_ho_sup = quenching(ec_ho, ei_ho + ei_err_ho, 2)
# qu_ho_inf = quenching(ec_ho, ei_ho - ei_err_ho, 2)    

# qu_ho_sup_aux = quenching(ec_ho, ei_ho + ei_err_ho, 2)
# er_ho_sup = energy_recoil(ec_ho, ei_ho + ei_err_ho, 2)
# qu_ho_sup = np.interp(er_theory, er_ho_sup, qu_ho_sup_aux)

# qu_ho_inf_aux = quenching(ec_ho, ei_ho - ei_err_ho, 2)
# er_ho_inf = energy_recoil(ec_ho, ei_ho - ei_err_ho, 2)
# qu_ho_inf = np.interp(er_theory, er_ho_inf, qu_ho_inf_aux)


# GAMMA
line_gamma, = ax_ecei.plot(
    ec_gamma,
    ei_gamma,
    label='gamma band',
    color='coral',
    path_effects=cartoon,
    zorder=10
)
ax_ecei.fill_between(
    ec_gamma,
    ei_gamma + ei_err_gamma,
    ei_gamma - ei_err_gamma,
    label='gamma band',
    color='coral',
    alpha=0.1,
)

line_gamma_fill = mpatches.Patch(color='coral', alpha=0.25, linewidth=0)

ax_qu.plot(
    er_theory,
    qu_gamma,
    label='gamma band',
    color='coral',
    path_effects=cartoon
)

for Q in (qu_gamma_inf, qu_gamma_sup):
    line_gamma_sigma, = ax_qu.plot(
        er_theory,
        Q,
        label='gamma band',
        color='coral',
        ls='--',
        path_effects=cartoon
    )


ax_qu.fill_between(
    er_theory,
    qu_gamma_sup,
    qu_gamma_inf,
    label='gamma band',
    color='coral',
    alpha=0.25,
)

# NEUTRON
line_neutron, = ax_ecei.plot(
    ec_neutron,
    ei_neutron,
    label='neutron band',
    color='deepskyblue',
    path_effects=cartoon,
    zorder=10
)
ax_ecei.fill_between(
    ec_neutron,
    ei_neutron + ei_err_neutron,
    ei_neutron - ei_err_neutron,
    label='neutron band',
    color='deepskyblue',
    alpha=0.1,
)

line_neutron_fill = mpatches.Patch(color='deepskyblue', alpha=0.25, linewidth=0)


ax_qu.plot(
    er_theory,
    qu_neutron,
    label='neutron band',
    color='deepskyblue',
    path_effects=cartoon
)

for Q in (qu_neutron_inf, qu_neutron_sup):
    line_neutron_sigma, = ax_qu.plot(
        er_theory,
        Q,
        label='neutron band',
        color='deepskyblue',
        ls='--',
        path_effects=cartoon
    )

ax_qu.fill_between(
    er_theory,
    qu_neutron_sup,
    qu_neutron_inf,
    label='neutron band',
    color='deepskyblue',
    alpha=0.25,
)

# # HEAT ONLY
# ax_ecei.plot(
#     ec_ho,
#     ei_ho,
#     label='ho band',
#     color='lightgrey',
#     path_effects=cartoon
# )
# ax_ecei.fill_between(
#     ec_ho,
#     ei_ho + ei_err_ho,
#     ei_ho - ei_err_ho,
#     label='ho band',
#     color='lightgrey',
#     alpha=0.5,
# )

# ax_qu.plot(
#     er_theory,
#     qu_ho,
#     label='ho band',
#     color='lightgrey',
#     path_effects=cartoon
# )
# ax_qu.fill_between(
#     er_theory,
#     qu_ho_sup,
#     qu_ho_inf,
#     label='ho band',
#     color='lightgrey',
#     alpha=0.5,
# )

ax_ecei.set_ylabel('Ionization Energy [keV]')
ax_ecei.set_xlabel('Heat Energy [keV]')
ax_qu.set_ylabel('Quenching factor')
ax_qu.set_xlabel('Recoil Energy [keV]')    

ax_qu.set_ylim(-0.65, 1.5)
ax_qu.set_xlim(0, 13)

ax_ecei.set_xlim(-5, 13)
ax_ecei.set_ylim(-5, 13)

ax_qu.legend(
    handles=[
        line_gamma,
        (line_gamma_sigma, line_gamma_fill),
        line_data,
        line_neutron,
        (line_neutron_sigma, line_neutron_fill),
    ],
    labels=[
            'Electronic Recoil Quenching Q=1',
            '2$\sigma$ Electronic Recoil Band',
            'Data',
            'Linhard Nuclear Recoil Quenching',
            '2$\sigma$ Nuclear Recoil Band'
            ],
    loc='lower right',
    framealpha=1,
    ncol=2
    )


for ax in (ax_ecei, ax_qu):
    ax.grid()

for fig in (fig_band_ecei, fig_band_quenching):
    fig.tight_layout()
