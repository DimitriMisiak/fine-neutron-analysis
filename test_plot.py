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


from plot_addon import (
    ax_hist_v2,
)

from pipeline_data_heat_calib import (
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
    lindhard,
    quenching,
    energy_recoil,
)

from pipeline_data_science import (
    sigma_function,
)


plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'
output_dir = '/'.join([analysis_dir, 'analysis_plots'])
extension='pdf'

h5type = 'data'

h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])   
 
# source = 'Calibration'
source = 'Background'

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'source = "{}"'
    ).format(source)
)

all_cut = (
    df_analysis['quality_cut']
    & df_analysis['charge_conservation_cut']
    & df_analysis['bulk_cut']
)

if h5type == 'simu':
    all_cut = all_cut & df_analysis['trigger_cut']

neutron_cut = df_analysis['neutron_cut'] & all_cut
gamma_cut = df_analysis['gamma_cut'] & all_cut
ho_cut = df_analysis['HO_cut'] & all_cut

df_all = df_analysis[all_cut]
df_neutron = df_analysis[neutron_cut]
df_gamma = df_analysis[gamma_cut]
df_ho = df_analysis[ho_cut]


nsigma = 2

x_range = np.linspace(0, 25, int(1e4))

sig0 = df_all['std_energy_ion_bulk'].unique().mean()
sig10 = df_all['std_calib_energy_ion_bulk'].unique().mean()

# ei_err = nsigma * sigma_function(x_range, sig0, sig10)
# ei_err_baseline = nsigma * sig0

# gamma_cut = ( abs(energy_ion - energy_heat) < ei_err )

# er_array = np.linspace(0, energy_heat.max(), int(1e4))
# dv=2
# # ec_array = er_array * (1 + lindhard(er_array)*dv/3) / (1 + dv/3)
# ec_array = energy_heat_from_er_and_quenching(
#     er_array,
#     lindhard(er_array),
#     dv
# )
# # ei_array = er_array * lindhard(er_array)
# ei_array = energy_ion_from_er_and_quenching(
#     er_array,
#     lindhard(er_array)
# )

# energy_ion_lindhard = np.interp(energy_heat, ec_array, ei_array)

# neutron_cut = ( abs(energy_ion - energy_ion_lindhard) < ei_err )

# HO_cut = ( abs(energy_ion) < ei_err_baseline )

V = 2

er_theory = np.linspace(0, 100, int(1e4))

qu_neutron = lindhard(er_theory)
qu_gamma = np.ones(er_theory.shape)
qu_ho = np.zeros(er_theory.shape)

# ec_gamma = energy_heat_from_er_and_quenching(er_theory, qu_gamma, V)
# ei_gamma = energy_ion_from_er_and_quenching(er_theory, qu_gamma)

# sigma_ion = sigma_function(ec_gamma, sig0, sig10)

# qu_gamma = np.ones(int(1e4))
# ec_gamma = energy_heat_from_er_and_quenching(er_theory, qu_gamma, 2)
# ei_gamma = energy_ion_from_er_and_quenching(er_theory, qu_gamma)
# ei_err_gamma = nsigma*std_energy_ion(ec_gamma)

# qu_gamma_sup_aux = quenching(ec_gamma, ei_gamma + ei_err_gamma, 2)
# er_gamma_sup = energy_recoil(ec_gamma, ei_gamma + ei_err_gamma, 2)
# qu_gamma_sup = np.interp(er_theory, er_gamma_sup, qu_gamma_sup_aux)

# qu_gamma_inf_aux = quenching(ec_gamma, ei_gamma - ei_err_gamma, 2)
# er_gamma_inf = energy_recoil(ec_gamma, ei_gamma - ei_err_gamma, 2)
# qu_gamma_inf = np.interp(er_theory, er_gamma_inf, qu_gamma_inf_aux)

# # neutron
# qu_neutron = lindhard(er_theory)
# ec_neutron = energy_heat_from_er_and_quenching(er_theory, qu_neutron, 2)
# ei_neutron = energy_ion_from_er_and_quenching(er_theory, qu_neutron)
# ei_err_neutron = nsigma*std_energy_ion(ec_neutron)

# qu_neutron_sup_aux = quenching(ec_neutron, ei_neutron + ei_err_neutron, 2)
# er_neutron_sup = energy_recoil(ec_neutron, ei_neutron + ei_err_neutron, 2)
# qu_neutron_sup = np.interp(er_theory, er_neutron_sup, qu_neutron_sup_aux)

# qu_neutron_inf_aux = quenching(ec_neutron, ei_neutron - ei_err_neutron, 2)
# er_neutron_inf = energy_recoil(ec_neutron, ei_neutron - ei_err_neutron, 2)
# qu_neutron_inf = np.interp(er_theory, er_neutron_inf, qu_neutron_inf_aux)



df_list = [
    df_all,
    df_neutron,
    df_gamma,
    df_ho,
]

color_list = [
    'grey',
    'coral',
    'forestgreen',
    'slateblue',    
]

quenching_list = [
    qu_neutron,
    qu_gamma,
    qu_ho
]

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(
    figsize=(6.3, 3.9),
)

ax.plot(
    df_analysis[df_analysis.quality_cut].energy_heat,
    df_analysis[df_analysis.quality_cut].energy_ion_total,
    marker='.',
    markersize=1.5,
    alpha=0.2,
    ls='none'
)


### ECEI
ax.set_xlabel('$E_{heat}$ / keV$_{ee}$')
ax.set_ylabel('$E_{Ion.}^{total}$ / keV')
ax.set_ylim(-2, 15)
ax.set_xlim(0, 15)

ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

                          

## GLOBAL FIGURE PARAMETERS
ax.grid(True, alpha=0.5, which='major')
ax.grid(True, alpha=0.1, which='minor')
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))


leg = ax.legend(
    title='RED80, Calibration streams\nAll events passing Quality Cuts',
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=3
)

plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.subplots_adjust(hspace=.0)

# fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/band_cuts.pdf')
fig.savefig(
    '/home/misiak/Bureau/ec_vs_ei.png',
    dpi=600
)
