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
from scipy.optimize import minimize

cartoon = [
        pe.Stroke(linewidth=3, foreground='k'),
        pe.Normal(),
]
cartoon_light = [
        pe.Stroke(linewidth=2, foreground='k'),
        pe.Normal(),
]

from plot_addon import (
    lighten_color,
    LegendTitle
)

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

from stats_addon import cdf_calc

from pipeline_data_raw import stream_configuration
from pipeline_data_quality import quality_parameters
from pipeline_data_heat_calib import energy_heat_from_er_and_quenching, lindhard

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1
from tqdm import tqdm

analysis_dir = '/home/misiak/Analysis/NEUTRON'

h5_data_path = '/'.join([analysis_dir, 'data_science.h5'])   
df_data = pd.read_hdf(
    h5_data_path,
    key='df',
)

eff_path = '/home/misiak/Analysis/NEUTRON/efficiency_dict.npy'
eff_dict = np.load(eff_path, allow_pickle=True).item()

corr_path = '/home/misiak/Analysis/NEUTRON/contamination_dict.npy'
corr_dict = np.load(corr_path, allow_pickle=True).item()


all_cut = np.ones(shape=df_data.shape[0], dtype=bool)
quality_cut = all_cut & df_data.quality_cut
charge_cut = quality_cut & df_data.charge_conservation_cut
bulk_cut = charge_cut & df_data.bulk_cut
gamma_cut = bulk_cut & df_data.gamma_cut & (df_data.recoil_energy_bulk < 50)
neutron_cut = bulk_cut & df_data.neutron_cut & (df_data.recoil_energy_bulk < 50)


source_list = ['Background', 'Calibration']
simulation_list =  ['NR', 'ER']

# dx = 1
# bins = np.arange(0, 50 + dx, dx)
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1] + (bins_width) / 2
# eff_x_array = bins_array

# bins = np.logspace(np.log(0.2), np.log(50), 100, base=np.exp(1))
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]

bins = eff_dict['bins']
bins_width = (bins[1:] - bins[:-1])
bins_array = bins[:-1] + (bins_width) / 2


###  ER from E heat and Quenching for nuclear recoils
er_array = np.linspace(0, 50, 1000)
qu_array = lindhard(er_array)
eh_array = energy_heat_from_er_and_quenching(er_array, qu_array, 2)

def recoil_energy_from_eh_for_NR(eh):
    return np.interp(eh, eh_array, er_array)

#%%
# =============================================================================
# PLOT
# =============================================================================
mass_ge = 0.038 #kg

exposure_dict = dict()
for source in source_list:
    
    exposure = 0
    for stream in stream_configuration[source]:
        
        raw_length = df_data[df_data.stream == stream].timestamp.max()
        
        glitch_time = quality_parameters[stream]['glitch_time_cut']
        malus = 0
        for window in glitch_time:
            inf, sup = window
            # to prevent infinity to mess with this
            if sup > raw_length:
                sup = raw_length
            if inf < 0:
                inf = 0
            malus += (sup - inf)
        
        exposure += (raw_length - malus)
    
        # for debug
        print(stream)
        print(raw_length)
        print(raw_length-malus)
    
    
    print('Exposure')
    print(exposure)

    exposure_dict[source] = exposure / 24 # from hours to days

DRU_dict = dict()
inte_dict = dict()
for source in ['Background', 'Calibration']:

    source_cut = (df_data.source == source)
    
    DRU_dict[source] = dict()
    for recoil in ['ER', 'NR']:

        if recoil == 'NR':
            band_cut = neutron_cut
        if recoil == 'ER':
            band_cut = gamma_cut
            
        df_local = df_data[source_cut & band_cut]

        # # data_bin_array = adv_data_dict[mode][stype]
        # counts_array = np.histogram(
        #     df_local.recoil_energy_bulk,
        #     bins=bins
        # )[0]
        counts_array = np.histogram(
            recoil_energy_from_eh_for_NR(df_local.energy_heat),
            bins=bins
        )[0]
        
        ### CONTAMINATION CORRECTION
        if recoil == 'ER':
            # counts_array = counts_array + corr_dict[source]['NR_correction']
            counts_array = corr_dict[source]['ER_background']
            
            
        if recoil == 'NR':
            counts_array = counts_array - corr_dict[source]['NR_correction']
            
        
        eff_array = eff_dict[source][recoil]['Band']
        exposure = exposure_dict[source]

        DRU_dict[source][recoil] = counts_array / (eff_array * exposure * bins_width * mass_ge)

        if recoil == 'NR':
            # cut_2kev = (bins_array >= 2)
            # cut_2kev = (bins_array >= 1)
            cut_2kev = (bins_array >= 1.16)
            
            # inte = np.trapz(
            #         DRU_dict[mode][stype][cut_2kev],
            #         bins_array[cut_2kev]
            # )

            inte = np.sum(
                DRU_dict[source][recoil][cut_2kev] * bins_width[cut_2kev]
            )

            inte_dict[source] = inte



inte_bkgd = inte_dict['Background']
inte_calib = inte_dict['Calibration']
ratio = inte_calib / inte_bkgd

# fig.suptitle(
#         (
#                 'In ROI [2keV-50keV]: {:.1f} Calib / {:.1f} Bkgd = {:.2f} Ratio'
#         ).format(inte_calib, inte_bkgd, ratio)
# )


#%%
### Money plot
calib_er = DRU_dict['Calibration']['ER']
calib_nr = DRU_dict['Calibration']['NR']
bkgd_er = DRU_dict['Background']['ER']
bkgd_nr = DRU_dict['Background']['NR']

def funk(x, a,c):
    return a * np.exp(-x/c)

def to_be_minimized(x):
    a,b,c = x
    
    cut_2kev = (bins_array >= 1.16)
    cut_2kev = (bins_array >= 3)
    
    data1 = calib_nr[cut_2kev]
    data2 = bkgd_nr[cut_2kev]
    
    # model1 = a * np.exp(-bins_array / c)
    # model2 = b * np.exp(-bins_array / c)
    
    model1 = funk(bins_array[cut_2kev], a,c)
    model2 = funk(bins_array[cut_2kev], b,c)
    
    
    # model_err = (2*model)**0.5
    # model_err[model_err<1] = 1
    # print(model_err)
    x2 = (
        chi2_simple(data1, model1, model1**0.5)
        + chi2_simple(data2, model2, model2**0.5)
    )
    # x2 = (
    #     chi2_simple(data1, model1, model1*0.1)
    #     + chi2_simple(data2, model2, model2*1)
    # )    
    
    return x2

x0 = [3e5, 3e5, 10]
res = minimize(to_be_minimized, x0, method='Nelder-Mead')


calib_tot = calib_er + calib_nr
bkgd_tot = bkgd_er + bkgd_nr

array_list = [
    # calib_tot,
    # bkgd_tot,
    calib_er,
    calib_nr,
    bkgd_er,
    bkgd_nr
]

color_list = [
    # 'grey',
    # 'lightgrey',
    'red',
    'blue',
    'coral',
    'cornflowerblue',
    ]

legend_list =[
    'Calibration ER band',
    'Calibration NR band',
    'Background ER band',
    'Background NR band',
]

fig, axes = plt.subplots(figsize=(6.3, 7), nrows=2)

for i,dru_array in enumerate(array_list):

    c = color_list[i]
    leg = legend_list[i]
    
    zorder=1
    
    if i in (0,2):
        zorder=5
    
    for ax in axes:
        ax.plot(
            bins_array,
            dru_array,
            drawstyle='steps-mid',
            alpha=1,
            color=c,
            lw=2,
            zorder=zorder,
            #path_effects=cartoon,
            label=leg
        )

    # ax.plot(
    #     bins_array,
    #     dru_array,
    #     ls='none',
    #     alpha=1,
    #     # marker='.',
    #     color=c,
    #     zorder=zorder,
    #     #path_effects=cartoon,
    # )

    if i in (1,3):
        
        if i == 1:
            msg = (
                'Calibration events in [2keV-50keV]:\n{:.2e} Counts/kg/days'
            ).format(inte_dict['Calibration'])
        if i == 3:
            msg = (
                'Background events in [2keV-50keV]:\n{:.2e} Counts/kg/days'
            ).format(inte_dict['Background'])
        
        for ax in axes:
            ax.fill_between(
                bins_array,
                dru_array,
                step='mid',
                color=lighten_color(c),
                zorder=-1,
                label=leg
            )
        
        print(msg)

for ax in axes:
    ax.axvspan(0, 1, color='k', alpha=0.3, zorder=5)
    
    ax.plot(
            bins_array,
            # 2.70808832e+04 * np.exp(-bins_array/6.11248034e+00),
            1.21626561e+04 * np.exp(-bins_array/6.87510284e+00),
            color='k',
            label='Background Adjusted Kinemtaic Distribution',
            lw=2,
    )
    
    ax.plot(
            bins_array,
            # 7.42133241e+05 * np.exp(-bins_array/6.11248034e+00),
            5.58464462e+05 * np.exp(-bins_array/6.87510284e+00),
            color='k',
            ls='--',
            lw=2,
            label='Calibration Adjusted Kinemtaic Distribution'
    )

# ax.axvline(2, ls='--', color='k', zorder=5, 
#             label='Analysis Threshold: 2keV')

# fig.suptitle(
#         (
#                 'In ROI [2keV-50keV]: {:.1f} Calib / {:.1f} Bkgd = {:.2f} Ratio'
#         ).format(inte_calib, inte_bkgd, ratio)
# )
for ax in axes:
    
    ax.set_xlim(0.5, 50)
    ax.set_ylim(1e2, 2e7)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Dayly Event Rate / ( Counts/keV/kg/days )')
    ax.set_xlabel('$E_R$ / keV')
    
    ax.grid(which='both', alpha=0.5)
    fig.tight_layout()

axes[1].set_xscale('linear')

# axes[0,0].set_ylabel(r'$\mathcal{F}$')
# axes[1,0].set_ylabel(r'$\mathcal{F}$')
# axes[1,0].set_xlabel(r'$E_R^{input}$ / keV')
# axes[1,1].set_xlabel(r'$E_R^{input}$ / keV')

# fig.tight_layout()
# fig.subplots_adjust(hspace=0, wspace=0)

# ## GLOBAL FIGURE PARAMETERS
# for ax in np.ravel(axes):
#     ax.grid(True, alpha=0.5, which='major')
#     ax.grid(True, alpha=0.1, which='minor')
#     ax.set_yscale('log')
#     ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
#     ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
#     ax.set_ylim(5e-4, 2)
#     ax.set_xlim(0, 49.9)

# axes[0, 0].spines['bottom'].set_linewidth(2)
# axes[0, 1].spines['bottom'].set_linewidth(2)
# axes[0, 0].spines['right'].set_linewidth(2)
# axes[1, 0].spines['right'].set_linewidth(2)


leg = axes[0].legend(
    handles=[
        plt.Line2D([], [], color='red', lw=3),
        plt.fill_between([], [], color=lighten_color('blue'), edgecolor='blue'),
        plt.Line2D([], [], color='k', ls ='--', lw=2),
        plt.fill_between([], [], color=lighten_color('grey'), edgecolor='grey'),
        plt.Line2D([], [], color='coral', lw=3),
        plt.fill_between([], [], color=lighten_color('cornflowerblue'), edgecolor='cornflowerblue'),
        plt.Line2D([], [], color='k', lw=2),
    ],
    labels =[
        'Calibration Electronic Background',
        'Calibration Nuclear Background',
        'Calibration Adjusted NR Model',
        'Affected by Noise Blob',
        'Background Electronic Background',
        'Background Nuclear Background',
        'Background Adjusted NR Model'
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    frameon=False,
    ncol=2
)

# plt.setp(leg.get_title(), multialignment='center')

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
# fig.subplots_adjust(hspace=.0, wspace=.0)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/neutron_background.pdf')

A = np.array([bins_array, calib_er, calib_nr, bkgd_er, bkgd_nr])
np.savetxt(
    '/home/misiak/Analysis/NEUTRON/thesis_plots/DRU_arrays.txt',
    A.T,
    delimiter=',',
    header='Bins_center, Calibration ER, Calibration NR, Background ER, Background NR'
)
np.savetxt(
    '/home/misiak/Analysis/NEUTRON/thesis_plots/Bin_edges.txt',
    bins,
    header='Bins edges'
)