#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:27:11 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from plot_addon import basic_corner, lighten_color, LegendTitle

plt.rcParams['text.usetex']=True
plt.rcParams.update({'font.size': 13})
cartoon = [
        pe.Stroke(linewidth=3, foreground='k'),
        pe.Normal(),
]


ANA_DIR = '/home/misiak/Analysis/fine_root_analysis/fond_neutron'


data_npy_dict = {
        'all': 'all.npy',
        'quality': 'quality.npy'
}

simu_dict = {
        'ER': 'Flat_Analytical_SimuOnly_0.0000_50.0000_ER',
        'NR': 'Flat_Analytical_SimuOnly_0.0000_50.0000_NR',
        '1kev': 'Line_Analytical_SimuOnly_1.3000_ER',
        '10kev': 'Line_Analytical_SimuOnly_10.3700_ER',
}

simu_npy_dict = {
        'all': 'all.npy',
        'quality': 'quality.npy',
        'trigger': 'trigger.npy',
        'input_all': 'input_all.npy',
        'input_quality': 'input_quality.npy',
        'input_trigger': 'input_trigger.npy',
}


mode_list = ['Background', 'Calibration']


# The dictionnary D contains all the data/simu.
D = dict()

for mode in mode_list:

    D[mode] = dict()

    # DATA
    data_dir = '/'.join([ANA_DIR, mode, 'data'])
    simu_dir = '/'.join([ANA_DIR, mode, 'simu'])

    D_data = dict()
    for data_key, data_name in data_npy_dict.items():
        save_path = '/'.join([data_dir, data_name])
        load_array = np.load(save_path)
        if load_array.ndim > 1:
            # discarding heat B
            load_array = np.delete(load_array, 1, axis=1)
        D_data[data_key] = load_array

    D[mode]['data'] = D_data

    # SIMU
    D_simu = dict()
    for simu_key, simu_name in simu_npy_dict.items():

        D_type = dict()
        for simu_type, simu_dirname in simu_dict.items():
            save_path = '/'.join([simu_dir, simu_dirname, simu_name])
            load_array = np.load(save_path)
            if load_array.ndim > 1:
                # discarding heat B
                load_array = np.delete(load_array, 1, axis=1)
            D_type[simu_type] = load_array

        D_simu[simu_key] = D_type

    D[mode]['simu'] = D_simu

labs = ['heatA', 'ionA', 'ionB', 'ionC', 'ionD']
    
# =============================================================================
# PLOT
plt.close('all')
# =============================================================================

#%%
# =============================================================================
# FID CUT
# =============================================================================
std_nb_fid = 1

std_10kev_fid = 1.5

alpha_fid = (std_10kev_fid**2 - std_nb_fid**2)**0.5 / 10.37


def std_fid(collect):
    return (std_nb_fid**2 + (alpha_fid*collect)**2)**0.5


def gen_fid_cut(energy_array, debug=True):
    heat, A, B, C, D = energy_array.T
    fid_thresh_a = std_fid(B)
    fid_thresh_c = std_fid(D)
    # HACK THE POSITIVE HEAT IN and the limit to 50keV
    fid_cut = (
        (abs(A) < fid_thresh_a)
        & (abs(C) < fid_thresh_c)
        & (heat > 0.025)
        & (heat < 50)
    )
    return fid_cut


for mode in mode_list:

    data = D[mode]['data']['quality']
    D[mode]['data']['fid'] = data[gen_fid_cut(data)]

    D[mode]['simu']['fid'] = dict()
    D[mode]['simu']['input_fid'] = dict()
    for stype in simu_dict.keys():

        simu = D[mode]['simu']['quality'][stype]
        input_simu = D[mode]['simu']['input_quality'][stype]
        D[mode]['simu']['fid'][stype] = simu[gen_fid_cut(simu)]
        D[mode]['simu']['input_fid'][stype] = input_simu[gen_fid_cut(simu)]

mode = 'Calibration'
stype = 'NR'

fig, axes = basic_corner(
        D[mode]['data']['quality'][:, 1:],
        labs[1:],
        num='FID CUT ' + mode + ' data',
        label='quality',
        color='deepskyblue', alpha=0.1
)

fig, axes = basic_corner(
        D[mode]['data']['fid'][:, 1:],
        labs[1:],
        axes=axes,
        label='FID + quality',
        color='k', alpha=0.1,
)

fig, axes = basic_corner(
        D[mode]['simu']['quality'][stype][:, 1:],
        labs[1:],
        num='FID CUT ' + mode + ' simu',
        label='quality',
        color='deepskyblue', alpha=0.1
)

fig, axes = basic_corner(
        D[mode]['simu']['fid'][stype][:, 1:],
        labs[1:],
        axes=axes,
        label='FID + quality',
        color='k', alpha=0.1,
)

# =============================================================================
# # BAND CUTLegendTitle
# =============================================================================


def energy_recoil(ec, ei, V):
    coeff = V / 3
    return ec*(1+coeff) - ei*coeff


def quenching(ec, ei, V):
    er = energy_recoil(ec, ei, V)
    return ei/er


def lindhard(er):

    A = 72.63
    Z = 32

    k = 0.133 * Z**(2./3) * A**(-1./2)
    epsilon = 11.5 * er * Z**(-7./3)
    g = 3 * epsilon**0.15 + 0.7 * epsilon**0.6 + epsilon

    Q = k*g/(1+k*g)

    return Q

###
def gen_er(energy_array):
    dv = 2
    heat, A, B, C, Dd = energy_array.T
    collect = (B+Dd)/2
    er_array = energy_recoil(heat, collect, dv)
    return er_array

def gen_qu(energy_array):
    dv = 2
    heat, A, B, C, Dd = energy_array.T
    collect = (B+Dd)/2
    qu_array = quenching(heat, collect, dv)
    return qu_array    
    
    
std_nb = 0.254
#std_1kev = 0.272
std_10kev = 0.318

alpha = (std_10kev**2 - std_nb**2)**0.5 / 10.37
def std_collect(ec):
    return ( std_nb**2 + (alpha*ec)**2 )**0.5


er_array = np.linspace(0, 200, int(1e4))
dv=2
ec_array = er_array * (1 + lindhard(er_array)*dv/3) / (1 + dv/3)
ei_array = er_array * lindhard(er_array)

def band_cut(energy):

    heat_fid, A_fid, B_fid, C_fid, D_fid = energy.T

    collect_fid = (B_fid + D_fid)/2
#    collect = (B + D)/2

    # gamma cut
    #ec_graph = np.linspace(0, 100, int(1e4))
    #std_graph = std_collect(ec_graph)
    #ax.plot(ec_graph, ec_graph,
    #        color='deepskyblue',  path_effects=cartoon,
    #        zorder=20)
    #ax.fill_between(ec_graph, ec_graph + 3*std_graph, ec_graph - 3*std_graph,
    #                color='deepskyblue', alpha=0.2,
    #                zorder=-10)

    std_fid = std_collect(heat_fid)
    gamma_cut = (collect_fid < (heat_fid + 3*std_fid)) & (collect_fid > (heat_fid - 3*std_fid))
    #ax.plot(heat_fid[gamma_cut], collect_fid[gamma_cut],
    #        label='gamma band',
    #        ls='none', marker='.', alpha=0.7, markersize=10, color='steelblue')

    # neutron cut
    #ei_graph = np.interp(ec_graph, ec_array, ei_array)
    #ax.plot(ec_graph, ei_graph,
    #        color='coral',  path_effects=cartoon,
    #        zorder=20)
    #ax.fill_between(ec_graph,  ei_graph + 3*std_graph, ei_graph - 3*std_graph,
    #                color='coral', alpha=0.2,
    #                zorder=-10)

    collect_lindhard = np.interp(heat_fid, ec_array, ei_array)
    neutron_cut = (collect_fid < (collect_lindhard + 3*std_fid)) & (collect_fid > (collect_lindhard - 3*std_fid))
    #ax.plot(heat_fid[neutron_cut], collect_fid[neutron_cut],
    #        label='neutron band',
    #        ls='none', marker='.', alpha=0.7, markersize=10, color='coral')

    #ax.legend()

    # HO cut
    #ax.plot(ec_graph, np.zeros(ec_graph.shape),
    #        color='k',  path_effects=cartoon,
    #        zorder=20)
    #ax.fill_between(ec_graph,  3*std_graph, - 3*std_graph,
    #                color='k', alpha=0.2,
    #                zorder=-10)

    HO_cut = (collect_fid < (0 + 3*std_fid)) & (collect_fid > (0 - 3*std_fid))
    #ax.plot(heat_fid[HO_cut], collect_fid[HO_cut],
    #        label='HO band',
    #        ls='none', marker='.', alpha=0.7, markersize=10, color='k', zorder=-10)
    #
    #ax.legend()

    return gamma_cut, neutron_cut, HO_cut


for mode in mode_list:

    data = D[mode]['data']['fid']
    gamma_cut, neutron_cut, HO_cut = band_cut(data)
    D[mode]['data']['neutron'] = data[neutron_cut]
    D[mode]['data']['gamma'] = data[gamma_cut]

    D[mode]['simu']['neutron'] = dict()
    D[mode]['simu']['gamma'] = dict()
    D[mode]['simu']['input_neutron'] = dict()
    D[mode]['simu']['input_gamma'] = dict()
    for stype in simu_dict.keys():

        simu = D[mode]['simu']['fid'][stype]
        input_simu = D[mode]['simu']['input_fid'][stype]
        gamma_cut, neutron_cut, HO_cut = band_cut(simu)
        D[mode]['simu']['neutron'][stype] = simu[neutron_cut]
        D[mode]['simu']['gamma'][stype] = simu[gamma_cut]
        D[mode]['simu']['input_neutron'][stype] = input_simu[neutron_cut]
        D[mode]['simu']['input_gamma'][stype] = input_simu[gamma_cut]

mode = 'Calibration'
stype = 'NR'

data = D[mode]['data']['fid']
heat, A, B, C, Dd = data.T
collect = (B+Dd)/2
plt.figure('BAND CUT ' + mode + ' data')
plt.plot(heat, collect, ls='none', marker='.', label='FID')

data_nr = D[mode]['data']['neutron']
heat, A, B, C, Dd = data_nr.T
collect = (B+Dd)/2
plt.plot(heat, collect, ls='none', marker='.', label='NR')

plt.legend()
plt.grid()
plt.xlabel('Heat Energy [keV]')
plt.ylabel('Collect Energy [keV]')

data = D[mode]['simu']['fid'][stype]
heat, A, B, C, Dd = data.T
collect = (B+Dd)/2
plt.figure('BAND CUT ' + mode + ' simu')
plt.plot(heat, collect, ls='none', marker='.', label='FID')

data_nr = D[mode]['simu']['neutron'][stype]
heat, A, B, C, Dd = data_nr.T
collect = (B+Dd)/2
plt.plot(heat, collect, ls='none', marker='.', label='NR')

plt.legend()
plt.grid()
plt.xlabel('Heat Energy [keV]')
plt.ylabel('Collect Energy [keV]')

#%%
# =============================================================================
# FIDUCIAL VOLUME
# =============================================================================

from red_magic import Data_Selector

heat, ionA, ionB, ionC, ionD = D['Background']['data']['quality'].T
collect = (ionA+ionB+ionC+ionD)/2

fig, ax = plt.subplots(num='Fiducial volume Estimation')

line0, = ax.plot(
        heat,
        collect,
        ls='none', marker='.',
        )

count_fun = lambda inds: print(inds.shape) 

DS = Data_Selector(ax, line0, proceed_func=count_fun)


# =============================================================================
# QUENCHING PLOT WITH BACKGROUND AND CALIBRATION COMPARISON
# =============================================================================

er_calib = gen_er(D['Calibration']['data']['fid'])
q_calib = gen_qu(D['Calibration']['data']['fid'])
er_bkgd = gen_er(D['Background']['data']['fid'])
q_bkgd = gen_qu(D['Background']['data']['fid'])

fig, ax = plt.subplots(num='Quenchin plot comparison')

ax.plot(
        er_calib,
        q_calib,
        ls='none', marker='.',
        color='deepskyblue',
        label='Calibration',
        alpha=0.5,
)

ax.plot(
        er_bkgd,
        q_bkgd,
        ls='none', marker='.',
        color='k',
        label='Background',
        alpha=0.5,
)

ax.legend()
ax.set_xlabel('Recoil Energy [keV]')
ax.set_ylabel('Quenching Factor [fraction]')
ax.set_ylim(-0.25, 1.5)
ax.set_xlim(0, 30)
ax.grid(True)

#%%
# =============================================================================
# PULSE SIMULATION
# =============================================================================
mode = 'Background'
amode= 'fid'
stype = 'NR'

label_list = [
    'Flat ER',
    'Flat NR',
    'Line 1.3keV',
    'Line 10.37keV',
]

plt.figure()

for i, stype in enumerate(['ER', 'NR', '1kev', '10kev']):
    lab = label_list[i]
    data_nr = D[mode]['simu'][amode][stype]
    heat, A, B, C, Dd = data_nr.T
    collect = (B+Dd)/2
    plt.plot(heat, collect, ls='none', marker='.', markersize=2, label=lab)

plt.legend()
plt.grid()
plt.xlabel('Heat Energy [keV]')
plt.ylabel('Collect Energy [keV]')
plt.tight_layout()

#%%
# =============================================================================
# EFFICIENCY
# plt.close('all')
# =============================================================================

mode = 'Background'
stype = 'NR'
#bins = 50
bins = np.arange(0, 51, 1)
bins_width = bins[1] - bins[0]
bins_array = bins[:-1] + (bins_width) / 2
eff_x_array = bins_array

eff_dict = dict()

###
fig, axes = plt.subplots(nrows=2, ncols=2, num='Hist E input',
                         figsize=(10, 7), sharex='col', sharey='row')

for im, mode in enumerate(['Background', 'Calibration']):

    ax = axes[im]
    eff_dict[mode] = dict()

    for js, stype in enumerate(['ER', 'NR']):

        a = ax[js]

        n_all, _, __ = a.hist(D[mode]['simu']['input_all'][stype], bins=bins, label='all')

        n_trig,_, __ = a.hist(D[mode]['simu']['input_trigger'][stype], bins=bins, label='trigger')

        n_qual,_, __ = a.hist(D[mode]['simu']['input_quality'][stype], bins=bins, label='quality')

        n_fid,_, __ = a.hist(D[mode]['simu']['input_fid'][stype], bins=bins, label='fid')

        if stype == 'NR':
            band = 'input_neutron'
        elif stype == 'ER':
            band = 'input_gamma'
        n_band, _, __ = a.hist(D[mode]['simu'][band][stype], bins=bins, label='NR')

        a.legend(loc='upper right', title='{} SIMU {}'.format(mode, stype))
        a.grid()
        a.set_yscale('log')

        eff_local = dict()
        eff_local['trigger'] = n_trig / n_all
        eff_local['quality'] = n_qual / n_all
        eff_local['fid'] = n_fid / n_all
        eff_local['band'] = n_band / n_all
        eff_dict[mode][stype] = eff_local

axes[0,0].set_ylabel('Counts')
axes[1,0].set_ylabel('Counts')
axes[1,0].set_xlabel('Input Energy [keV]')
axes[1,1].set_xlabel('Input Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

###
fig, axes = plt.subplots(nrows=2, ncols=2, num='Efficiency as f(E input)',
                         figsize=(10, 7), sharex='col', sharey='row')

for im, mode in enumerate(['Background', 'Calibration']):

    ax = axes[im]

    for js, stype in enumerate(['ER', 'NR']):

        a = ax[js]

        for eff_key, eff_array in eff_dict[mode][stype].items():
            line, = a.plot(
                    bins_array, eff_array,
                    drawstyle='steps-mid', alpha=0.3,
            )
            a.errorbar(
                    bins_array, eff_array,
                    xerr = bins_width/2, yerr = eff_array*0.05,
                    label=eff_key,
                    ls='none', marker='.',
                    color=line.get_color()
            )

        a.legend(loc='upper right', title='{} SIMU {}'.format(mode, stype))
        a.grid()
        a.set_yscale('log')

axes[0,0].set_ylabel('Efficiency Fraction')
axes[1,0].set_ylabel('Efficiency Fraction')
axes[1,0].set_xlabel('Input Energy [keV]')
axes[1,1].set_xlabel('Input Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


#%%
# =============================================================================
# BACKGROUND
#plt.close('all')
# =============================================================================

mode = 'Background'
stype = 'NR'
bins = np.arange(0, 51, 0.5)
bins_width = bins[1] - bins[0]
bins_array = bins[:-1] + (bins_width) / 2

adv_eff_dict = dict()
for mode, eff_mode in eff_dict.items():
    adv_eff_dict[mode] = dict()

    for stype, eff_stype in eff_mode.items():
        adv_eff_dict[mode][stype] = dict()

        for ctype, eff_ctype in eff_stype.items():
            adv_eff_dict[mode][stype][ctype] = np.interp(
                    bins_array,
                    eff_x_array,
                    eff_ctype
            )


###
fig, axes = plt.subplots(nrows=2, ncols=2, num='Energy Spectrum with Cuts',
                         figsize=(10, 7), sharex='col', sharey='row')

adv_data_dict = dict()

for im, mode in enumerate(['Background', 'Calibration']):

    ax = axes[im]
    adv_data_dict[mode] = dict()

    for js, stype in enumerate(['ER', 'NR']):

        a = ax[js]

        for data_key, data_array in D[mode]['data'].items():

            if (data_key == 'neutron') and (stype == 'ER'):
                continue

            if (data_key == 'gamma') and (stype == 'NR'):
                continue

            n ,bins, patches = a.hist(
                    gen_er(data_array),
                    bins=bins,
                    label=data_key
            )

            if (data_key == 'gamma') or (data_key == 'neutron'):
                adv_data_dict[mode][stype] = n

        a.legend(loc='upper right', title='{} SIMU {}'.format(mode, stype))
        a.grid()
        a.set_yscale('log')

axes[0,0].set_ylabel('Counts')
axes[1,0].set_ylabel('Counts')
axes[1,0].set_xlabel('Recoil Energy [keV]')
axes[1,1].set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


### NORMALIZATION
### COUNTS TO DRU

mass_ge = 0.038 #kg

exposure_dict = {
        'Background':37.45/24, #days
        'Calibration':76./24, #days
}

DRU_dict = dict()
inte_dict = dict()

fig, axes = plt.subplots(nrows=2, ncols=2, num='Energy Spectrum in DRU',
                         figsize=(10, 8), sharex='col', sharey='row')

for im, mode in enumerate(['Background', 'Calibration']):

    ax = axes[im]
    DRU_dict[mode] = dict()

    for js, stype in enumerate(['ER', 'NR']):

        a = ax[js]

        data_bin_array = adv_data_dict[mode][stype]
        eff_array = adv_eff_dict[mode][stype]['band']
        exposure = exposure_dict[mode]

        DRU_dict[mode][stype] = data_bin_array / (eff_array * exposure * bins_width * mass_ge)

        a.plot(
                bins_array,
                DRU_dict[mode][stype],
                label='{} {} [DRU]'.format(mode, stype),
                drawstyle='steps-mid',
        )

        if stype == 'NR':
            cut_2kev = (bins_array >= 2)

            inte = np.trapz(
                    DRU_dict[mode][stype][cut_2kev],
                    bins_array[cut_2kev]
            )

            inte_dict[mode] = inte

            a.fill_between(
                    bins_array[cut_2kev],
                    DRU_dict[mode][stype][cut_2kev],
                    label='In [2keV-50keV]:\n{:.1f} Counts/kg/days'.format(inte),
                    step='mid',
                    color='coral',

            )

        a.legend(loc='upper right', title='{} SIMU {}'.format(mode, stype))
        a.grid()
        a.set_yscale('log')

axes[0,0].set_ylabel('DRU [Efficiency corrected Counts/keV/kg/days]')
axes[1,0].set_ylabel('DRU [Efficiency corrected Counts/keV/kg/days]')
axes[1,0].set_xlabel('Recoil Energy [keV]')
axes[1,1].set_xlabel('Recoil Energy [keV]')


for ax in axes:
    for a in ax:
        a.set_xscale('log')

inte_bkgd = inte_dict['Background']
inte_calib = inte_dict['Calibration']
ratio = inte_calib / inte_bkgd

fig.suptitle(
        (
                'In ROI [2keV-50keV]: {:.1f} Calib / {:.1f} Bkgd = {:.2f} Ratio'
        ).format(inte_calib, inte_bkgd, ratio)
)

fig.tight_layout(rect=(0, 0, 1, 0.950))
fig.subplots_adjust(hspace=0, wspace=0)


#%%
### Money plot
calib_er = DRU_dict['Calibration']['ER']
calib_nr = DRU_dict['Calibration']['NR']
bkgd_er = DRU_dict['Background']['ER']
bkgd_nr = DRU_dict['Background']['NR']

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
    'deepskyblue',
    ]

legend_list =[
    'Calibration ER band',
    'Calibration NR band',
    'Background ER band',
    'Background NR band',
]

fig, ax = plt.subplots()

for i,dru_array in enumerate(array_list):

    c = color_list[i]
    leg = legend_list[i]
    
    zorder=1
    
    if i in (0,2):
        zorder=5
    
    ax.plot(
        bins_array,
        dru_array,
        drawstyle='steps-mid',
        alpha=1,
        color=c,
        lw=3,
        zorder=zorder,
        #path_effects=cartoon,
        label=leg
    )

    ax.plot(
        bins_array,
        dru_array,
        ls='none',
        alpha=1,
        marker='.',
        color=c,
        zorder=zorder,
        #path_effects=cartoon,
    )

    if i in (1,3):
        
        if i == 1:
            msg = (
                'Calibration events in [2keV-50keV]:\n{:.2e} Counts/kg/days'
            ).format(inte_dict['Calibration'])
        if i == 3:
            msg = (
                'Background events in [2keV-50keV]:\n{:.2e} Counts/kg/days'
            ).format(inte_dict['Background'])
            
        ax.fill_between(
            bins_array,
            dru_array,
            step='mid',
            color=lighten_color(c),
            zorder=-1,
            label=msg
        )

ax.axvspan(0, 2, color='k', alpha=0.3, zorder=5)
ax.axvline(2, ls='--', color='k', zorder=5, 
           label='Analysis Threshold: 2keV')



ax.legend(handler_map={str: LegendTitle()})

ax.set_xlim(0.25, 50)
ax.set_ylim(1e2, 1e7)
ax.set_yscale('log')
ax.set_ylabel('DRU [Efficiency corrected Counts/keV/kg/days]')
ax.set_xlabel('Recoil Energy [keV]')

ax.grid(which='both', alpha=0.5)
fig.tight_layout()










