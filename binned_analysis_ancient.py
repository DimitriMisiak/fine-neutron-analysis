#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:15:05 2020

@author: misiak
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plot_addon import lighten_color, LegendTitle

analysis_dir = '/home/misiak/Analysis/neutron_background'
analysis_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])
analysis_data_path = '/'.join([analysis_dir, 'data_science.h5'])

df_simu = pd.read_hdf(
    analysis_simu_path,
    key='df',
    where='glitch_time_cut = True'
)

df_data = pd.read_hdf(
    analysis_data_path,
    key='df',
    where='glitch_time_cut = True'
)

#%%
plt.close('all')

# bins = np.arange(0, 50.5, 0.5)
# bins_width = bins[1] - bins[0]
# bins_array = bins[:-1] + (bins_width) / 2

bins = np.logspace(np.log10(0.2), np.log10(50), 100)
bins_width = (bins[1:] - bins[:-1])
bins_array = bins[:-1]

bins = np.logspace(np.log(0.2), np.log(50), 100, base=np.exp(1))
bins_width = (bins[1:] - bins[:-1])
bins_array = bins[:-1]


# =============================================================================
# CONTAMINATION calculation
# =============================================================================
energy_column = 'recoil_energy_bulk'

source_list = ['Calibration', 'Background']
simulation_list = ['flat_ER', 'line_1keV', 'line_10keV', 'flat_NR']

fine_cut = (
    df_simu['trigger_cut']
    & df_simu['quality_cut']
    & df_simu['bulk_cut']
    & df_simu['charge_conservation_cut']
)

df_fine = df_simu[fine_cut]

fine_data_cut = (
    df_data['quality_cut']
    & df_data['bulk_cut']
    & df_data['charge_conservation_cut']
)

df_fine_data = df_data[fine_data_cut]

def event_population_cuts(df):
    gamma_cut = df['gamma_cut']
    neutron_cut = df['neutron_cut']
    ho_cut = df['HO_cut']
    all_cut = pd.Series(True, index=df.index)
    
    other_cut = ~(gamma_cut | neutron_cut | ho_cut)
    pure_gamma_cut = gamma_cut & ~(neutron_cut | ho_cut)
    pure_neutron_cut = neutron_cut & ~(gamma_cut | ho_cut)
    pure_ho_cut = ho_cut & ~(gamma_cut | neutron_cut)
    
    mix_gamma_neutron_cut = gamma_cut & neutron_cut & ~ho_cut
    mix_gamma_ho_cut = gamma_cut & ho_cut & ~neutron_cut
    mix_neutron_ho_cut = neutron_cut & ho_cut & ~gamma_cut
    
    mix_gamma_neutron_ho_cut = gamma_cut & neutron_cut & ho_cut
    
    pop_cut_dict = {
        'all': all_cut,
        'others': other_cut,
        'pure gamma': pure_gamma_cut,
        'pure neutron': pure_neutron_cut,
        'pure heat-only': pure_ho_cut,
        'mix gamma-neutron': mix_gamma_neutron_cut,
        'mix gamma-ho': mix_gamma_ho_cut,
        'mix neutron-ho': mix_neutron_ho_cut,
        'mix gamma-neutron-ho': mix_gamma_neutron_ho_cut
    }

    return pop_cut_dict


pop_cut_dict = event_population_cuts(df_fine)
pop_data_cut_dict = event_population_cuts(df_fine_data)

pop_dict = dict()
pop_data_dict = dict()
for source in source_list:
    
    pop_dict[source] = dict()
    pop_data_dict[source] = dict()
    
     # DATA
    all_data_cut = (df_fine_data['source'] == source)
    for key, cut in pop_data_cut_dict.items():
        num = np.histogram(
            df_fine_data[all_data_cut & cut][energy_column],
            bins=bins
        )[0]
    
        pop_data_dict[source][key] = num   
    
    for simulation in simulation_list:
        
        # SIMU
        all_cut = (
            (df_fine['source'] == source)
            & (df_fine['simulation'] == simulation)
        )

        pop_dict[source][simulation] = dict()
        for key, cut in pop_cut_dict.items():
            num = np.histogram(
                df_fine[all_cut & cut][energy_column],
                bins=bins
            )[0]
            
            pop_dict[source][simulation][key] = num
            

#%%
### PLOT number of events passing the cuts
fig, axes = plt.subplots(nrows=4, ncols=2, num='SIMU Stacked hist events population',
                          figsize=(10, 7), sharex='col', sharey='row')

###
for im, simulation in enumerate(simulation_list):

    ax = axes[im]
    for js, source in enumerate(source_list):

            a = ax[js]

            bot = bins_array * 0
            for key, num in pop_dict[source][simulation].items():
                
                if key=='all':
                    a.plot(
                        bins_array,
                        num,
                        color='k',
                        drawstyle='steps-mid',
                        zorder=10,
                        label=key
                    )
                    continue
                a.bar(
                    bins_array,
                    num,
                    width=bins_width,
                    bottom=bot,
                    label=key
                )
                bot += num

            msg = '{} {} Events'.format(source, simulation).replace('_', ' ')
            a.text(
                0.5, 0.1,
                msg,
                horizontalalignment='center',
                verticalalignment='center',
                transform=a.transAxes
            )
            a.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 1.05)
)

for ax in axes[:, 0]:
    ax.set_ylabel('Counts')
for ax in axes[-1, :]:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

### ok seems very good
### the difference between the calibration and the background
### should come from the difference in exposure.


### PLOT DATA population
fig, axes = plt.subplots(ncols=2, nrows=2, num='DATA Stacked hist events population',
                          figsize=(10, 7), sharex='col', sharey='row')

###
for js, source in enumerate(source_list):

        a = axes[0][js]
        a_delta = axes[1][js]
        
        bot = bins_array * 0
        for key, num in pop_data_dict[source].items():
            
            if key=='all':
                num_all = num
                a.plot(
                    bins_array,
                    num,
                    color='k',
                    drawstyle='steps-mid',
                    zorder=10,
                    label=key
                )
                continue
            a.bar(
                bins_array,
                num,
                width=bins_width,
                bottom=bot,
                label=key
            )
            bot += num

        a_delta.plot(
            bins_array,
            num_all - bot,
            color='k',
            drawstyle='steps-mid',
            zorder=10
        )

        msg = '{} Data Events'.format(source).replace('_', ' ')
        a.text(
            0.5, 0.1,
            msg,
            horizontalalignment='center',
            verticalalignment='center',
            transform=a.transAxes
        )
        a.grid()
        a_delta.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5)
)
a.set_yscale('log')

axes[0][0].set_ylabel('Counts')
axes[1][0].set_ylabel('Delta "all"')
for ax in axes[-1,:]:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


#%%
# =============================================================================
# PURE GAMMA FITTING
# =============================================================================
from scipy.optimize import curve_fit

fig, axes = plt.subplots(
    ncols=2,
    num='Pure gamma fitting',
    figsize=(10, 7)
)

popt_dict = dict()
for i, source in enumerate(source_list):

    spectrum_data = pop_data_dict[source]['pure gamma']
    
    spectrum_simu_flat = pop_dict[source]['flat_ER']['pure gamma']
    spectrum_simu_1kev = pop_dict[source]['line_1keV']['pure gamma']
    spectrum_simu_10kev = pop_dict[source]['line_10keV']['pure gamma']

    def pure_gamma_model(x, a,b,c):
        return a*spectrum_simu_flat + b*spectrum_simu_1kev + c*spectrum_simu_10kev

    p0 = [5/200, 900/3500, 3000/8000]

    popt, pcov = curve_fit(
        pure_gamma_model,
        bins_array,
        spectrum_data,
        p0=p0,
        sigma=spectrum_data*0.01+3
    )

    
    spectrum_mod0 = pure_gamma_model(bins_array, *popt)
    popt_dict[source] = popt
    
    ax = axes[i]

    ax.plot(
            bins_array,
            spectrum_data,
            label='data',
            drawstyle='steps-mid'
    )
    
    ax.plot(
            bins_array,
            spectrum_mod0,
            label='model {}'.format(popt),
            drawstyle='steps-mid'
    )

    ax.grid()
    ax.legend()
    ax.set_xlabel('Recoil Energy [keV]')

axes[0].set_ylabel('Pure Gamma Counts')
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


### ER Substraction
gamma_corr_dict = dict()
for source, popt in popt_dict.items():
    
    a,b,c = popt
    
    pure_gamma_corr = list()
    others_corr = list()
    gamma_corr_dict[source] = dict()
    for key in pop_data_dict[source].keys():
        
        data_hist = pop_data_dict[source][key]
        flat_hist = pop_dict[source]['flat_ER'][key]
        line_1kev_hist = pop_dict[source]['line_1keV'][key]
        line_10kev_hist = pop_dict[source]['line_10keV'][key]     
        
        
        if key in ['all', 'others', 'pure gamma']:
            corr_hist = 0 * bins_array
            
        if key in [
                'pure neutron',
                'pure heat-only',
                'mix gamma-neutron',
                'mix gamma-ho',
                'mix neutron-ho',
                'mix gamma-neutron-ho'
            ]:
            
            gamma_corr = (
                a*flat_hist
                + b*line_1kev_hist
                + c*line_10kev_hist
            )
            corr_hist = - gamma_corr
            
            pure_gamma_corr.append(gamma_corr)
            
        gamma_corr_dict[source][key] = corr_hist

    for corr in pure_gamma_corr:
        gamma_corr_dict[source]['pure gamma'] += corr
    for corr in others_corr:
        gamma_corr_dict[source]['others'] += corr

### PLOT GAMMA CORRECTION
fig, axes = plt.subplots(
    ncols=2,
    num='GAMMA CORRECTION',
    figsize=(20, 7),
    sharex='col',
    sharey='row'
)

for js, source in enumerate(source_list):

        a = axes[js]

        all_num = 0 * bins_array
        for key, num in gamma_corr_dict[source].items():
            
            if key=='all':
                continue
            a.plot(
                bins_array,
                num,
                drawstyle='steps-mid',
                label=key,
            )
            all_num += num
            
        a.plot(
            bins_array,
            all_num,
            color='k',
            drawstyle='steps-mid',
            zorder=10,
            label='total'
        )
        
        msg = '{} Data Events'.format(source).replace('_', ' ')
        a.text(
            0.5, 0.1,
            msg,
            horizontalalignment='center',
            verticalalignment='center',
            transform=a.transAxes
        )
        a.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5)
)

axes[0].set_ylabel('Counts')
for ax in axes:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

#%%
## APPLYING GAMMA CORRECTION
corr_data_dict = dict()
for source in source_list:
    corr_data_dict[source] = dict()
    for key, num in pop_data_dict[source].items():
        corr_data_dict[source][key] = num + gamma_corr_dict[source][key]
    
    
### PLOT DATA population
fig, axes = plt.subplots(ncols=2, nrows=2, num='CORR GAMMA DATA Stacked hist events population',
                          figsize=(20, 7), sharex='col', sharey='row')

###
for js, source in enumerate(source_list):

        a = axes[0][js]
        a_delta = axes[1][js]
        
        bot = bins_array * 0
        for key, num in corr_data_dict[source].items():
            
            if key=='all':
                num_all = num
                a.plot(
                    bins_array,
                    num,
                    color='k',
                    drawstyle='steps-mid',
                    zorder=10,
                    label=key
                )
                continue
            a.bar(
                bins_array,
                num,
                width=bins_width,
                bottom=bot,
                label=key
            )
            bot += num

        a_delta.plot(
            bins_array,
            num_all - bot,
            color='k',
            drawstyle='steps-mid',
        )

        msg = '{} Data Events'.format(source).replace('_', ' ')
        a.text(
            0.5, 0.1,
            msg,
            horizontalalignment='center',
            verticalalignment='center',
            transform=a.transAxes
        )
        a.grid()
        a_delta.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5)
)

axes[0][0].set_ylabel('Counts')
axes[1][0].set_ylabel('Delta "all"')
for ax in axes[1,:]:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


#%%
#=============================================================================
# PURE NEUTRON FITTING
# =============================================================================
from scipy.optimize import curve_fit


fig, axes = plt.subplots(
    ncols=2,
    num='Pure neutron fitting',
    figsize=(10, 7)
)

neutron_popt_dict = dict()
for i, source in enumerate(source_list):

    spectrum_data = corr_data_dict[source]['pure neutron']
    
    spectrum_simu_flat = pop_dict[source]['flat_NR']['pure neutron']

    def pure_gamma_model(x, a,b):
        return a*spectrum_simu_flat*np.exp(-b*x)

    p0 = [1.6, 0.08]
    # spectrum_mod0 = pure_gamma_model(bins_array, *p0)

    popt, pcov = curve_fit(
        pure_gamma_model,
        bins_array,
        spectrum_data,
        p0=p0,
        sigma=spectrum_data*0.1 + 1
    )
    spectrum_mod0 = pure_gamma_model(bins_array, *popt)
    neutron_popt_dict[source] = popt
    
    ax = axes[i]

    ax.plot(
            bins_array,
            spectrum_data,
            label='data',
            drawstyle='steps-mid'
    )
    
    ax.plot(
            bins_array,
            spectrum_mod0,
            label='model {}'.format(popt),
            drawstyle='steps-mid'
    )

    ax.grid()
    ax.legend()
    ax.set_xlabel('Recoil Energy [keV]')
    ax.set_yscale('log')
    
axes[0].set_ylabel('Pure Gamma Counts')
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

### NR Substraction
neutron_corr_dict = dict()
for source, popt in neutron_popt_dict.items():
    
    a,b = popt
    
    pure_neutron_corr = list()
    others_corr = list()
    neutron_corr_dict[source] = dict()
    for key in corr_data_dict[source].keys():
        
        data_hist = corr_data_dict[source][key]
        flat_hist = pop_dict[source]['flat_NR'][key]
        
        if key in ['all', 'others', 'pure neutron', 'pure gamma']:
            corr_hist = 0 * bins_array
            
        if key in [
                'pure heat-only',
                'mix gamma-neutron',
                'mix gamma-ho',
                'mix neutron-ho',
                'mix gamma-neutron-ho'
            ]:
            
            neutron_corr = (
                a*flat_hist*np.exp(-b*bins_array)
            )
            corr_hist = - neutron_corr
            
            pure_neutron_corr.append(neutron_corr)
            
        neutron_corr_dict[source][key] = corr_hist

    for corr in pure_neutron_corr:
        neutron_corr_dict[source]['pure neutron'] += corr
    for corr in others_corr:
        neutron_corr_dict[source]['others'] += corr

### PLOT NEUTRON CORRECTION
fig, axes = plt.subplots(
    ncols=2,
    num='NEUTRON CORRECTION',
    figsize=(20, 7),
    sharex='col',
    sharey='row'
)

for js, source in enumerate(source_list):

        a = axes[js]

        all_num = 0 * bins_array
        for key, num in neutron_corr_dict[source].items():
            
            if key=='all':
                continue
            a.plot(
                bins_array,
                num,
                drawstyle='steps-mid',
                label=key,
            )
            all_num += num
            
        a.plot(
            bins_array,
            all_num,
            color='k',
            drawstyle='steps-mid',
            zorder=10,
            label='total'
        )
        
        msg = '{} Data Events'.format(source).replace('_', ' ')
        a.text(
            0.5, 0.1,
            msg,
            horizontalalignment='center',
            verticalalignment='center',
            transform=a.transAxes
        )
        a.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5)
)

axes[0].set_ylabel('Counts')
for ax in axes:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

#%%
## APPLYING NEUTRON CORRECTION
neutron_corr_data_dict = dict()
for source in source_list:
    neutron_corr_data_dict[source] = dict()
    for key, num in corr_data_dict[source].items():
        neutron_corr_data_dict[source][key] = num + neutron_corr_dict[source][key]
    
    
### PLOT DATA population
fig, axes = plt.subplots(ncols=2, nrows=2, num='CORR NEUTRON DATA Stacked hist events population',
                          figsize=(20, 7), sharex='col', sharey='row')

###
for js, source in enumerate(source_list):

        a = axes[0][js]
        a_delta = axes[1][js]

        bot = bins_array * 0
        for key, num in neutron_corr_data_dict[source].items():
            
            if key=='all':
                num_all = num
                a.plot(
                    bins_array,
                    num,
                    color='k',
                    drawstyle='steps-mid',
                    zorder=10,
                    label=key
                )
                continue
            a.bar(
                bins_array,
                num,
                width=bins_width,
                bottom=bot,
                label=key
            )
            bot += num

        a_delta.plot(
            bins_array,
            num_all - bot,
            color='k',
            drawstyle='steps-mid',
        )

        msg = '{} Data Events'.format(source).replace('_', ' ')
        a.text(
            0.5, 0.1,
            msg,
            horizontalalignment='center',
            verticalalignment='center',
            transform=a.transAxes
        )
        a.grid()
        a_delta.grid()

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5)
)

axes[0][0].set_ylabel('Counts')
axes[1][0].set_ylabel('Delta "all"')
for ax in axes[-1,:]:
    ax.set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

#%%
### GATHERING THE GAMMA, NEUTRON, HO events
adv_data_dict = dict()
for source in source_list:
    
    adv_data_dict[source] = dict()
    
    neutron_corr = neutron_corr_data_dict[source]
    
    # adv_data_dict[source]['ER'] = np.sum(
    #     [num for key,num in neutron_corr.items() if 'gamma' in key],
    #     axis=0
    # )

    # ### HACK
    # adv_data_dict[source]['ER'] = np.sum(
    #     [num for key,num in neutron_corr.items() if ('gamma' in key) and ~('ho' in key)],
    #     axis=0
    # )


    # adv_data_dict[source]['NR'] = np.sum(
    #     [num for key,num in neutron_corr.items() if 'neutron' in key],
    #     axis=0
    # )   

    ### HACK
    adv_data_dict[source]['ER'] = neutron_corr['pure gamma']


    adv_data_dict[source]['NR'] = neutron_corr['pure neutron']

#%%
# =============================================================================
# EFFICIENCY calculation
# =============================================================================
energy_column = 'input_energy'

source_list = ['Calibration', 'Background']
simulation_list = ['ER', 'NR']

num_dict = dict()
eff_dict = dict()
for source in source_list:
    
    num_dict[source] = dict()
    eff_dict[source] = dict()
    for sim in simulation_list:
        
        simulation = 'flat_'+sim
        
        all_cut = (
            (df_simu['source'] == source)
            & (df_simu['simulation'] == simulation)
        )
        trigger_cut = df_simu['trigger_cut'] & all_cut
        quality_cut = df_simu['quality_cut'] & trigger_cut
        charge_cut = df_simu['charge_conservation_cut'] & quality_cut
        bulk_cut = df_simu['bulk_cut'] & charge_cut
        
        no_other = (
            df_simu['gamma_cut']
            | df_simu['neutron_cut']
            | df_simu['HO_cut']
        )
        # if sim == 'ER':
        #     no_other = (
        #         df_simu['gamma_cut']
        #     )
        # elif sim == 'NR':
        #     no_other = (
        #         df_simu['neutron_cut']
        band_cut = bulk_cut & no_other 
        
        cut_dict = {
            'all': all_cut,
            'trigger': trigger_cut,
            'quality': quality_cut,
            'charge': charge_cut,
            'bulk': bulk_cut,
            'band': band_cut,
        }
        
        num_dict[source][sim] = dict()
        eff_dict[source][sim] = dict()
        for key, cut in cut_dict.items():
            num = np.histogram(
                df_simu[cut][energy_column],
                bins=bins
            )[0]
            num_dict[source][sim][key] = num

            eff_dict[source][sim][key] = (
                num / num_dict[source][sim]['all']
            )

        
### PLOT number of events passing the cuts
fig, axes = plt.subplots(nrows=2, ncols=2, num='Histogramm events passing cuts',
                          figsize=(10, 7), sharex='col', sharey='row')

for im, source in enumerate(source_list):

    ax = axes[im]

    for js, simulation in enumerate(simulation_list):

            a = ax[js]
            
            for key, num in num_dict[source][simulation].items():
            
                line, = a.plot(
                    bins_array,
                    num,
                    drawstyle='steps-mid',
                    marker='.',
                    label=key
                )
                a.fill_between(
                    bins_array,
                    num,
                    color=lighten_color(line.get_color()),
                    step='mid',
                )
                
            msg = '{} {} Events'.format(source, simulation).replace('_', ' ')
            a.text(
                0.5, 0.1,
                msg,
                horizontalalignment='center',
                verticalalignment='center',
                transform=a.transAxes
            )
            a.grid()
            a.set_yscale('log')

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 1.05)
)
axes[0,0].set_ylabel('Counts')
axes[1,0].set_ylabel('Counts')
axes[1,0].set_xlabel('Recoil Energy [keV]')
axes[1,1].set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


### PLOT number of events passing the cuts
fig, axes = plt.subplots(nrows=2, ncols=2, num='Histogramm cut efficiency',
                          figsize=(10, 7), sharex='col', sharey='row')

for im, source in enumerate(source_list):

    ax = axes[im]

    for js, simulation in enumerate(simulation_list):

            a = ax[js]
            
            for key, eff in eff_dict[source][simulation].items():
            
                line, = a.plot(
                    bins_array,
                    eff,
                    drawstyle='steps-mid',
                    marker='.',
                    label=key
                )
                a.fill_between(
                    bins_array,
                    eff,
                    color=lighten_color(line.get_color()),
                    step='mid',
                )
                
            msg = '{} {} Efficiency'.format(source, simulation).replace('_', ' ')
            a.text(
                0.5, 0.1,
                msg,
                horizontalalignment='center',
                verticalalignment='center',
                transform=a.transAxes
            )
            a.grid()
            a.set_yscale('log')

a.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 1.05)
)
axes[0,0].set_ylabel('Efficiency')
axes[1,0].set_ylabel('Efficiency')
axes[1,0].set_xlabel('Recoil Energy [keV]')
axes[1,1].set_xlabel('Recoil Energy [keV]')

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)



# =============================================================================
# ### NORMALIZATION, COUTNS TO DRU
# =============================================================================
#%%
mass_ge = 0.038 #kg

from pipeline_data_quality import quality_parameters
from pipeline_data_raw import stream_configuration

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
    

    exposure_dict[source] = exposure / 24 # in hours

DRU_dict = dict()
inte_dict = dict()
for mode in ['Background', 'Calibration']:

    DRU_dict[mode] = dict()

    for stype in ['ER', 'NR']:


        data_bin_array = adv_data_dict[mode][stype]
        eff_array = eff_dict[mode][stype]['band']
        exposure = exposure_dict[mode]

        DRU_dict[mode][stype] = data_bin_array / (eff_array * exposure * bins_width * mass_ge)

        if stype == 'NR':
            cut_2kev = (bins_array >= 2)

            # inte = np.trapz(
            #         DRU_dict[mode][stype][cut_2kev],
            #         bins_array[cut_2kev]
            # )

            inte = np.sum(
                DRU_dict[mode][stype][cut_2kev] * bins_width[cut_2kev]
            )

            inte_dict[mode] = inte



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

# ax.axvspan(0, 2, color='k', alpha=0.3, zorder=5)
# ax.axvline(2, ls='--', color='k', zorder=5, 
#             label='Analysis Threshold: 2keV')

fig.suptitle(
        (
                'In ROI [2keV-50keV]: {:.1f} Calib / {:.1f} Bkgd = {:.2f} Ratio'
        ).format(inte_calib, inte_bkgd, ratio)
)



ax.legend(handler_map={str: LegendTitle()})

ax.set_xlim(0.25, 50)
ax.set_ylim(1e2, 1e7)
ax.set_yscale('log')
ax.set_ylabel('DRU [Efficiency corrected Counts/keV/kg/days]')
ax.set_xlabel('Recoil Energy [keV]')

ax.grid(which='both', alpha=0.5)
fig.tight_layout()


#%%
fig, ax = plt.subplots()

ax.plot(
        df_fine_data[df_fine_data.source == 'Background']['recoil_energy_bulk'],
        df_fine_data[df_fine_data.source == 'Background']['quenching_bulk'],
        ls='none',
        marker='.',
        alpha=0.3,
        color='deepskyblue'
)
   
ax.plot(
        df_fine_data[df_fine_data.source == 'Calibration']['recoil_energy_bulk'],
        df_fine_data[df_fine_data.source == 'Calibration']['quenching_bulk'],
        ls='none',
        marker='.',
        alpha=0.3,
        color='k'
)
