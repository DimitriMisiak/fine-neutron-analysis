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

df_simu_science = pd.read_hdf(
    analysis_simu_path,
    key='df',
    where='glitch_time_cut = True'
)

df_data_science = pd.read_hdf(
    analysis_data_path,
    key='df',
    where='glitch_time_cut = True'
)

#%%
plt.close('all')

bins = np.arange(0, 50.5, 0.5)
bins_width = (bins[1:] - bins[:-1])
bins_array = bins[:-1] + (bins_width) / 2
ax_scale = 'linear'

# bins = np.logspace(np.log10(0.2), np.log10(50), 100)
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]
# ax_scale = ax_scale

# bins = np.logspace(np.log(0.2), np.log(50), 100, base=np.exp(1))
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1]
# ax_scale = ax_scale

# =============================================================================
# GENERAL VARIABLES
# =============================================================================
energy_column = 'recoil_energy_bulk'

source_list = ['Calibration', 'Background']
simulation_list = ['flat_ER', 'line_1keV', 'line_10keV', 'flat_NR']


def event_population_cuts(df):
    """
    Return a dictionnary of truth arrays. Each truth array corresponds
    a population cut according to the gamma, neutron and ho band.
    """
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
        'pure_gamma': pure_gamma_cut,
        'pure_neutron': pure_neutron_cut,
        'pure_ho': pure_ho_cut,
        'mix_gamma_neutron': mix_gamma_neutron_cut,
        'mix_gamma_ho': mix_gamma_ho_cut,
        'mix_neutron_ho': mix_neutron_ho_cut,
        'mix_gamma_neutron_ho': mix_gamma_neutron_ho_cut
    }

    for k,v in pop_cut_dict.items():
        df[k] = v

    return None

event_population_cuts(df_simu_science)
event_population_cuts(df_data_science)

fine_simu_cut = (
    df_simu_science['trigger_cut']
    & df_simu_science['quality_cut']
    & df_simu_science['bulk_cut']
    & df_simu_science['charge_conservation_cut']
)

df_fine_simu = df_simu_science[fine_simu_cut]

fine_data_cut = (
    df_data_science['quality_cut']
    & df_data_science['bulk_cut']
    & df_data_science['charge_conservation_cut']
)

df_fine_data = df_data_science[fine_data_cut]

#%%
# =============================================================================
# FIRST VIZUALIZATION
# =============================================================================

fig, axes = plt.subplots(
    num='First vizualization',
    ncols=2, nrows=5, figsize=(10,7)
)

for ax_col, source in zip(axes.T, source_list):
    
    df_data = df_fine_data[ df_fine_data["source"] == source ]
    df_simu = df_fine_simu[ df_fine_simu["source"] == source ]
    
    ax_col[0].hist(
        df_data[energy_column],
        bins=bins
    )
    ax_col[0].set_ylabel('Data')
    
    for ax, simulation in zip(ax_col[1:], simulation_list):
        
        df_simu_local = df_simu[ df_simu["simulation"] == simulation ]
        
        ax.hist(
            df_simu_local[energy_column],
            bins=bins
        )
        ax.set_ylabel(simulation)
            
    for ax in ax_col:
        ax.grid()
        ax.set_yscale(ax_scale)
        ax.set_xscale(ax_scale)


fig.text(
    0.25, 0.98, 
    source_list[0],
     horizontalalignment='center',
     verticalalignment='center',
     bbox=dict(facecolor='lime', alpha=0.5)
)

fig.text(
    0.75, 0.98, 
    source_list[1],
     horizontalalignment='center',
     verticalalignment='center',
     bbox=dict(facecolor='lime', alpha=0.5)
)

fig.tight_layout(rect=(0,0, 1, 0.98))     
fig.subplots_adjust(hspace=.0)

# =============================================================================
# VIZUALIZATION WITH PURE GAMMA POPULATION
# =============================================================================

fig, axes = plt.subplots(
    num='Pure gamma population',
    ncols=2, nrows=5, figsize=(10,7)
)

for ax_col, source in zip(axes.T, source_list):
    
    data_cut = (
        (df_fine_data["source"] == source)
        & df_fine_data["pure_gamma"]
        & ~df_fine_data["HO_cut"]
    )
    
    simu_cut = (
        (df_fine_simu["source"] == source)
        & df_fine_simu["pure_gamma"]
        & ~df_fine_data["HO_cut"]
    )
    
    df_data = df_fine_data[data_cut]
    df_simu = df_fine_simu[simu_cut]
    
    ax_col[0].hist(
        df_data[energy_column],
        bins=bins
    )
    ax_col[0].set_ylabel('Data')
    
    for ax, simulation in zip(ax_col[1:], simulation_list):
        
        df_simu_local = df_simu[ df_simu["simulation"] == simulation ]
        
        ax.hist(
            df_simu_local[energy_column],
            bins=bins
        )
        ax.set_ylabel(simulation)
            
    for ax in ax_col:
        ax.grid()
        ax.set_yscale(ax_scale)
        ax.set_xscale(ax_scale)


fig.text(
    0.25, 0.98, 
    source_list[0],
     horizontalalignment='center',
     verticalalignment='center',
     bbox=dict(facecolor='lime', alpha=0.5)
)

fig.text(
    0.75, 0.98, 
    source_list[1],
     horizontalalignment='center',
     verticalalignment='center',
     bbox=dict(facecolor='lime', alpha=0.5)
)

fig.tight_layout(rect=(0,0, 1, 0.98))     
fig.subplots_adjust(hspace=.0)

#%%
# =============================================================================
# EFFICIENCY calculation
# =============================================================================

fig, ax = plt.subplots()

local_cut = ( df_simu_science['simulation'] == 'flat_ER' )
fine_local_cut = ( df_fine_simu['simulation'] == 'flat_ER' )

df_simu_local = df_simu_science[local_cut]
df_fine_simu_local = df_fine_simu[fine_local_cut]

ax.hist(
        df_simu_local[energy_column],
        bins=bins,
        color='b'
)

ax.hist(
        df_fine_simu_local[energy_column],
        bins=bins,
        color='coral'
)

ax.set_yscale(ax_scale)
ax.set_xscale(ax_scale)

binned_efficiency_dict = dict()

recoil_type_list = ['ER', 'NR']

for source in source_list:
    
    binned_efficiency_dict[source] = dict()
    for recoil_type in recoil_type_list:
        
        simulation = 'flat_' + recoil_type
        
        local_cut = ( df_simu_science['simulation'] == simulation )
        fine_local_cut = ( df_fine_simu['simulation'] == simulation )
        
        df_simu_local = df_simu_science[local_cut]
        df_fine_simu_local = df_fine_simu[fine_local_cut]

        simu_hist,_ = np.histogram(
            df_simu_local['input_energy'],
            bins=bins
        )
        
        fine_simu_hist,_ = np.histogram(
            df_fine_simu_local['input_energy'],
            bins=bins
        )

        binned_efficiency_dict[source][recoil_type] = (
            fine_simu_hist / simu_hist
        )
        
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
                    ls='steps-mid',
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
            a.set_yscale(ax_scale)

# a.legend(
#     loc='center left',
#     bbox_to_anchor=(1.05, 1.05)
# )
# axes[0,0].set_ylabel('Counts')
# axes[1,0].set_ylabel('Counts')
# axes[1,0].set_xlabel('Recoil Energy [keV]')
# axes[1,1].set_xlabel('Recoil Energy [keV]')

# fig.tight_layout()
# fig.subplots_adjust(hspace=0, wspace=0)


### PLOT number of events passing the cuts
fig, axes = plt.subplots(nrows=2, ncols=2, num='Histogramm cut efficiency',
                          figsize=(10, 7), sharex='col', sharey='row')

for im, source in enumerate(source_list):

    ax = axes[im]

    for js, recoil_type in enumerate(recoil_type_list):

            a = ax[js]

            eff = binned_efficiency_dict[source][recoil_type]
                
            line, = a.plot(
                bins_array,
                eff,
                drawstyle='steps-mid',
                marker='.',
                label='fine'
            )
            a.fill_between(
                bins_array,
                eff,
                color=lighten_color(line.get_color()),
                step='mid',
            )
                
            msg = '{} {} Efficiency'.format(source, recoil_type).replace('_', ' ')
            a.text(
                0.5, 0.1,
                msg,
                horizontalalignment='center',
                verticalalignment='center',
                transform=a.transAxes
            )
            a.grid()
            a.set_yscale(ax_scale)
            a.set_xscale(ax_scale)

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
# CONTAMINATION calculation
# =============================================================================
pop_cut_dict_simu = event_population_cuts(df_fine_simu)
pop_cut_dict_data = event_population_cuts(df_fine_data)

pop_dict_simu = dict()
pop_dict_data = dict()
for source in source_list:
    
    pop_dict_simu[source] = dict()
    pop_dict_data[source] = dict()
    
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
                        ls='steps-mid',
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
                    ls='steps-mid',
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
            ls='steps-mid',
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
a.set_yscale(ax_scale)

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
            ls='steps-mid'
    )
    
    ax.plot(
            bins_array,
            spectrum_mod0,
            label='model {}'.format(popt),
            ls='steps-mid'
    )

    from model_spectrum import fid_mixture
    import scipy.stats as st
    xdata_fit = spectrum_data
    
    popt = st.norm.fit(xdata_fit)

    xrange = np.linspace(xdata_fit.min(), xdata_fit.max(), 1000)

    pdf = st.norm.pdf(xrange, *popt)
    cdf = st.norm.cdf(xrange, *popt)

    normalization = xdata_fit.size
    pdf_norm = pdf * normalization * np.interp(xrange, bins_array, bins_width)

    ax.plot(xrange, pdf_norm,
            ls='--', color='yellow',
            label='fit')
    
    # a0.plot(xrange, cdf,
    #         ls='-.', color='yellow',
    #         label='fit')

    ax.grid()
    ax.legend()
    ax.set_xlabel('Recoil Energy [keV]')

axes[0].set_ylabel('Pure Gamma Counts')
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

# ## MEANT TO REPRESENT THE 10keV LINE FITTING, not implemented yet
# #                if ind in run_tree.chan_signal:
#                 if ind == 0: # only for heat channel
                    
#                     xdata_fit = xdata_qual[(xdata_qual>1000) & (xdata_qual<1400)]
#                     popt = norm.fit(xdata_fit)
                    
#                     if ana.calibration_peak.cut_type == 'fiducial':
#                         xdata_fit = xdata_fid
                    
# #                    popt = getattr(ana.model.popt, label)
#                     xrange = np.linspace(xdata_fit.min(), xdata_fit.max(), 1000)
# #                    pdf = ana.model.dist.pdf(xrange, *popt)
# #                    cdf = ana.model.dist.cdf(xrange, *popt)
#                     pdf = norm.pdf(xrange, *popt)
#                     cdf = norm.cdf(xrange, *popt)
#                     normalization = getattr(trig,
#                                             'nsamples_{}'.format(
#                                                     ana.calibration_peak.cut_type
#                                             ))
#                     pdf_norm = pdf * normalization * (bin_edges[1] - bin_edges[0])
                    
#                     ax.autoscale(False)
#                     ax.plot(xrange, pdf_norm,
#                             ls='--', color='yellow',
#                             label='fit')
                    
#                     a0.plot(xrange, cdf,
#                             ls='-.', color='yellow',
#                             label='fit')

#%%

# # =============================================================================
# # ### COMPOSITION OF DISTRIButIoN FAiLLED
# # =============================================================================
# import scipy.stats as st
# from scipy.integrate import cumtrapz

# eff_er = binned_efficiency_dict['Background']['ER']
# normed_efficiency = eff_er / np.trapz(eff_er, bins_array)

# eff_funk = lambda x: np.interp(x, bins_array, normed_efficiency)

# class mixture_dist(st.rv_continuous):
#     """ Double Gaussian distribution plus uniform distribution """

#     def _cdf(self, x, f, loc1, scale1, loc2, scale2, fu, loc3, scale3):
#         cdf1 = (1-fu) * (1-f) * st.norm.cdf(x, loc=loc1, scale=scale1)
#         cdf2 = (1-fu) * f * st.norm.cdf(x, loc=loc2, scale=scale2)
#         cdf3 = fu * st.uniform.cdf(x, loc=loc3, scale=scale3)
#         cdf = cdf1 + cdf2 + cdf3
#         return cdf

#     def _pdf(self, x, f, loc1, scale1, loc2, scale2, fu, loc3, scale3):
#         pdf1 = (1-fu) * (1-f)* st.norm.pdf(x, loc=loc1, scale=scale1)
#         pdf2 = (1-fu) * f* st.norm.pdf(x, loc=loc2, scale=scale2)
#         pdf3 = fu * st.uniform.pdf(x, loc=loc3, scale=scale3)
#         pdf = pdf1 + pdf2 + pdf3
#         return pdf

#     def _argcheck(self, *args):
#         """Default check for correct values on args and keywords.
#         Returns condition array of 1's where arguments are correct and
#           0's where they are not.
#         """
#         cond = 1
# #        for arg in args:
# #            cond = np.logical_and(cond, (np.asarray(arg) > 0))
#         return cond
    
#     # def _fitstart(self, data):
#     #     mu01, mu02 = np.quantile(data, [0.10, 0.70])
#     #     mu03 = np.mean(data)
#     #     sig01 = sig02 = abs(mu01 - mu02)/50
#     #     sig03 = abs(mu01 - mu02)
#     #     p0_light = [0.50, mu01, sig01, mu02, sig02, 0.05, mu03, sig03]
#     #     p0 = np.append(p0_light, [0, 1])
#     #     return tuple(p0)

# class mixture_dist_corr(st.rv_continuous):
#     """ Double Gaussian distribution plus uniform distribution """

#     # def _cdf(self, x, f, loc1, scale1, loc2, scale2, fu, loc3, scale3):
        
#     #     full_range = np.linspace(0, 50, 10000)
#     #     pdf_full = self._pdf(
#     #         full_range,
#     #         f, loc1, scale1, loc2, scale2, fu, loc3, scale3
#     #     )
#     #     norm_factor = np.trapz(pdf_full, full_range)
        
#     #     energy_range = np.linspace(0, x, 10000)
#     #     pdf_corr = self._pdf(
#     #         energy_range,
#     #         f, loc1, scale1, loc2, scale2, fu, loc3, scale3
#     #     )
        
#     #     cdf_corr = np.trapz(pdf_corr, energy_range)
#     #     cdf_corr /= norm_factor
        
#     #     return cdf

#     def _pdf(self, x, f, loc1, scale1, loc2, scale2, fu, loc3, scale3):
#         pdf1 = (1-fu) * (1-f)* st.norm.pdf(x, loc=loc1, scale=scale1)
#         pdf2 = (1-fu) * f* st.norm.pdf(x, loc=loc2, scale=scale2)
#         pdf3 = fu * st.uniform.pdf(x, loc=loc3, scale=scale3)
#         pdf = (pdf1 + pdf2 + pdf3) * eff_funk(x)
#         return pdf


#     def _argcheck(self, *args):
#         """Default check for correct values on args and keywords.
#         Returns condition array of 1's where arguments are correct and
#           0's where they are not.
#         """
#         cond = 1
# #        for arg in args:
# #            cond = np.logical_and(cond, (np.asarray(arg) > 0))
#         return cond
    


# p0 = [0.9, 1, 0.2, 10, 0.5, 0.3, 0, 50]

# energy_range = np.linspace(0, 50, 10000)
# spacing = energy_range[1] - energy_range[0]

# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10,7),
#                          sharex=True)

# pdf = mixture_dist().pdf(energy_range, *p0)
# cumsum = np.cumsum(pdf)
# cumsum /= cumsum[-1]

# from scipy.integrate import cumtrapz
# cumsum_trapz = cumtrapz(pdf, energy_range, initial=0)
# cumsum_trapz /= cumsum_trapz[-1]

# cdf = mixture_dist().cdf(energy_range, *p0)
# diff = np.gradient(cdf, spacing)

# eff = eff_funk(energy_range)
# eff_cumsum = np.cumsum(eff)
# eff_cumsum /= eff_cumsum[-1]

# pdf_corr = mixture_dist_corr().pdf(energy_range, *p0)
# pdf_corr /= np.trapz(pdf_corr, energy_range)

# cumsum_corr = np.cumsum(pdf_corr)
# cumsum_corr /= cumsum_corr[-1]

# # cdf_corr = mixture_dist_corr().cdf(energy_range, *p0)
# cdf_corr = cumtrapz(pdf_corr, energy_range, initial=0)
# cdf_corr /= cdf_corr[-1]

# # cdf_corr_funk = lambda x: cumtrapz()

# # def cdf_corr_funk(x):
# #     energy_range = np.linspace(0, )
    
# #     return np.interp()

# # def funk(x):
# #     full_range = np.linspace(0, 50, 10000)
# #     pdf_full = mixture_dist_corr().pdf(
# #         full_range,
# #         *p0
# #     )
# #     norm_factor = np.trapz(pdf_full, full_range)
    
# #     energy_range = np.linspace(0, x, 10000)
# #     pdf_corr = mixture_dist_corr().pdf(
# #         energy_range,
# #         *p0
# #     )
    
# #     cdf_corr = np.trapz(pdf_corr, energy_range)
# #     cdf_corr /= norm_factor

# #     return cdf_corr

# diff_corr = np.gradient(cdf_corr, spacing)


# axes[0, 0].plot(
#     energy_range,
#     pdf,
#     label='pdf'
# )
# axes[0, 0].plot(
#     energy_range,
#     diff,
#     label='diff'
# )

# axes[0, 0].legend()

# axes[0, 1].plot(
#     energy_range,
#     cdf,
#     label='cdf'
# )

# axes[0, 1].plot(
#     energy_range,
#     cumsum,
#     label='cumsum'
# )

# axes[0, 1].plot(
#     energy_range,
#     cumsum_trapz,
#     label='cumsum trapz'
# )

# axes[0,1].legend()

# axes[1, 0].plot(
#     energy_range,
#     eff
# )

# axes[1, 1].plot(
#     energy_range,
#     eff_cumsum
# )

# axes[2, 0].plot(
#     energy_range,
#     pdf_corr,
#     label='pdf_corr'
# )
# axes[2, 0].plot(
#     energy_range,
#     diff_corr,
#     label='diff'
# )

# axes[2, 0].legend()

# axes[2, 1].plot(
#     energy_range,
#     cdf_corr,
#     label='cdf_corr'
# )

# axes[2, 1].plot(
#     energy_range,
#     cumsum_corr,
#     label='cumsum'
# )

# axes[2,1].legend()

# ## FITTING

# from scipy.optimize import minimize


# bins = np.arange(0, 50.5, 0.5)
# bins_width = (bins[1:] - bins[:-1])
# bins_array = bins[:-1] + (bins_width) / 2

# fig, axes = plt.subplots(ncols=2, figsize=(10,7))

# for ax, source in zip(axes, source_list):

#     fine_data_cut_local = (
#         df_fine_data['pure_gamma']
#         & (df_fine_data[energy_column] > 0.2)
#         & (df_fine_data[energy_column] < 50)
#         & (df_fine_data['source'] == source)
#     )
    
#     df_fine_data_local = df_fine_data[fine_data_cut_local]

#     # binned_efficiency_dict['Background']['ER']

#     # energy_array = df_fine_simu_local[energy_column]
#     energy_array = df_fine_data_local[energy_column]
    
#     # energy_array = np.random.normal(10.37, 3, size=5000)
#     # p0 = [0.5, 1, 0.5, 10, 2, 0.3, 0, 50]
#     # energy_array = mixture_dist().rvs(*p0, size=1000)

#     ax.hist(
#         energy_array,
#         bins=bins,
#     )

#     def minus_logpdf(p0):
#         value = -sum(mixture_dist_corr().logpdf(energy_array, *p0))
#         print(value)
#         return value
        
#     minimizer_result = minimize(
#         minus_logpdf,
#         p0,
#         method='Nelder-Mead'
#     )  
    
    
#     popt = minimizer_result.x
#     # popt = st.norm.fit(energy_array)
#     # popt = mixture_dist().fit(energy_array, *p0, floc=0, fscale=1)

#     energy_range = np.linspace(energy_array.min(), energy_array.max(), 1000)

#     pdf = mixture_dist_corr().pdf(energy_range, *popt)
#     pdf /= np.trapz(pdf, energy_range)
    
#     # pdf = mixture_dist().pdf(energy_range, *popt)
#     # pdf = st.norm.pdf(energy_range, *popt)
#     # cdf = st.norm.cdf(xrange, *popt)

#     normalization = energy_array.size
#     pdf_norm = pdf * normalization * np.interp(
#         energy_range, bins_array, bins_width
#     )
#     # pdf_norm = pdf
    
#     ax.plot(
#         energy_range,
#         pdf_norm,
#         ls='--', color='coral',
#         label='fit')

# # a0.plot(xrange, cdf,
# #         ls='-.', color='yellow',
# #         label='fit')

# for ax in axes:
#     ax.grid()
#     ax.legend()
#     ax.set_xlabel('Recoil Energy [keV]')

# axes[0].set_ylabel('Pure Gamma Counts')
# fig.tight_layout()
# fig.subplots_adjust(hspace=0, wspace=0)
    
# =============================================================================
# ER Substraction
# =============================================================================
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
                ls='steps-mid',
                label=key,
            )
            all_num += num
            
        a.plot(
            bins_array,
            all_num,
            color='k',
            ls='steps-mid',
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
                    ls='steps-mid',
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
            ls='steps-mid',
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
            ls='steps-mid'
    )
    
    ax.plot(
            bins_array,
            spectrum_mod0,
            label='model {}'.format(popt),
            ls='steps-mid'
    )

    ax.grid()
    ax.legend()
    ax.set_xlabel('Recoil Energy [keV]')
    ax.set_yscale(ax_scale)
    
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
                ls='steps-mid',
                label=key,
            )
            all_num += num
            
        a.plot(
            bins_array,
            all_num,
            color='k',
            ls='steps-mid',
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
                    ls='steps-mid',
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
            ls='steps-mid',
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
        ls='steps-mid',
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
ax.set_yscale(ax_scale)
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
