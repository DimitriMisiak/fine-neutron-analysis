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

cartoon = [
        pe.Stroke(linewidth=3, foreground='k'),
        pe.Normal(),
]

from plot_addon import (
    LegendTitle,
    custom_autoscale,
    ax_hist,
    basic_corner,
    save_figure_dict
)

from pipeline_data_quality import (
    ion_chi2_threshold_function,
    heat_chi2_threshold_function,
    quality_parameters,
)


def histogram_adu(title, df, bins=1000):
    
    ax_tuples = ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
 
    channel_suffix = [
        'heat',
        'ionA',
        'ionB',
        'ionC',
        'ionD',
    ]    
    
    ax_titles =[
        'Heat',
        'Ion A',
        'Ion B',
        'Ion C',
        'Ion D',
    ]
    
    quality_cut = df['quality_cut']
    # bulk_cut = df['bulk_cut']
    
    num = '{} : Quality Cut Histogram'.format(title)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11.69, 8.27),
                             num=num)
    
    for suffix, tupl, label in zip(channel_suffix, ax_tuples, ax_titles):
        
        xdata = df['energy_adu_{}'.format(suffix)]
        
        ax = axes[tupl]
        xdata_qual = xdata[quality_cut]
        

        # try:
        #     bin_edges = custom_bin_edges(xdata_qual, 
        #                                  getattr(noise.sigma0, label))
        
        bin_edges = np.histogram_bin_edges(xdata[quality_cut], bins=bins)

        ax_hist(ax, bin_edges, xdata,
                'All events', color='coral')
        ax_hist(ax, bin_edges, xdata_qual,
                'Quality events', color='slateblue')[0]
        


        # xdata_fid = xdata[quality_cut & bulk_cut]
        # ax_hist(ax, bin_edges, xdata_fid,
        #         'Fiducial events', color='limegreen')[0]        
    
### MEANT TO REPRESENT THE 10keV LINE FITTING, not implemented yet
##                if ind in run_tree.chan_signal:
#                if ind == 0: # only for heat channel
#                    
#                    xdata_fit = xdata_qual[(xdata_qual>1000) & (xdata_qual<1400)]
#                    popt = norm.fit(xdata_fit)
#                    
#                    if ana.calibration_peak.cut_type == 'fiducial':
#                        xdata_fit = xdata_fid
#                    
##                    popt = getattr(ana.model.popt, label)
#                    xrange = np.linspace(xdata_fit.min(), xdata_fit.max(), 1000)
##                    pdf = ana.model.dist.pdf(xrange, *popt)
##                    cdf = ana.model.dist.cdf(xrange, *popt)
#                    pdf = norm.pdf(xrange, *popt)
#                    cdf = norm.cdf(xrange, *popt)
#                    normalization = getattr(trig,
#                                            'nsamples_{}'.format(
#                                                    ana.calibration_peak.cut_type
#                                            ))
#                    pdf_norm = pdf * normalization * (bin_edges[1] - bin_edges[0])
#                    
#                    ax.autoscale(False)
#                    ax.plot(xrange, pdf_norm,
#                            ls='--', color='yellow',
#                            label='fit')
#                    
#                    a0.plot(xrange, cdf,
#                            ls='-.', color='yellow',
#                            label='fit')
        
        ax.legend(loc=2)
        ax.set_title(label.replace('_', ' '))
        
    
    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))

### MESSAGE BOX ON THE LEFT, to tell about the calibration, sensitivity etc..
    # msg_list = list()
    # for ind in run_tree.chan_signal:
    #     lab = run_tree.chan_label[ind]
    #     polar, pos10kev, sens_adu, gain_chan, sens_nv, sigma_adu, sigma_ev = optimization_info(ana, ind)
    #     msg_title = r'\underline{'+lab.replace('_', ' ')+'}'
    #     msg_core = (
    #     r'\\ '
    #     r'Position 10keV = {:.0f} ADU \\ '
    #     r'Gain = {} nV/ADU \\ '
    #     r'Sensitivity = {:.2e} ADU/eV = {:.1f} nV/keV \\ '
    #     r'Resolution = {:.2f} ADU = {:.1f} eV '
    #     ).format(pos10kev, gain_chan, sens_adu, sens_nv, sigma_adu, sigma_ev)
        
    #     msg_chan = msg_title+msg_core
    #     msg_list.append(msg_chan)

    # msg = r'\\'.join(msg_list)
    
    # fig.text(0.1, 0.82, msg,
    #          horizontalalignment='left',
    #          verticalalignment='center',
    #          bbox=dict(facecolor='white', alpha=0.5))
    

    # resize the plots
    fig.get_axes()[0].set_xlim(-200, 2000)
    for i, ax in enumerate(fig.get_axes()[:5]):
        if i==0:
            ax.set_xlim(-200, 2000)
        else:
            ax.set_xlim(-70, 70)    
 
    fig.delaxes(axes[0,0])    
    fig.tight_layout()
        
    return fig


def crosstalk_correction(title, df):
    
    samples = df[df.quality_cut][[
        'energy_adu_ionA',
        'energy_adu_ionB',
        'energy_adu_ionC',
        'energy_adu_ionD',
    ]]
    samples_corr = df[df.quality_cut][[
        'energy_adu_corr_ionA',
        'energy_adu_corr_ionB',
        'energy_adu_corr_ionC',
        'energy_adu_corr_ionD',
    ]]
    fig_cross, axes = basic_corner(
        samples.values,
        samples.columns,
        num = '{}: Cross-talk Correction'.format(title),
        label='raw',
        alpha=0.1,
    )
    basic_corner(
        samples_corr.values,
        samples_corr.columns,
        axes=axes,
        color='slateblue',
        zorder=-1,
        label='corrected'
    )
    for ax in fig_cross.get_axes():
        ax.axvline(0, color='r', zorder=-5)
        ax.axhline(0, color='r', zorder=-5)
        
    return fig_cross


def nodecor_crosstalk_correction(title, df):
    
    samples = df[df.quality_cut][[
        'energy_adu_nodecor_ionA',
        'energy_adu_nodecor_ionB',
        'energy_adu_nodecor_ionC',
        'energy_adu_nodecor_ionD',
    ]]
    samples_corr = df[df.quality_cut][[
        'energy_adu_corr_nodecor_ionA',
        'energy_adu_corr_nodecor_ionB',
        'energy_adu_corr_nodecor_ionC',
        'energy_adu_corr_nodecor_ionD',
    ]]
    fig_cross, axes = basic_corner(
        samples.values,
        samples.columns,
        num = '{}: Cross-talk Correction Nodecor'.format(title),
        label='raw',
        alpha=0.1,
    )
    basic_corner(
        samples_corr.values,
        samples_corr.columns,
        axes=axes,
        color='slateblue',
        zorder=-1,
        label='corrected'
    )
    for ax in fig_cross.get_axes():
        ax.axvline(0, color='r', zorder=-5)
        ax.axhline(0, color='r', zorder=-5)
        
    return fig_cross


def histogram_ev(title, df, bins=1000):


    channel_suffix = [
        'heat',
        'ionB',
        'ionD',
        'ion_bulk'
    ]    
    
    ax_titles =[
        'Heat',
        'Ion B',
        'Ion D',
        'Ion Bulk',
    ]
    
    ax_tuples = ((0, 0), (0, 1), (1, 0), (1, 1))

    quality_cut = df['quality_cut']
    
    try:
        quality_cut = quality_cut & df['trigger_cut']
    except:
        pass    
    
    # bulk_cut = df['bulk_cut']

    num = '{} : Quality Cut Histogram EV'.format(title)


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.69, 8.27),
                             num=num)
    
    for suffix, tupl, label in zip(channel_suffix, ax_tuples, ax_titles):
        
        xdata = df['energy_{}'.format(suffix)]
        ax = axes[tupl]
        xdata_qual = xdata[quality_cut]
        
        # if etype is trig:
        #     bin_edges = custom_bin_edges(xdata_qual, 
        #                                  getattr(noise.sigma0_ev, label))
    
        bin_edges = np.histogram_bin_edges(xdata[quality_cut], bins=bins)
    
        ax_hist(ax, bin_edges, xdata,
                'All events', color='coral')
        ax_hist(ax, bin_edges, xdata_qual,
                'Quality events', color='slateblue')
        
        # xdata_fid = xdata[quality_cut & bulk_cut]
        # ax_hist(ax, bin_edges, xdata_fid,
        #         'Fiducial events', color='limegreen')     
            
        ax.set_xlabel('Enregy [keV]')
        ax.legend(loc=2)
        ax.set_title(label.replace('_', ' '))
        
        ax.set_xlim(-2.5, 15)
        

    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
  
    fig.tight_layout()

    return fig


def ion_vs_ion(title, df):
    
    quality_cut = df['quality_cut']
    
    try:
        quality_cut = quality_cut & df['trigger_cut']
    except:
        pass    
    
    # bulk_cut = df['bulk_cut']
    
    # initializing pseudo-corner plot
    ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
    ax_discard = [(0, 1), (1, 2), (0, 2)]
    
    # chan_x = np.insert(run_tree.chan_veto, 0, run_tree.chan_collect[1])
    # chan_y = np.append(run_tree.chan_veto, run_tree.chan_collect[0])    
    chan_x = ['ionD', 'ionA', 'ionC']
    chan_y = ['ionA', 'ionC', 'ionB']
   
    num = '{} : Ion vs Ion'.format(title)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.27, 8.27),
                             num=num, sharex='col', sharey='row')
    
    # actually plotting the data
    for atupl in ax_tuples:
        
        ax = axes[atupl]
        xind = chan_x[atupl[1]]
        yind = chan_y[atupl[0]]
    
        # energy_x = energy[:, xind]
        # energy_y = energy[:, yind]
        energy_x = df['energy_{}'.format(xind)]
        energy_y = df['energy_{}'.format(yind)]
        
        # ax.plot(
        #         energy_x[quality_cut & bulk_cut], energy_y[quality_cut & bulk_cut],
        #         ls='none', marker='2', zorder=11, color='limegreen',
        #         label='Fiducial Events'
        # )

        ax.plot(
                energy_x[quality_cut], energy_y[quality_cut],
                ls='none', marker='1', zorder=10, color='slateblue',
                label='Quality Events'
        )
            
        ax.plot(
                energy_x, energy_y,
                ls='none', marker=',', zorder=9, color='coral',
                label='All events'
        )
    
        ### FOR THE DRAWING OF THE BULK CUT
        # if n_sigma != 0:
        #     if xind in run_tree.chan_veto:
        #         lab = run_tree.chan_label[xind]
        #         xamp = n_sigma*getattr(noise.sigma0, lab)
        #         ymin, ymax = energy_y.min(), energy_y.max()
        #         ax.fill_betweenx([ymin, ymax], -xamp, +xamp, color='lavender')
        
        #     if yind in run_tree.chan_veto:
        #         lab = run_tree.chan_label[yind]
        #         yamp = n_sigma*getattr(noise.sigma0, lab)
        #         xmin, xmax = energy_x.min(), energy_x.max()
        #         ax.fill_between([xmin, xmax], -yamp, +yamp, color='lavender',
        #                          label='Fiducial selection ({}$\sigma$)'.format(n_sigma)
        #                         )

            
        custom_autoscale(ax, energy_x[quality_cut], energy_y[quality_cut])
        
        ax.grid(alpha=0.3)
        
        if atupl == (0,0):
            ax.legend(loc='lower left', framealpha=1,
                      bbox_to_anchor=(1.05, 0.05), borderaxespad=0.,
            )
        
        if atupl[0] == 2:
            ax.set_xlabel(
                    'Energy {} [ADU]'.format(xind)
            )
                
        if atupl[1] == 0:
            ax.set_ylabel(
                    'Energy {} [ADU]'.format(yind)
            )
    
    fig.text(0.65, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    for tupl in ax_discard:
        fig.delaxes(axes[tupl])
    fig.tight_layout()
    fig.subplots_adjust(hspace=.0, wspace=.0)

    axes = fig.get_axes()
    for ax in axes:
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    
    return fig


def virtual_vs_virtual_ev(title, df):
    
    quality_cut = df['quality_cut']
    
    try:
        quality_cut = quality_cut & df['trigger_cut']
    except:
        pass
    
    # bulk_cut = df['bulk_cut']
    
    
    # initializing pseudo-corner plot
    ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
    ax_discard = [(0, 1), (1, 2), (0, 2)]
    
    
    chan_x = ['heat', 'ionB', 'ionD']
    chan_y = ['ionB', 'ionD', 'ion_bulk']
 
    num = '{} : VIRTUAL vs VIRTUAL EV'.format(title)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.27, 8.27),
                             num=num, sharex='col', sharey='row')
    
    # actually plotting the data
    for atupl in ax_tuples:
        
        ax = axes[atupl]
        xlab = chan_x[atupl[1]]
        ylab = chan_y[atupl[0]]
    
        energy_x = df['energy_{}'.format(xlab)]
        energy_y = df['energy_{}'.format(ylab)]
    
        # ax.plot(
        #         energy_x[quality_cut & bulk_cut], energy_y[quality_cut & bulk_cut],
        #         ls='none', marker='2', zorder=11, color='limegreen',
        #         label='Fiducial Events'
        # )
        
        ax.plot(
                energy_x[quality_cut], energy_y[quality_cut],
                ls='none', marker='1', zorder=10, color='slateblue',
                label='Quality Events'
        )
            
        ax.plot(
                energy_x, energy_y,
                ls='none', marker=',', zorder=9, color='coral',
                label='All events'
        )
            
        custom_autoscale(ax, energy_x[quality_cut], energy_y[quality_cut])
        
        ax.grid(alpha=0.3)
        
        if atupl == (0,0):
            ax.legend(loc='lower left', framealpha=1,
                      bbox_to_anchor=(1.05, 0.05), borderaxespad=0.,
            )
        
        if atupl[0] == 2:
            ax.set_xlabel(
                    'Energy {} [eV]'.format(
                            xlab.replace('_', ' ')
                    )
            )
                
        if atupl[1] == 0:
            ax.set_ylabel(
                    'Energy {} [eV]'.format(
                            ylab.replace('_', ' ')
                    )
            )
    
    fig.text(0.65, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    for tupl in ax_discard:
        fig.delaxes(axes[tupl])
    fig.tight_layout()
    fig.subplots_adjust(hspace=.0, wspace=.0)
    
    for ax in fig.get_axes():
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)    
    
    return fig


def plot_10kev(title, df_analysis):
    
    delta_volt = 2 #V
    quality_cut = df_analysis['quality_cut']
    
    try:
        quality_cut = quality_cut & df_analysis['trigger_cut']
    except:
        pass
    
    fig_10kev, ax = plt.subplots(num='Tot Ion vs Heat', figsize=(10, 7))
    ax.set_title('{} : 10keV events'.format(title))
    ax.plot(
        df_analysis[quality_cut]['energy_heat'],
        df_analysis[quality_cut]['energy_ion_total'],
        label='quality events',
        ls='none',
        marker='.',
        color='b',
        alpha=0.3
    )
    #guide for 10keV
    ax.plot(
        [10.37/(1+delta_volt/3), 10.37],
        [0, 10.37], 
        zorder=-20, lw=10,
        color='gold', label='10keV band (theory)'
    )
    ax.grid()
    ax.set_xlim(-2, 13)
    ax.set_ylim(-2, 13)
    ax.set_ylabel('Total Ionization Energy A+B+C+D [keV]')
    ax.set_xlabel('Heat Energy [keV]')
    fig_10kev.tight_layout()
    
    return fig_10kev


def calibration_plots(
        stream,
        title,
        df_analysis,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
    
    fig_dict = dict()
    
    ### histogramm ADU
    fig_hist_trig = histogram_adu(title, df_analysis, bins=10000)
    fig_dict['histogramm_ADU'] = fig_hist_trig

    ### crosstalk correction
    fig_cross = crosstalk_correction(title, df_analysis)
    fig_dict['crosstalk_correction'] = fig_cross

    ## nodecor
    ### crosstalk correction
    fig_cross_nodecor = nodecor_crosstalk_correction(title, df_analysis)
    fig_dict['nodecor_crosstalk_correction'] = fig_cross_nodecor

    ### histogramm ev
    fig_hist_trig_ev = histogram_ev(title, df_analysis, bins=10000)
    fig_dict['histogramm_ev'] = fig_hist_trig_ev
    
    ### ion vs ion
    fig_ion = ion_vs_ion(title, df_analysis)
    fig_dict['ion_vs_ion'] = fig_ion
    
    ### ion vs ion virtual
    fig_virtual = virtual_vs_virtual_ev(title, df_analysis)
    fig_dict['ion_vs_ion_virtual'] = fig_virtual
    
    ### 10kev plot
    fig_10kev = plot_10kev(title, df_analysis)
    fig_dict['plot_10kev'] = fig_10kev

    return fig_dict


if __name__ == '__main__':
    
    plt.close('all')
    plt.rcParams['text.usetex']=True
    from tqdm import tqdm
    

    analysis_dir = '/home/misiak/Analysis/neutron_background'
    output_dir = '/'.join([analysis_dir, 'analysis_plots'])
    extension='png'
    
    h5type_list = [
        'data',
        'noise',
        'simu'
    ]
    
    stream_list = [
        'tg18l005',
        'tg27l000',
        'tg28l000',
        'tg17l007',
        'tg19l010',
        'tg20l000',
        'tg21l000'
    ]

    stream = stream_list[0]
    h5type = 'data'
    h5_path = '/'.join([ analysis_dir, "data_calibrated.h5" ])

    df_analysis = pd.read_hdf(
        h5_path,
        key='df',
        where=(
            'stream = "{0}"'
        ).format(stream)
    )
    
    source = df_analysis['source'].unique()[0]
    title = (
        '{0} {1} {2}'
    ).format(stream, h5type, source).replace('_', ' ') 
    
    
    quality_cut = df_analysis['quality_cut']
    A = df_analysis[quality_cut]['energy_adu_heat']
    B = (1000 < A) & (A < 1400)
    
    inf, med, sup = np.quantile(A[B], [0.16, 0.5, 0.84])
    
    C = (inf < A) & (A < sup)

    
    plt.figure()
    
    plt.plot(
        df_analysis[quality_cut]['energy_adu_heat'],
        df_analysis[quality_cut]['energy_ion_total'],
        ls='none',
        marker='.'
    )
    
    fig, ax = plt.subplots()
    
    bin_edges = np.linspace(1000, 1400, int(1e2))
    ax0 = ax_hist(ax, bin_edges, A[B], 'quality')[0]
    ax.axvline(inf, color='k')
    ax.axvline(sup, color='k')
    ax.axvline(med, color='k', lw=3)
    
    print('Heat ', inf, med, sup)
    
    ### IONIZATION CALIBRATION
    num = '{} : Ionization Calibration'.format(title)
    fig, axes = plt.subplots(ncols=2, num=num, figsize=(10,7))
    
    D = df_analysis[quality_cut & C]
    
    xB = D['energy_adu_corr_ionB']
    yA = D['energy_adu_corr_ionA']
    line, = axes[0].plot(
        xB,
        yA,
        ls='none',
        marker='.',
        label='Quality 10keV events (1$\sigma$)',
    )
    
    axes[0].set_xlabel('Amplitude B')
    axes[0].set_ylabel('Amplitude A')
    axes[0].grid()


    from graphic_selection import Data_Selector
    
    linear = lambda x,*p: p[0]*x + p[1]
    from scipy.optimize import curve_fit
    
    
    def funAB(indexes):
        x_data = D['energy_adu_ionB'].array[indexes]
        y_data = D['energy_adu_ionA'].array[indexes]

        popt, pcov = curve_fit(
            linear,
            x_data,
            y_data,
            p0=[1,1]
        )

        ion_array = np.linspace(
            x_data.min(),
            x_data.max(),
            int(1e3)
        )       
        
        axes[0].plot(
            ion_array,
            linear(ion_array, *popt),
            color='k'
        )
        
        # axes[0].legend(title=popt)
        print(popt)
        return popt
        
    # ds = Data_Selector(axes[0], line, funAB)

    from scipy.odr import ODR, Model, Data, RealData

    # def func(beta, x):
    #     y = beta[0]+beta[1]*x
    #     return y

    def func(beta, x):
        y = (-beta[1]/beta[0])*x+beta[1]
        return y


    ### Calibration A & B
    data = RealData(xB, yA, 5, 5)
    model = Model(func)
   
    odr = ODR(data, model, [1,1])
    
    odr.set_job(fit_type=0)
    output = odr.run()
    
    xn = np.linspace(
        xB.min(),
        xB.max(),
        int(1e3)
    )       
    yn = func(output.beta, xn)
    
    axes[0].plot(xn,yn,'r-',label='Model fitting')


    ### Calibration C & D

    xD = D['energy_adu_corr_ionD']
    yC = D['energy_adu_corr_ionC']

    axes[1].plot(
        xD,
        yC,
        ls='none',
        marker='.',
        label='Quality 10keV events (1$\sigma$)'
    )

    axes[1].set_xlabel('Amplitude D')
    axes[1].set_ylabel('Amplitude C')
    axes[1].grid()

    data = RealData(xD, yC, 5, 5)
    model = Model(func)
   
    odr = ODR(data, model, [1,1])
    
    odr.set_job(fit_type=0)
    output = odr.run()
    
    peakB, peakA = output.beta
    sigmaB = output.cov_beta[0,0]**0.5
    sigmaA = output.cov_beta[1,1]**0.5
    
    xn = np.array([0, peakB])
       
    # yn = func(output.beta, xn)
    yn = np.array([peakA, 0])
    
    axes[1].errorbar(
        xn,
        yn,
        yerr=sigmaA,
        xerr=sigmaB,
        marker='o',
        color='r',
        label='Model fitting'
    )
    
    for ax in axes:
        ax.legend()
    
    fig.text(0.5, 0.98, num,
         horizontalalignment='center',
         verticalalignment='center',
         bbox=dict(facecolor='lime', alpha=0.5))
        
    fig.tight_layout(rect=(0,0, 1, 0.98))