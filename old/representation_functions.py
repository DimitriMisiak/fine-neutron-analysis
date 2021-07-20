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

from plot_addon import LegendTitle, custom_autoscale, ax_hist, basic_corner

from data_analysis import (
    ion_chi2_threshold_function,
    heat_chi2_threshold_function,
    analysis_parameters,
    ionization_baseline_resolution,
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
    std_energy_ion,
    quenching,
    energy_recoil,
    lindhard,
    charge_conservation_threshold,
)

def temporal_plot(title, df):
    """
    Monitoring plots.
    Several key quantities in function of time.

    Parameters
    ----------
    title : str
        Figure denomination.
    df : pandas.DataFrame
        DF containing the analysed data from the "data_analysis.py" script.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cut = df['quality_cut']
    
    time = df['timestamp']
    energy_adu_heat = df['energy_adu_heat']
    chi2_heat = df['chi2_heat']
    offset_heat = df['offset_heat']
    
    energy_adu_ion = {
        'ionA': df['energy_adu_ionA'],
        'ionB': df['energy_adu_ionB'],
        'ionC': df['energy_adu_ionC'],
        'ionD': df['energy_adu_ionD'],        
    }
    
    offset_ion = {
        'ionA': df['offset_ionA'],
        'ionB': df['offset_ionB'],
        'ionC': df['offset_ionC'],
        'ionD': df['offset_ionD'],        
    }
    
    slope_ion = {
        'ionA': df['slope_ionA'],
        'ionB': df['slope_ionB'],
        'ionC': df['slope_ionC'],
        'ionD': df['slope_ionD'],        
    }    
    
    # Init figure
    num = '{0}: Monitoring'.format(title)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 10),
                             sharex=True, num=num)
    
    # heat trig vs time
    ax = axes[0]
    ax.set_ylabel('Energy Heat [ADU]')
    ax.set_yscale('symlog')
    
    ax.plot(
            time[cut], energy_adu_heat[cut],
            label='heat a', zorder=10,
            ls='none', marker='2', mew=0.8,
    )
    ax.autoscale(False)
    ax.plot(
            time, energy_adu_heat,
            label='All events',
            ls='none', marker=',', color='silver',
    )
    
    # ion trig vs time
    ax = axes[1]
    ax.set_ylabel('Energy Ion [ADU]')
    ax.set_yscale('symlog')
        
    for i, items in enumerate(energy_adu_ion.items()):
        lab, series = items
        ax.plot(
                time[cut], series[cut],
                label=lab, zorder=10,
                ls='none', marker=str(i+1), mew=0.8
        )        
        
        
    ax.autoscale(False)
    for lab, series in energy_adu_ion.items():
        ax.plot(
                time, series,
                label='All events',
                ls='none', marker=',', color='silver',
        )
    
    # # heat offset vs time
    ax = axes[2]
    ax.set_ylabel('Offset Heat [ADU]')    
    ax.plot(
            time[cut], offset_heat[cut],
            label='heat a', zorder=10,
            ls='none', marker='2', mew=0.8,
    )
    ax.autoscale(False)
    ax.plot(
            time, offset_heat,
            label='All events',
            ls='none', marker=',', color='silver',
    )
    
    # ion offset vs time
    ax = axes[3]
    ax.set_ylabel('Offset Ion [ADU]')
    
    for i, items in enumerate(offset_ion.items()):
        lab, series = items
        ax.plot(
                time[cut], series[cut],
                label=lab, zorder=10,
                ls='none', marker=str(i+1), mew=0.8
        )        
        
    ax.autoscale(False)
    for lab, series in offset_ion.items():
        ax.plot(
                time, series,
                label='All events',
                ls='none', marker=',', color='silver',
        )
    
    # ion slope vs time
    ax = axes[4]
    ax.set_ylabel('Slope Ion [ADU/s]')
    for i, items in enumerate(slope_ion.items()):
        lab, series = items
        ax.plot(
                time[cut], series[cut],
                label=lab, zorder=10,
                ls='none', marker=str(i+1), mew=0.8
        )        
        
    ax.autoscale(False)
    for lab, series in slope_ion.items():
        ax.plot(
                time, series,
                label='All events',
                ls='none', marker=',', color='silver',
        )
    
    # chi2 vs time
    ax = axes[5]
    ax.set_ylabel('$\chi^2$')
    ax.set_yscale('log')
    label = 'chi2 heat A'
    ax.plot(
            time[cut], chi2_heat[cut],
            label='heat a', zorder=10,
            ls='none', marker='2', mew=0.8,
    )
    ax.autoscale(False)
    ax.plot(
            time, chi2_heat,
            label='All events',
            ls='none', marker=',', color='silver',
    )
    
    # formatting the axes
    for ax in axes:
        ax.grid(True, alpha=0.3)
        
        # custom legend
        handles = ['Quality events:',]
        labels = ['',]
        for line in ax.get_lines():
            label = line.get_label()
            if label == 'All events':
                if label != labels[0]:
                    handles.insert(0, line)
                    labels.insert(0, label)
            else:
                handles.append(line)
                labels.append(label)
        
        # handler_map + LegendTitle allow for subtitle in legend
        ax.legend(
                handles, labels, loc=2, framealpha=1,
                bbox_to_anchor=(1.05, 1), borderaxespad=0.,
                handler_map={str: LegendTitle()}
        )
    
        if ax is not axes[-1]:
            # removing the first tick label
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)
    
        if ax is axes[-1]:
            ax.set_xlabel('Time [hours]')
    
    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.subplots_adjust(hspace=.0)
    
    return fig


def temporal_plot_heat_only(title, df):

    cut = df['quality_cut']
    
    time = df['timestamp']
    energy_adu_heat = df['energy_adu_heat']
    
    # Init figure
    num = '{}: Monitoring heat channel'.format(title)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5),
                             sharex=True, num=num)
    
    axes = [axes,]
    
    # heat trig vs time
    ax = axes[0]
    ax.set_ylabel('Energy Heat [ADU]')
    ax.set_yscale('symlog')
    
    ax.plot(
            time[cut], energy_adu_heat[cut],
            label='heat a', zorder=10,
            ls='none', marker='2', mew=0.8,
    )
    ax.autoscale(False)
    ax.plot(
            time, energy_adu_heat,
            label='All events',
            ls='none', marker=',', color='silver',
    )
    
    # formatting the axes
    for ax in axes:
        ax.grid(True, alpha=0.3)
        
        # custom legend
        handles = ['Quality events:',]
        labels = ['',]
        for line in ax.get_lines():
            label = line.get_label()
            if label == 'All events':
                if label != labels[0]:
                    handles.insert(0, line)
                    labels.insert(0, label)
            else:
                handles.append(line)
                labels.append(label)
        
        # handler_map + LegendTitle allow for subtitle in legend
        ax.legend(
                handles, labels, loc=2, framealpha=1,
                bbox_to_anchor=(1.05, 1), borderaxespad=0.,
                handler_map={str: LegendTitle()}
        )
    
        if ax is not axes[-1]:
            # removing the first tick label
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)
    
        if ax is axes[-1]:
            ax.set_xlabel('Time [hours]')
    
    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.subplots_adjust(hspace=.0)
    
    return fig


def plot_chi2_vs_energy(title, df, stream):

    channel_suffix = [
        'heat',
        'ionA',
        'ionB',
        'ionC',
        'ionD',
    ]
    
    # chi2 vs Energy plot
    ax_tuples = ((1, 0), (0, 1), (0, 2), (1, 1), (1, 2))
    
    ax_titles =[
        'Heat',
        'Ion A',
        'Ion B',
        'Ion C',
        'Ion D',
    ]

    quality_cut = df['quality_cut']
    
    num = '{}: $\chi^2$ Cut'.format(title)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11.69, 8.27),
                             num=num
    )

    for suffix, tupl, title in zip(channel_suffix, ax_tuples, ax_titles):
        
        xdata = abs( df['energy_adu_{}'.format(suffix)] )
        ydata = df['chi2_{}'.format(suffix)]
        
        nsamples = xdata.shape[0]
        
        ax = axes[tupl]
        
        ax.plot(xdata, ydata,
                label='All events: {}'.format(nsamples),
                c='red', marker=',', ls='none')
        
        xdata_cut = xdata[quality_cut]
        ydata_cut = ydata[quality_cut]
        
        if nsamples < 1000:
            marker = '.'
        else:
            marker = ','

        ax.plot(xdata_cut, ydata_cut,
                label='Quality events: {}'.format(xdata_cut.shape[0]),
                c='slateblue', marker=marker, ls='none')
    
        ax.legend()
        ax.set_title(title.replace('_', ' '))
        ax.set_xlabel('Energy [ADU]')
        ax.set_ylabel('$\chi^2$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        ax.set_xlim(xdata_cut.min()*0.5, ax.get_xlim()[1])
        ax.set_ylim(ydata_cut.min()*0.5, ax.get_ylim()[1])

    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
        
    fig.delaxes(axes[0,0])    
    fig.tight_layout(rect=(0, 0, 1, 0.98))


    # plotting the limit
    x_data = 10**np.linspace(-2, 5, int(1e4))
    cut_ion = ion_chi2_threshold_function(
        analysis_parameters['ion_chi2_threshold'],
        x_data
    )
    cut_heat = heat_chi2_threshold_function(
        analysis_parameters[stream]['heat_chi2_threshold'],
        x_data)    
    for i, ax in enumerate(fig.get_axes()):
        if i == 2:
            ax.plot(x_data, cut_heat, lw=1, color='k', label='quality cut')
        else:
            ax.plot(x_data, cut_ion, lw=1, color='k', label='quality cut')
        ax.set_xlim(10**-2, 10**5)
        ax.set_ylim(10**1, 10**9)
        ax.legend()            

    return fig


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
    bulk_cut = df['bulk_cut']
    
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
        


        xdata_fid = xdata[quality_cut & bulk_cut]
        ax_hist(ax, bin_edges, xdata_fid,
                'Fiducial events', color='limegreen')[0]        
    
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



def trigger_cut_plot(title, df_analysis):
    
    # trigger cut (specific to simulation)
    trigger_cut = df_analysis['trigger_cut']
    fig_trigger, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(
        df_analysis['timestamp'],
        df_analysis['t_input_simu_trigger'],
        label='all simu',
        ls='none', marker='.', markersize=3
    )
    axes[0].plot(
        df_analysis[trigger_cut]['timestamp'],
        df_analysis[trigger_cut]['t_input_simu_trigger'],
        label='trig simu',
        ls='none', marker='.', color='k'
    )
    axes[0].axhspan(-5, 5, color='limegreen')
    axes[1].plot(
            df_analysis['timestamp'],
            df_analysis['t_nearest_data_trigger'],
            label='all simu',
            ls='none', marker='.', markersize=3
    )
    axes[1].plot(
            df_analysis[trigger_cut]['timestamp'],
            df_analysis[trigger_cut]['t_nearest_data_trigger'],
            label='trig simu',
            ls='none', marker='.', color='k'
    )
    axes[1].axhspan(-5, 5, color='coral')
    axes[0].set_ylabel(r'$\Delta T_{simu}$')
    axes[1].set_ylabel(r'$\Delta T_{data}$')
    axes[1].set_xlabel('Time [hours]')
    axes[0].legend()
    axes[1].legend()
    fig_trigger.tight_layout()
    fig_trigger.subplots_adjust(hspace=.0)

    return fig_trigger


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
    
    bulk_cut = df['bulk_cut']

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
        
        xdata_fid = xdata[quality_cut & bulk_cut]
        ax_hist(ax, bin_edges, xdata_fid,
                'Fiducial events', color='limegreen')     
            
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
    
    bulk_cut = df['bulk_cut']
    
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
        
        ax.plot(
                energy_x[quality_cut & bulk_cut], energy_y[quality_cut & bulk_cut],
                ls='none', marker='2', zorder=11, color='limegreen',
                label='Fiducial Events'
        )

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
    
    bulk_cut = df['bulk_cut']
    
    
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
    
        ax.plot(
                energy_x[quality_cut & bulk_cut], energy_y[quality_cut & bulk_cut],
                ls='none', marker='2', zorder=11, color='limegreen',
                label='Fiducial Events'
        )
        
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

def fid_cut_plot(title, df_analysis, nsigma=2):
    
    quality_cut = df_analysis['quality_cut'] & df_analysis['energy_cut']
    
    try:
        quality_cut = quality_cut & df_analysis['trigger_cut']
    except:
        pass
    
    bulk_cut = df_analysis['bulk_cut'] & quality_cut
    guard_cut = df_analysis['guard_cut'] & quality_cut
    all_cut = pd.Series(data=True, index=bulk_cut.index) & quality_cut
    
    event_dict = {
        'all': [all_cut, 'k', 10],
        'bulk': [bulk_cut, 'b', 7],
        'guard': [guard_cut, 'r', 4],
    }
    
    axes=None
    for key, char in event_dict.items():
        cut, color, mks = char
    
        samples = df_analysis[cut][[
            'energy_ionA',
            'energy_ionB',
            'energy_ionC',
            'energy_ionD',
        ]]    
    
        fig_fid, axes = basic_corner(
            samples.values,
            samples.columns,
            num = '{} : fid cut'.format(title),
            label=key,
            axes=axes,
            markersize=mks,
            color=color
        )
    
    axes = fig_fid.get_axes()
    
    ei_array = np.linspace(-50, 50, int(1e3))
    thresh_array = nsigma * ionization_baseline_resolution() * np.ones(int(1e3))
    
    color_bulk='deepskyblue'
    for i in (0, 3, 5):
        axes[i].plot(
            ei_array,
            thresh_array,
            path_effects=cartoon,
            color=color_bulk,
        )
        axes[i].plot(
            ei_array,
            -thresh_array,
            path_effects=cartoon,
            color=color_bulk,
        )
        axes[i].fill_between(
            ei_array,
            thresh_array,
            -thresh_array,
            color=color_bulk,
        )

    color_guard='coral'
    for i in (0, 3, 5):
        axes[i].plot(
            thresh_array,
            ei_array,
            path_effects=cartoon,
            color=color_guard,
        )
        axes[i].plot(
            -thresh_array,
            ei_array,
            path_effects=cartoon,
            color=color_guard,
        )
        axes[i].fill_betweenx(
            ei_array,
            thresh_array,
            -thresh_array,
            color=color_guard,
        )
        
    for i in (2,):
        axes[i].plot(
            thresh_array,
            ei_array,
            path_effects=cartoon,
            color=color_bulk,
        )
        axes[i].plot(
            -thresh_array,
            ei_array,
            path_effects=cartoon,
            color=color_bulk,
        )
        axes[i].fill_betweenx(
            ei_array,
            thresh_array,
            -thresh_array,
            color=color_bulk,
        )

    for i in (2,):
        axes[i].plot(
            ei_array,
            thresh_array,
            path_effects=cartoon,
            color=color_guard,
        )
        axes[i].plot(
            ei_array,
            -thresh_array,
            path_effects=cartoon,
            color=color_guard,
        )
        axes[i].fill_between(
            ei_array,
            thresh_array,
            -thresh_array,
            color=color_guard,
        )
        
    for ax in axes:
        ax.set_xlim(-15, 15) 
        ax.set_ylim(-15, 15) 
    
    return fig_fid


def band_cut_plots(title, df_analysis, nsigma=2):

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

    event_dict = {
        'all': [all_cut, 'grey', 12],
        'ho': [ho_cut, 'k', 9],
        'gamma': [gamma_cut, 'r', 6],
        'neutron': [neutron_cut, 'b', 3]
    }
    
    fig_band_ecei, ax_ecei = plt.subplots(
        num = '{} : band cut ecei'.format(title),
        figsize=(10,7),
    )
    ax_ecei.set_title('{} : band cut ecei'.format(title))
    fig_band_quenching, ax_qu = plt.subplots(
        num = '{} : band cut quenching'.format(title),
        figsize=(10,7),
    )
    ax_qu.set_title('{} : band cut quenching'.format(title))
    
    for key, char in event_dict.items():
        cut, color, mks = char
        
        ec_array = df_analysis[cut]['energy_heat']
        ei_array = df_analysis[cut]['energy_ion_bulk']
        er_array = energy_recoil(ec_array, ei_array, 2)
        qu_array = quenching(ec_array, ei_array, 2)
        
        ax_ecei.plot(
            ec_array,
            ei_array,
            ls='none',
            marker='o',
            markersize=mks,
            color=color,
            label=key
        )

        ax_qu.plot(
            er_array,
            qu_array,
            ls='none',
            marker='o',
            markersize=mks,
            color=color,
            label=key
        )
    
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
    
    # heat only
    qu_ho = np.zeros(int(1e4))
    ec_ho = energy_heat_from_er_and_quenching(er_theory, qu_ho, 2)    
    ei_ho = energy_ion_from_er_and_quenching(er_theory, qu_ho)
    ei_err_ho = nsigma*std_energy_ion(np.zeros(ec_ho.shape))
    
    qu_ho_sup = quenching(ec_ho, ei_ho + ei_err_ho, 2)
    qu_ho_inf = quenching(ec_ho, ei_ho - ei_err_ho, 2)    
    
    qu_ho_sup_aux = quenching(ec_ho, ei_ho + ei_err_ho, 2)
    er_ho_sup = energy_recoil(ec_ho, ei_ho + ei_err_ho, 2)
    qu_ho_sup = np.interp(er_theory, er_ho_sup, qu_ho_sup_aux)
    
    qu_ho_inf_aux = quenching(ec_ho, ei_ho - ei_err_ho, 2)
    er_ho_inf = energy_recoil(ec_ho, ei_ho - ei_err_ho, 2)
    qu_ho_inf = np.interp(er_theory, er_ho_inf, qu_ho_inf_aux)
    
    
    # GAMMA
    ax_ecei.plot(
        ec_gamma,
        ei_gamma,
        label='gamma band',
        color='coral',
        path_effects=cartoon
    )
    ax_ecei.fill_between(
        ec_gamma,
        ei_gamma + ei_err_gamma,
        ei_gamma - ei_err_gamma,
        label='gamma band',
        color='coral',
        alpha=0.5,
    )
    
    ax_qu.plot(
        er_theory,
        qu_gamma,
        label='gamma band',
        color='coral',
        path_effects=cartoon
    )
    ax_qu.fill_between(
        er_theory,
        qu_gamma_sup,
        qu_gamma_inf,
        label='gamma band',
        color='coral',
        alpha=0.5,
    )
    
    # NEUTRON
    ax_ecei.plot(
        ec_neutron,
        ei_neutron,
        label='neutron band',
        color='deepskyblue',
        path_effects=cartoon
    )
    ax_ecei.fill_between(
        ec_neutron,
        ei_neutron + ei_err_neutron,
        ei_neutron - ei_err_neutron,
        label='neutron band',
        color='deepskyblue',
        alpha=0.5,
    )
    
    ax_qu.plot(
        er_theory,
        qu_neutron,
        label='neutron band',
        color='deepskyblue',
        path_effects=cartoon
    )
    ax_qu.fill_between(
        er_theory,
        qu_neutron_sup,
        qu_neutron_inf,
        label='neutron band',
        color='deepskyblue',
        alpha=0.5,
    )
    
    # HEAT ONLY
    ax_ecei.plot(
        ec_ho,
        ei_ho,
        label='ho band',
        color='lightgrey',
        path_effects=cartoon
    )
    ax_ecei.fill_between(
        ec_ho,
        ei_ho + ei_err_ho,
        ei_ho - ei_err_ho,
        label='ho band',
        color='lightgrey',
        alpha=0.5,
    )
    
    ax_qu.plot(
        er_theory,
        qu_ho,
        label='ho band',
        color='lightgrey',
        path_effects=cartoon
    )
    ax_qu.fill_between(
        er_theory,
        qu_ho_sup,
        qu_ho_inf,
        label='ho band',
        color='lightgrey',
        alpha=0.5,
    )
    
    ax_ecei.set_ylabel('Ionization Energy [keV]')
    ax_ecei.set_xlabel('Heat Energy [keV]')
    ax_qu.set_ylabel('Quenching factor')
    ax_qu.set_xlabel('Recoil Energy [keV]')    
    
    ax_qu.set_ylim(-0.5, 1.5)
    ax_qu.set_xlim(0, 50)
    
    ax_ecei.set_xlim(-5, 50)
    ax_ecei.set_ylim(-5, 50)
    
    
    for ax in (ax_ecei, ax_qu):
        ax.legend()
        ax.grid()
    
    for fig in (fig_band_ecei, fig_band_quenching):
        fig.tight_layout()

    return fig_band_ecei, fig_band_quenching


def charge_conservation(title, df):
    quality_cut = df['quality_cut']
    charge_cut = df['charge_conservation_cut']   
    
    energy_heat = df['energy_heat'][quality_cut]
    ion_conservation = df['energy_nodecor_ion_conservation'][quality_cut]

    x_array = np.linspace(
        energy_heat.min(),
        energy_heat.max(),
        int(1e4)
    )
    
    fig, ax = plt.subplots(
        num = '{} : charge conservation'.format(title),
        figsize=(10,7),
    )
    
    ax.plot(
        energy_heat[quality_cut & charge_cut],
        ion_conservation[quality_cut & charge_cut],
        ls='none',
        marker='.',
        color='b',
        alpha=0.1,
        label='Charge Conservation Cut'
    )

    ax.plot(
        energy_heat[quality_cut & ~charge_cut],
        ion_conservation[quality_cut & ~charge_cut],
        ls='none',
        marker='.',
        alpha=0.1,
        color='r',
        label='Quality_ Cut'
    )

    ax.plot(
        x_array,
        charge_conservation_threshold(x_array),
        color='coral'
    )
    
    ax.plot(
        x_array,
        -charge_conservation_threshold(x_array),
        color='coral'
    )
    
    ax.set_title('{} : charge conservation'.format(title))
    ax.grid()
    ax.set_xlim(5e-2, 300)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xscale('log')
    
    ax.set_xlabel('Heat Energy [keV]')
    ax.set_ylabel('Charge Conservation: A + B - C - D [keV]')
    
    fig.tight_layout()
    
    return fig


if __name__ == '__main__':
    
    plt.close('all')
    plt.rcParams['text.usetex']=True

    analysis_dir = '/home/misiak/Analysis/neutron_background'
    analysis_data_path = '/'.join([analysis_dir, 'data_analysis.h5'])

    # stream_list = pd.read_hdf(
    #     fine_data_path,
    #     key='df',
    #     columns=['stream',]
    # )['stream'].unique()    
    
    # from tqdm import tqdm
    # for stream in stream_list:
    if True:
        stream = 'tg28l000'
    
        df_analysis = pd.read_hdf(
            analysis_data_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        fid_cut_plot(stream, df_analysis)
        
        # fig_temp = temporal_plot(stream, df_analysis)
        # fig_chi2 = plot_chi2_vs_energy(stream, df_analysis)
        # fig_hist_trig = histogram_adu(stream, df_analysis)
        # fig_hist_trig_ev = histogram_ev(stream, df_analysis)
        # fig_ion = ion_vs_ion(stream, df_analysis)
        # fig_virtual = virtual_vs_virtual_ev(stream, df_analysis)
        # fig = charge_conservation(stream, df_analysis)
        