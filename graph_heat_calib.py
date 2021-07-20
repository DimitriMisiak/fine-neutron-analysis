#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:04:43 2020

@author: misiak
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams['text.usetex']=True

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

# =============================================================================
# Function as usual
# =============================================================================

def histogram_adu(title, df, bins=10000):
    
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
    # bulk_cut = df['bulk_cut']1000
    
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
        
        ax.legend(loc=2)
        ax.set_title(label.replace('_', ' '))
        
    
    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))

    # resize the plots
    fig.delaxes(axes[0,0])
    for i, ax in enumerate(fig.get_axes()):
        if i==0:
            ax.set_xlim(-200, 2000)
        else:
            ax.set_xlim(-70, 70)    
 
    axes[0,1].set_xlim(-200, 1000)
        
        
    fig.tight_layout()
        
    return fig


def histogram_ev(title, df, bins=1000):


    channel_suffix = [
        'ionA',
        'ionB',
        'ionC',
        'ionD',
    ]    
    
    ax_titles =[
        'Ion A',
        'Ion B',
        'Ion C',
        'Ion D',
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


def plot_10kev(title, df_analysis):
    
    delta_volt = 2 #V
    quality_cut = df_analysis['quality_cut']
    
    try:
        quality_cut = quality_cut & df_analysis['trigger_cut']
    except:
        pass
    
    selection_cut = df_analysis['selection_cut']
    
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
    
    ax.plot(
        df_analysis[selection_cut]['energy_heat'],
        df_analysis[selection_cut]['energy_ion_total'],
        label='10kev selected events',
        ls='none',
        marker='.',
        color='r',
        alpha=0.5
    )
    
    # #guide for 10keV
    # ax.plot(
    #     [10.37/(1+delta_volt/3), 10.37],
    #     [0, 10.37], 
    #     zorder=-20, lw=10,
    #     color='gold', label='10keV band (theory)'
    # )
    ax.grid()
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_ylabel('Total Ionization Energy A+B+C+D [keV]')
    ax.set_xlabel('Heat Energy [keV]')
    fig_10kev.tight_layout()
    
    return fig_10kev


def xtalk_plots(
        stream,
        title,
        df_analysis,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
    
    fig_dict = dict()

    # ### histogramm ADU
    # fig_hist_trig = histogram_adu(title, df_analysis, bins=1000)
    # fig_dict['histogramm_ADU'] = fig_hist_trig

    # ### histogramm ev
    # fig_hist_trig_ev = histogram_ev(title, df_analysis, bins=100)
    # fig_dict['histogramm_ev'] = fig_hist_trig_ev
    
    # ### ion vs ion
    # fig_ion = ion_vs_ion(title, df_analysis)
    # fig_dict['ion_vs_ion'] = fig_ion

    ### 10kev plot
    fig_10kev = plot_10kev(title, df_analysis)
    fig_dict[stream + '_plot_10kev_true'] = fig_10kev

    return fig_dict



if __name__ == '__main__':
    
    plt.close('all')
    plt.rcParams['text.usetex']=True
    from tqdm import tqdm
    debug = False

    analysis_dir = '/home/misiak/Analysis/NEUTRON'
    output_dir = '/'.join([analysis_dir, 'analysis_plots'])
    extension='png'
    
    h5type_list = [
        'data',
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

    if debug:
        h5type_list = [
            'data',
        ]
    
        stream_list = [
            'tg18l005',
        ]

   
    for stream in tqdm(stream_list):
        
        for h5type in h5type_list:
            h5_path = '/'.join([analysis_dir, '{}_heat_calib.h5'.format(h5type)])

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
            
            fig_dict = xtalk_plots(stream, title, df_analysis)
            
            if debug:
                continue

            # saving all the figures
            save_dir = '/'.join([
                analysis_dir,
                "10keV_plots_calibrated",
                h5type,
            ])

            save_figure_dict(fig_dict, save_dir, extension=extension)