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
        df_analysis[quality_cut]['energy_adu_heat'], # heat in adu, as not calibrated yet !
        df_analysis[quality_cut]['energy_ion_total'],
        label='quality events',
        ls='none',
        marker='.',
        color='b',
        alpha=0.3
    )
    # #guide for 10keV
    # ax.plot(
    #     [10.37/(1+delta_volt/3), 10.37],
    #     [0, 10.37], 
    #     zorder=-20, lw=10,
    #     color='gold', label='10keV band (theory)'
    # )
    ax.grid()
    ax.set_xlim(-50, 1500)
    ax.set_ylim(-2, 13)
    ax.set_ylabel('Total Ionization Energy A+B+C+D [keV]')
    ax.set_xlabel('Heat Energy [keV]')
    fig_10kev.tight_layout()
    
    return fig_10kev


def selection_plots(
        stream,
        title,
        df_analysis,
        close_all=True,
        analysis_dir=''
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
    fig_dict[stream + '_plot_10kev'] = fig_10kev

    ax = fig_10kev.get_axes()[0]
    line = ax.lines[0]

    from graphic_selection import Data_Selector
    output_file = '/'.join([analysis_dir, '{}_10kev_selection.npy'.format(stream)])
    proceed_func = lambda x: np.save(output_file, x)
    ds = Data_Selector(ax, line, proceed_func=proceed_func)
    
    plt.show()
    
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
            h5_path = '/'.join([analysis_dir, '{}_ion_calib.h5'.format(h5type)])
                
            h5_path = '/'.join([analysis_dir, '{}_ion_calib.h5'.format(h5type)])
                
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
            
            fig_dict = selection_plots(stream, title, df_analysis, analysis_dir=analysis_dir)
