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
    basic_corner,
    save_figure_dict
)

# from batch_pipeline_data_quality import (
#     ion_chi2_threshold_function,
#     heat_chi2_threshold_function,
#     quality_parameters,
# )

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
        ax.set_ylim(-70, 70)
        ax.set_xlim(-70, 70)
        
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
        ax.set_ylim(-70, 70)
        ax.set_xlim(-70, 70)
    return fig_cross


def xtalk_plots(
        stream,
        title,
        df_analysis,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
    
    fig_dict = dict()

    ### crosstalk correction
    fig_cross = crosstalk_correction(title, df_analysis)
    fig_dict[stream+'crosstalk_correction'] = fig_cross

    ## nodecor
    ### crosstalk correction
    fig_cross_nodecor = nodecor_crosstalk_correction(title, df_analysis)
    fig_dict[stream+'nodecor_crosstalk_correction'] = fig_cross_nodecor

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

    if debug:
        h5type_list = [
            'data',
        ]
    
        stream_list = [
            'tg18l005',
        ]


    simulation_list = [
        'flat_ER',
        'flat_NR',
        'line_1keV',
        'line_10keV',
    ]
   
    for stream in tqdm(stream_list):
        
        for h5type in h5type_list:
            h5_path = '/'.join([analysis_dir, '{}_xtalk.h5'.format(h5type)])
            
            if h5type == 'simu':
                
                if debug:
                    continue
            
                for simulation in simulation_list:
    
                    df_analysis = pd.read_hdf(
                        h5_path,
                        key='df',
                        where=(
                            'stream = "{0}"'
                            '& simulation = "{1}"'
                        ).format(stream, simulation)
                    )
                    
                    source = df_analysis['source'].unique()[0]
                    title = (
                        '{0} {1} {2} {3}'
                    ).format(stream, h5type, simulation, source).replace('_', ' ') 
                    
                    fig_dict = xtalk_plots(stream, title, df_analysis)
                    
                    # saving all the figures
                    save_dir = '/'.join([
                        output_dir,
                        stream,
                        simulation
                    ])
                    
                    save_figure_dict(fig_dict, save_dir, extension=extension)
                    
            else:
                
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
                    output_dir,
                    stream,
                    h5type,
                ])
                
                save_figure_dict(fig_dict, save_dir, extension=extension)
