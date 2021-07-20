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


def offset_pretty(title, df, stream):
    
    quality_cut = df['quality_cut']
    
    offset_ion = abs(df['offset_ionD'])
    energy_ion = df['energy_adu_ionD']
    
    # Init figure
    num = '{}: Ion. Sensitivity vs Ion. Offset'.format(title)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), num=num)
    
    ax.plot(
        offset_ion,
        energy_ion,
        ls='none',
        marker='.',
        color='k',
        alpha=0.3,
        label='All events'
    )

    ax.plot(
        offset_ion[quality_cut],
        energy_ion[quality_cut],
        ls='none',
        marker='.',
        color='slateblue',
        alpha=0.3,
        label='Quality events'
    )
    
    ax.axvspan(
        quality_parameters["offset_ion_threshold"],
        35000,
        label='Region discarded by the Offset Cut',
        color='r',
        alpha=0.3
    )
    
    ax.set_ylabel('Amplitude Ionization D [ADU]')
    ax.set_xlabel('Absolute Value of the Offset Ionization D [ADU]')
    
    ax.set_ylim(-10, 80)
    ax.set_xlim(-1500, 34000)
    
    ax.legend(loc='upper right', framealpha=1)
    ax.grid()
 
    fig.text(0.5, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    fig.tight_layout(rect=(0, 0, 1, 0.98))    
    
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
    
    num = '{}: All $\chi^2$ Cut'.format(title)
    
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
        quality_parameters['ion_chi2_threshold'],
        x_data
    )
    cut_heat = heat_chi2_threshold_function(
        quality_parameters[stream]['heat_chi2_threshold'],
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


def plot_chi2_vs_energy_pretty(title, df, stream):

    channel_suffix = [
        'heat',
        'ionD',
    ]
    
    ax_titles =[
        'Heat',
        'Ion D',
    ]

    quality_cut = df['quality_cut']
    
    num = '{}: $\chi^2$ Cut'.format(title)
    
    fig, axes = plt.subplots(ncols=2, figsize=(12, 7),
                             num=num
    )

    for suffix, ax, title in zip(channel_suffix, axes, ax_titles):
        
        xdata = abs( df['energy_adu_{}'.format(suffix)] )
        ydata = df['chi2_{}'.format(suffix)]
        
        nsamples = xdata.shape[0]
        
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
        
    fig.tight_layout(rect=(0, 0, 1, 0.98))


    # plotting the limit
    x_data = 10**np.linspace(-2, 5, int(1e4))
    cut_ion = ion_chi2_threshold_function(
        quality_parameters['ion_chi2_threshold'],
        x_data
    )
    cut_heat = heat_chi2_threshold_function(
        quality_parameters[stream]['heat_chi2_threshold'],
        x_data)    
    for i, ax in enumerate(fig.get_axes()):
        if i == 0:
            ax.plot(x_data, cut_heat, lw=1, color='k', label='quality cut')
        else:
            ax.plot(x_data, cut_ion, lw=1, color='k', label='quality cut')
        ax.set_xlim(10**-2, 10**5)
        ax.set_ylim(10**1, 10**9)
        ax.legend()            

    return fig


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


def quality_plots(
        stream,
        title,
        df_analysis,
        simu_flag=False,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
        
    fig_dict = dict()
    
    ### temporal multi
    fig_temp = temporal_plot(title, df_analysis)
    fig_dict['temporal_multi'] = fig_temp
    
    ### temporal heat
    fig_temp_heat = temporal_plot_heat_only(title, df_analysis)
    fig_dict['temporal_heat'] = fig_temp_heat
    
    ### offset
    fig_offset = offset_pretty(title, df_analysis, stream)
    fig_dict['offset_plot'] = fig_offset
    
    ### chi2 plot
    fig_chi2 = plot_chi2_vs_energy(title, df_analysis, stream)
    fig_dict['chi2_plot'] = fig_chi2

    ### chi2 plot
    fig_chi2_pretty = plot_chi2_vs_energy_pretty(title, df_analysis, stream)
    fig_dict['chi2_plot_pretty'] = fig_chi2_pretty

    ### trigger plot
    if simu_flag:
        fig_chi2 = plot_chi2_vs_energy(title, df_analysis, stream)
        fig_dict['chi2_plot'] = fig_chi2

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
            h5_path = '/'.join([analysis_dir, '{}_quality.h5'.format(h5type)])
            
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
                    
                    fig_dict = quality_plots(stream, title, df_analysis, simu_flag=True)
                    
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
                
                fig_dict = quality_plots(stream, title, df_analysis, simu_flag=False)
                
                if debug:
                    continue
                
                # saving all the figures
                save_dir = '/'.join([
                    output_dir,
                    stream,
                    h5type,
                ])
                
                save_figure_dict(fig_dict, save_dir, extension=extension)
