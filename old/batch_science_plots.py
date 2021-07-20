#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:35:22 2020

@author: misiak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.close('all')
plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1

from plot_addon import (
    LegendTitle,
    custom_autoscale,
    ax_hist,
    basic_corner,
    save_figure_dict
)

stream = 'tg09l000'


#%%
# =============================================================================
# baseline resolution
# =============================================================================

def baseline_resolution(dfn_quality):
    resolution_dict = dict()
    
    fig, axes = plt.subplots(
        nrows=5,
        sharex=True,
        figsize=(6.3, 6)
    )
    
    bins = 10
    
    for i, suffix in enumerate(['heat', 'ionA', 'ionB', 'ionC', 'ionD']):
        
        ax = axes[i]
        x_data = dfn_quality['energy_{}'.format(suffix)] 
        
        resolution_dict[suffix] = np.std(x_data)
        
        bin_edges = np.histogram_bin_edges(x_data, bins=bins)
        ax_hist(ax,
                bin_edges,
                x_data,
                suffix,
        )
        ax.legend(
            title='$\sigma_{0}={1:.3f}$ keV'.format(suffix, resolution_dict[suffix]),
            loc='upper left'
        )

    return fig, resolution_dict

#%%
# =============================================================================
# Histograms channels
# =============================================================================

def res_main(df_selection):

    ax_tuples = ((0, 0), (0, 1))
     
    quant_dict = dict()
    
    channel_suffix = [
        'ion_total',
        'heat',
    ]    
    
    ax_titles =[
        'Ion Total',
        'Heat',
    ]
    
    num = '{} : Resolution Peak'.format(stream)
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(11.69, 3.27),
        num=num,
        squeeze=False,
    )
    
    for suffix, tupl, label in zip(channel_suffix, ax_tuples, ax_titles):
        
        xdata = df_selection['energy_{}'.format(suffix)]

        if suffix == 'ion_total':
            blob_cut = (xdata >= 10.37)
        elif suffix == 'heat':
            blob_cut = (xdata >= 10.37)
            
        xdata_alt = xdata[blob_cut]
        
        sup_alt = np.quantile(xdata_alt, [0.68,])[0]
        quant_dict[suffix+"_alt"] = sup_alt
        
        if suffix == 'ion_total':
            blob_cut = (xdata > 9)
        elif suffix == 'heat':
            blob_cut = (xdata > 8)
            
        xdata = xdata[blob_cut]
        
        med, inf, sup = np.quantile(xdata, [0.5, 0.16, 0.84])
        quant_dict[suffix] = (med, inf, sup)


        
        ax = axes[tupl]
        
        bin_edges = np.histogram_bin_edges(xdata, bins=100)
    
        ax_hist(ax, bin_edges, xdata,
                'All events', color='coral')
    
    
        ax.axvline(med, lw=2, ls='--', color='k')
        ax.axvline(inf, lw=1, ls='--', color='k')
        ax.axvline(sup, lw=1, ls='--', color='k')

        ax.axvline(10.37, lw=2, ls='--', color='b')
        ax.axvline(sup_alt, lw=1, ls='--', color='b')
    

        ax.legend(
            title="Peak=${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$".format(med, med-inf, sup-med),
            loc=2
        )
        ax.set_title(label.replace('_', ' '))
        
    
    fig.text(0.5, 0.98, num,
              horizontalalignment='center',
              verticalalignment='center',
              bbox=dict(facecolor='lime', alpha=0.5))
    
    
    # resize the plots
    fig.get_axes()[0].set_xlim(-200, 2000)
    for i, ax in enumerate(fig.get_axes()):
            ax.set_xlim(-2, 12)    
     
    # fig.delaxes(axes[0,0])    
    fig.tight_layout()

    return fig, quant_dict

#%%
# =============================================================================
# EICI PLOT
# =============================================================================

def eici_plot(df_quality, df_selection, df_trapped, v_bias):

    delta_volt = v_bias
    
    fig_eiec, ax = plt.subplots(
        num='Tot Ion vs Heat',
        figsize=(6.3, 3.9)
    )
    ax.set_title('{} : 10keV events'.format(stream))
    ax.plot(
        df_quality.energy_heat,
        df_quality.energy_ion_total,
        label='Quality events',
        ls='none',
        marker='.',
        color='b',
        alpha=0.3
    )
    ax.plot(
        df_selection.energy_heat,
        df_selection.energy_ion_total,
        label='10keV selected events',
        ls='none',
        marker='.',
        markersize=12,
        color='coral',
        alpha=0.5,
        zorder=-19
    )
    
    ax.plot(
        df_trapped.energy_heat,
        df_trapped.energy_ion_total,
        label='Trapped events',
        ls='none',
        marker='.',
        markersize=18,
        color='k',
        alpha=0.5,
        zorder=-19
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
    ax.legend()
    fig_eiec.tight_layout()

    return fig_eiec

#%%
# =============================================================================
# Ion vs Ion
# =============================================================================

def ion_vs_ion(df_selection, df_trapped, resolution_dict):
    # initializing pseudo-corner plot
    ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
    ax_discard = [(0, 1), (1, 2), (0, 2)]
    
    # chan_x = np.insert(run_tree.chan_veto, 0, run_tree.chan_collect[1])
    # chan_y = np.append(run_tree.chan_veto, run_tree.chan_collect[0])    
    chan_x = ['ionD', 'ionA', 'ionC']
    chan_y = ['ionA', 'ionC', 'ionB']
       
    num = '{} : Ion vs Ion'.format(stream)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.27, 8.27),
                              num=num, sharex='col', sharey='row')
    
    # actually plotting the data
    for atupl in ax_tuples:
        
        ax = axes[atupl]
        xind = chan_x[atupl[1]]
        yind = chan_y[atupl[0]]
    
        # energy_x = energy[:, xind]
        # energy_y = energy[:, yind]
        energy_x = df_selection['energy_{}'.format(xind)]
        energy_y = df_selection['energy_{}'.format(yind)]
    
        energy_x_trapped = df_trapped['energy_{}'.format(xind)]
        energy_y_trapped = df_trapped['energy_{}'.format(yind)]
    
        ax.plot(
                energy_x, energy_y,
                ls='none', marker='.', alpha=0.1, zorder=10, color='slateblue',
                label='Selected Events'
        )
    
        ax.plot(
                energy_x_trapped, energy_y_trapped,
                ls='none', marker='.', alpha=0.5, zorder=10, color='k',
                label='Trapped Events'
        )
            
        custom_autoscale(ax, energy_x, energy_y)
        
        ax.grid(alpha=0.3)
        
        # resolution vizualition
        try:
            res_y = resolution_dict[yind] * 10
            ax.axhspan(-res_y/2, res_y/2, alpha=0.3)
            
            res_x = resolution_dict[xind] * 10 
            ax.axvspan(-res_x/2, res_x/2, alpha=0.3)
        except:
            pass
        
        
        
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
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        
    return fig

#%%
# =============================================================================
# Histograms channels
# =============================================================================

def histogram_channels(df_selection):

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
    
    num = '{} : Quality Cut Histogram'.format(stream)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(11.69, 8.27),
        num=num
    )
    
    for suffix, tupl, label in zip(channel_suffix, ax_tuples, ax_titles):
        
        xdata = df_selection['energy_{}'.format(suffix)]
        
        ax = axes[tupl]
        
        bin_edges = np.histogram_bin_edges(xdata, bins=100)
    
        ax_hist(ax, bin_edges, xdata,
                'All events', color='coral')
    
        ax.legend(loc=2)
        ax.set_title(label.replace('_', ' '))
        
    
    fig.text(0.5, 0.98, num,
              horizontalalignment='center',
              verticalalignment='center',
              bbox=dict(facecolor='lime', alpha=0.5))
    
    
    # resize the plots
    fig.get_axes()[0].set_xlim(-200, 2000)
    for i, ax in enumerate(fig.get_axes()):
            ax.set_xlim(-2, 12)    
     
    fig.delaxes(axes[0,0])    
    fig.tight_layout()

    return fig

#%%
# =============================================================================
# charge conservation
# =============================================================================

def charge_conservation(df_selection):
        
    fig, axes = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(6.3, 3.9)
    )
    
    bins = 100
    
    # energy conservation
    cons_decor = df_selection['energy_ion_conservation']
    cons_nodecor = df_selection['energy_nodecor_ion_conservation']
    
    # DECORELLATION
    ax = axes[0]
    bin_edges = np.histogram_bin_edges(cons_decor, bins=bins)
    ax_hist(ax,
            bin_edges,
            cons_decor,
            'Decor',
            color='b'
    )
    # NODECOR
    ax = axes[1]
    bin_edges = np.histogram_bin_edges(cons_nodecor, bins=bins)
    ax_hist(ax,
            bin_edges,
            cons_nodecor,
            'Nodecor',
            color='r'
    )
    
    return fig

#%%
# =============================================================================
# TEST FIELD
# =============================================================================

def test_plot(df_selection, df_trapped):
    
    # initializing pseudo-corner plot
    ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
    ax_discard = [(0, 1), (1, 2), (0, 2)]
    
    # chan_x = np.insert(run_tree.chan_veto, 0, run_tree.chan_collect[1])
    # chan_y = np.append(run_tree.chan_veto, run_tree.chan_collect[0])    
    # chan_x = ['heat', 'ion_bulk', 'ion_guard']
    # chan_y = ['ion_bulk', 'ion_guard', 'nodecor_ion_conservation']
    
    chan_x = ['ion_total', 'ion_bulk', 'ion_guard']
    chan_y = ['ion_bulk', 'ion_guard', 'nodecor_ion_conservation']
    
    num = '{} : Testy'.format(stream)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.27, 8.27),
                              num=num, sharex='col', sharey='row')
    
    # actually plotting the data
    for atupl in ax_tuples:
        
        ax = axes[atupl]
        xind = chan_x[atupl[1]]
        yind = chan_y[atupl[0]]
    
        energy_x = df_selection['energy_{}'.format(xind)]
        energy_y = df_selection['energy_{}'.format(yind)]
    
        energy_x_trapped = df_trapped['energy_{}'.format(xind)]
        energy_y_trapped = df_trapped['energy_{}'.format(yind)]
    
        ax.plot(
                energy_x, energy_y,
                ls='none', marker='.', alpha=0.1, zorder=10, color='slateblue',
                label='Selected Events'
        )
    
        ax.plot(
                energy_x_trapped, energy_y_trapped,
                ls='none', marker='.', alpha=0.5, zorder=10, color='k',
                label='Trapped Events'
        )
            
        custom_autoscale(ax, energy_x, energy_y)
        
        ax.grid(alpha=0.3)
        
        # # resolution vizualition
        # try:
        #     res_y = resolution_dict[yind] * 10
        #     ax.axhspan(-res_y/2, res_y/2, alpha=0.3)
            
        #     res_x = resolution_dict[xind] * 10 
        #     ax.axvspan(-res_x/2, res_x/2, alpha=0.3)
        # except:
        #     pass
        
        
        
        if atupl == (0,0):
            ax.legend(loc='lower left', framealpha=1,
                      bbox_to_anchor=(1.05, 0.05), borderaxespad=0.,
            )
        
        if atupl[0] == 2:
            ax.set_xlabel(
                    'Energy {} [ADU]'.format(xind.replace('_', ' '))
            )
                
        if atupl[1] == 0:
            ax.set_ylabel(
                    'Energy {} [ADU]'.format(yind.replace('_', ' '))
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
    # for ax in axes:
    #     ax.set_xlim(-2, 12)
    #     ax.set_ylim(-2, 12)
    
    return fig

# =============================================================================
# ROUTINE
# =============================================================================

def science_plots(
        stream,
        df_quality,
        df_selection,
        dfn_quality,
        df_trapped,
        v_bias,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
    
    fig_dict = dict()

    fig, resolution_dict = baseline_resolution(
        dfn_quality
    )
    fig_dict['baseline_resolution_'+stream] = fig

    fig_dict['eiec_plot_'+stream] = eici_plot(
        df_quality, df_selection, df_trapped, v_bias
    )
    
    fig_dict['ion_vs_ion_'+stream] = ion_vs_ion(
        df_selection, df_trapped, resolution_dict
    )

    fig_dict['histogram_channels_'+stream] = histogram_channels(df_selection)


    fig_dict['charge_conservation_'+stream] = charge_conservation(df_selection)
    
    fig_dict['test_plots_'+stream] = test_plot(df_selection, df_trapped)
    
    return fig_dict


def batch_launcher(stream_list):
    
    analysis_dir = '/home/misiak/Analysis/RED80/{}'.format(stream_list[0])
    
    for stream in stream_list:
    
        ### noise file
        h5_noise_path = '/home/misiak/Analysis/RED80/{}/noise_heat_calib.h5'.format(stream)
        
        dfn_analysis = pd.read_hdf(
            h5_noise_path,
            key='df',
            where=(
                'stream = "{0}"'
            ).format(stream)
        )
        dfn_quality = dfn_analysis[dfn_analysis.quality_cut]
 
        fig, res = baseline_resolution(
            dfn_quality
        )
        plt.close('all')

        # special for REDN1
        res_tot = (
            res['ionA']**2 + res['ionB']**2 + res['ionC']**2 + res['ionD']**2
        )**0.5
        

        ### data file
        h5_path = '/home/misiak/Analysis/RED80/{}/data_heat_calib.h5'.format(stream)
        
        df_analysis = pd.read_hdf(
            h5_path,
            key='df',
            where=(
                'stream = "{0}"'
            ).format(stream)
        )
    
        df_quality = df_analysis[df_analysis.quality_cut]
        df_selection = df_analysis[df_analysis.selection_cut]
        
        incomplete_heat = ( df_selection.energy_heat < 10.73 - res['heat']*3 )
        incomplete_ion = ( df_selection.energy_ion_total < 10.37 - res_tot*3 )
        trapped_cut = ( incomplete_heat | incomplete_ion )
        df_trapped = df_selection[incomplete_heat | incomplete_ion]
        
        complete_cut = ~trapped_cut
        
        bulk_cut = (
            complete_cut
            & (df_selection.energy_ionA < res['ionA']*5)
            & (df_selection.energy_ionC < res['ionC']*5)
        )
        
        equator_cut = (
            complete_cut
            & (df_selection.energy_ionB < res['ionB']*5)
            & (df_selection.energy_ionD < res['ionD']*5)
        )
        
        top_cut = (
            complete_cut
            & (df_selection.energy_ionC < res['ionC']*5)
            & (df_selection.energy_ionD < res['ionD']*5)
        )
        
        bottom_cut = (
            complete_cut
            & (df_selection.energy_ionA < res['ionA']*5)
            & (df_selection.energy_ionB < res['ionB']*5)
        )
        
        fid_cut = (
            complete_cut
            & (df_selection.energy_ionB > 9)
            & (df_selection.energy_ionD > 9)
        ) 
        
        counts_dict = {
            "total" : (df_analysis.selection_cut).sum(),
            "incomplete" : trapped_cut.sum(),
            "complete" : complete_cut.sum(),
            "bulk": bulk_cut.sum(),
            "fid": fid_cut.sum(),
            "equator": equator_cut.sum(),
            "top": top_cut.sum(),
            "bottom": bottom_cut.sum(),
        }
        
        # for k,v in counts_dict.items():
        #     if k not in ('fid', 'incomplete'):
        #         continue
        #     print("Number of {} events = {}".format(k, v))
        #     print("Percentage of {} events = {}".format(k, v/counts_dict['total']))
            
        # return (
        #     counts_dict['incomplete']/counts_dict['total'],
        #     counts_dict['bulk']/counts_dict['complete'],
        #     # counts_dict['fid']/counts_dict['complete'],
        #     counts_dict['bulk']/counts_dict['total'],
        #     counts_dict['tot
        # )
        
        # return counts_dict
        
    
       
        ### polarization
        polar_list = list()
        for lab in 'ABCD':
            polar = df_selection['polar_{}'.format(lab)].unique()[0]
            polar_list.append(polar)
            
        v_bias = np.max(polar_list) - np.min(polar_list)          
    
        
        ## PLOTS
        
        fig_dict = science_plots(
            stream,
            df_quality,
            df_selection,
            dfn_quality,
            df_trapped,
            v_bias,
            close_all=True
        )
        
        # saving all the figures
        save_dir = '/'.join([
            os.path.dirname(analysis_dir),
            "science_plots",
        ])
        
        save_figure_dict(fig_dict, save_dir, extension='png')


if __name__ == '__main__':
    stream_name_list = np.loadtxt(
        fname='/home/misiak/projects/fine_red80_analysis/stream_list.txt',
        dtype=str
    )

    from tqdm import tqdm
    for stream in tqdm(stream_name_list):
        batch_launcher([stream,])
        
