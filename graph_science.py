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


from pipeline_data_calibrated import (
    quenching,
    lindhard,
    energy_recoil,
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
)


from pipeline_data_science import (
    ionization_baseline_resolution,
    std_energy_ion,
    charge_conservation_threshold
)


def plot_chi2_vs_energy_pretty(title, df):

    channel_suffix = [
        'heat',
        'ionD',
    ]
    
    ax_titles =[
        'Heat',
        'Ion D',
    ]

    quality_cut = df['quality_cut'] & df['bulk_cut']
    
    if 'simulation' in df.columns:
        quality_cut = df['quality_cut'] & df['bulk_cut'] & df['trigger_cut']
    
    num = '{}: $\chi^2$ Cut'.format(title)
    
    fig, axes = plt.subplots(ncols=2, figsize=(6.3, 3.9),
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

    for i, ax in enumerate(fig.get_axes()):
        ax.set_xlim(10**-2, 10**5)
        ax.set_ylim(10**1, 10**9)
        ax.legend()            

    return fig


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
        figsize=(6.3, 3.9),
    )
    ax_ecei.set_title('{} : band cut ecei'.format(title))
    fig_band_quenching, ax_qu = plt.subplots(
        num = '{} : band cut quenching'.format(title),
        figsize=(6.3, 3.9),
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

    
    # ### ZOOMIN PLOT

    # from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    # axins = zoomed_inset_axes(ax_ecei, 2, loc=2) # zoom-factor: 2.5, location: upper-left

    # axins.plot(
    #     ec_gamma,
    #     ei_gamma,
    #     label='gamma band',
    #     color='coral',
    #     path_effects=cartoon
    # )

    # x1, x2, y1, y2 = -0.5, 12, -0.5, 12 # specify the limits
    # axins.set_xlim(x1, x2) # apply the x-limits
    # axins.set_ylim(y1, y2) # apply the y-limits
  
    # axins.xaxis.set_visible('False')
    # axins.yaxis.set_visible('False')
    
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # mark_inset(ax_ecei, axins, loc1=2, loc2=4, fc="none", ec="0.5")

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
        figsize=(6.3, 3.9),
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


def science_plots(
        title,
        df_analysis,
        close_all=True
    ):
    
    if close_all:
        plt.close('all')
        
    fig_dict = dict()
    
    ### plot chi2
    fig_chi2_science = plot_chi2_vs_energy_pretty(title, df_analysis)
    fig_dict['chi2_science'] = fig_chi2_science
    
    ### plot charge conservation
    fig_charge = charge_conservation(title, df_analysis)
    fig_dict['charge_conservation'] = fig_charge

    ### plot fig cut
    fig_fid = fid_cut_plot(title, df_analysis)
    fig_dict['fid_cut'] = fig_fid

    ### plot band cut
    fig_ecei, fig_quenching = band_cut_plots(title, df_analysis)
    fig_dict['band_cut_ecei'] = fig_ecei
    fig_dict['band_cut_quenching'] = fig_quenching
    
    return fig_dict


if __name__ == '__main__':
    
    plt.close('all')
    plt.rcParams['text.usetex']=True
    plt.rcParams['font.size']=9
    plt.rcParams['lines.linewidth']=1
    from tqdm import tqdm
    debug = True

    analysis_dir = '/home/misiak/Analysis/neutron_background'
    output_dir = '/'.join([analysis_dir, 'analysis_plots'])
    extension='png'
    
    h5type_list = [
        'data',
        'simu'
    ]
    
    source_list = [
        'Calibration',
        'Background'
    ]

    simulation_list = [
        'flat_ER',
        'flat_NR',
        'line_1keV',
        'line_10keV',
    ]    
    
    if debug:
        h5type_list = [
            'data',
        ]
        
        source_list = [
            'Background',
        ]       

        simulation_list = [
            'flat_ER',
        ]        
    
    for source in tqdm(source_list):
        
        for h5type in h5type_list:
        
            h5_path = '/'.join([analysis_dir, '{}_science.h5'.format(h5type)])
            
            if h5type == 'simu':
                for simulation in simulation_list:
                    
                    df_analysis = pd.read_hdf(
                        h5_path,
                        key='df',
                        where=(
                            'source = "{0}"'
                            '& simulation = "{1}"'
                        ).format(source, simulation)
                    )
                    
                    title = (
                        '{0} {1} {2}'
                    ).format(h5type, simulation, source).replace('_', ' ') 
                    
                    fig_dict = science_plots(title, df_analysis)
                    
                    if debug:
                        continue
                    
                    # saving all the figures
                    save_dir = '/'.join([
                        output_dir,
                        source,
                        simulation
                    ])
                    
                    save_figure_dict(fig_dict, save_dir, extension=extension)
                    
            else:
                
                df_analysis = pd.read_hdf(
                    h5_path,
                    key='df',
                    where=(
                        'source = "{0}"'
                    ).format(source)
                )
                
                title = (
                    '{0} {1}'
                ).format(h5type, source).replace('_', ' ') 
                
                fig_dict = science_plots(title, df_analysis)
                
                if debug:
                    continue
                
                # saving all the figures
                save_dir = '/'.join([
                    output_dir,
                    source,
                    h5type,
                ])
                
                save_figure_dict(fig_dict, save_dir, extension=extension)

