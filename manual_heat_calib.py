#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:17:33 2020

@author: misiak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
from tqdm import tqdm

from plot_addon import (
    save_figure_dict
)


def calibration_stream(stream, save_fig=True):

    h5_path = '/home/misiak/Analysis/NEUTRON/data_ion_calib.h5'.format(stream)
    selection_path = '/home/misiak/Analysis/NEUTRON/{0}_10kev_selection.npy'.format(stream)
    ind = np.load(selection_path)
    
    df_analysis = pd.read_hdf(
        h5_path,
        key='df',
        where=(
            'stream = "{0}"'
        ).format(stream)
    )
    
    df_quality = df_analysis[df_analysis.quality_cut]

    df_reset = df_quality.reset_index(drop=True)
    selection_cut = np.in1d(np.array(df_reset.index), ind, assume_unique=True)
    df_10kev = df_quality[selection_cut]

    heat = df_10kev.energy_adu_heat
    ion = df_10kev.energy_ion_total
    
    ### auto calibration
    def func(beta, x):
        y = beta[0] * x + beta[1]
        return y
    
    data = RealData(heat, ion, 20, 1)
    model = Model(func)
       
    odr = ODR(data, model, [1,1])
    
    odr.set_job(fit_type=0)
    output = odr.run()
    
    beta = output.beta
    peak_trapped = -beta[1]/beta[0]
    peak_10kevee = (10.37-beta[1])/beta[0]
    
    ### Figure
    if save_fig:
        plt.close('all')
        fig, ax = plt.subplots()
        
        ax.plot(
            df_quality.energy_adu_heat,
            df_quality.energy_ion_total,
            ls='none',
            marker='.',
            color='b',
            alpha=0.3
        )
        
        ax.plot(
            heat,
            ion,
            ls='none',
            marker='.',
            color='r',
        )
        
        ax.plot(
            [peak_trapped, peak_10kevee],
            [0, 10.37],
            marker='o',
            color='k'
        )
        
        ax.grid()
        ax.set_ylabel('Energy Ionization Total [keV]')
        ax.set_xlabel('Energy Heat [ADU]')
        
        ax.set_xlim(-peak_10kevee*0.1, peak_10kevee*1.1)
        ax.set_ylim(-1, 11)
        
        fig_dict = {
            "{}_10kev_heat_calib".format(stream): fig
        }
        
        save_dir = '/home/misiak/Analysis/NEUTRON/10kev_heat_calib'
        save_figure_dict(fig_dict, save_dir, extension='png')
    
    return peak_10kevee

    
if __name__ == "__main__":
    
    stream_name_list = [
        'tg18l005',
        'tg27l000',
        'tg28l000',
        'tg17l007',
        'tg19l010',
        'tg20l000',
        'tg21l000'
    ]
    
  
    calibration_dict = dict()
    for stream in stream_name_list:
        calib = calibration_stream(stream)
        calibration_dict[stream] = calib
       
    # mean_calib = np.mean(calib_array, axis=0)
    # print("Abs. value of the Position 10keV peak")
    # print("[A,B,C,D]")
    # print(mean_calib)
    
    print(calibration_dict)
