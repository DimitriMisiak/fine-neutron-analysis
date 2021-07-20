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

def calibration_stream(stream):

    h5_path = '/home/misiak/Analysis/NEUTRON/data_xtalk.h5'

    df_analysis = pd.read_hdf(
        h5_path,
        key='df',
        where=(
            'stream = "{0}"'
        ).format(stream)
    )
    
    df_quality = df_analysis[df_analysis.quality_cut]
    
    heat = df_quality.energy_adu_heat
    ion_a = df_quality.energy_adu_corr_ionA
    ion_b = df_quality.energy_adu_corr_ionB
    ion_c = df_quality.energy_adu_corr_ionC
    ion_d = df_quality.energy_adu_corr_ionD
    
    ion_tot = ion_a + ion_b - ion_c - ion_d
    ion_cons = ion_a + ion_b + ion_c + ion_d
    
    calib_selection = (
        (df_quality.energy_adu_heat > 500)
        & (abs(ion_tot) > 100)
        & (abs(ion_tot) < 125)
    )
    
    df_calib = df_quality[calib_selection]
    
    heat = df_calib.energy_adu_heat
    ion_a = df_calib.energy_adu_corr_ionA
    ion_b = df_calib.energy_adu_corr_ionB
    ion_c = df_calib.energy_adu_corr_ionC
    ion_d = df_calib.energy_adu_corr_ionD



    def func(beta, x):
        y = (-beta[1]/beta[0])*x+beta[1]
        return y
    
    ### Calibration A & B
    data = RealData(ion_b, ion_a, 5, 5)
    model = Model(func)
       
    odr = ODR(data, model, [1,1])
    
    odr.set_job(fit_type=0)
    output = odr.run()
    
    peakB, peakA = output.beta

    ### Calibration C,D
    data = RealData(ion_d, ion_c, 5, 5)
    model = Model(func)
       
    odr = ODR(data, model, [1,1])
    
    odr.set_job(fit_type=0)
    output = odr.run()
    
    peakD, peakC = output.beta
    
    return peakA, peakB, peakC, peakD

    
if __name__ == "__main__":

    stream_list = [
        'tg18l005',
        'tg27l000',
        'tg28l000',
        'tg17l007',
        'tg19l010',
        'tg20l000',
        'tg21l000'
    ]
    
    calib_list = []
    for stream in stream_list:
        calib = calibration_stream(stream)
        calib_list.append(calib)
        
    calib_array = np.abs(calib_list)
       
    mean_calib = np.mean(calib_array, axis=0)
    print("Abs. value of the Position 10keV peak")
    print("[A,B,C,D]")
    print(mean_calib)
    

