#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

calibration_parameters = {
    "tg18l005": {
        "position_10kev_line_heat": 1177,
    },
    "tg27l000": {
        "position_10kev_line_heat": 1293,
    },
    "tg28l000": {
        "position_10kev_line_heat": 1291,
    },
    "tg17l007": {
        "position_10kev_line_heat": 1165,
    },
    "tg19l010": {
        "position_10kev_line_heat": 1199,
    },
    "tg20l000": {
        "position_10kev_line_heat": 1197,
    },
    "tg21l000": {
        "position_10kev_line_heat": 1235,
    },
    "position_10kev_line_ionA": -56,
    "position_10kev_line_ionB": -54,
    "position_10kev_line_ionC": 57,
    "position_10kev_line_ionD": 56,
}

ion_channel_labels = ('A', 'B', 'C', 'D')

crosstalk_correction_matrix = np.array([
        [1, -0.052, 0, 0],
        [-0.03, 1, 0, 0],
        [-0.012, 0.001, 1, -0.025],
        [0, 0, -0.03, 1]
])

nodecor_crosstalk_correction_matrix = np.array([
        [1, -0.044, 0, 0],
        [-0.03, 1, 0, 0],
        [-0.025, 0.001, 1, -0.031],
        [0, 0, -0.03, 1]
])


def crosstalk_correction(df):
    """ 
    Create new columns for the cross-talk corrected ionization channels.
    """        
    
    ion_energy = df[[
        'energy_adu_ionA',
        'energy_adu_ionB',
        'energy_adu_ionC',
        'energy_adu_ionD'
    ]]
    
    
    corr_ion_energy = np.dot(crosstalk_correction_matrix, ion_energy.T)
    energy_corr_cname_list = [
        'energy_adu_corr_ionA',
        'energy_adu_corr_ionB',
        'energy_adu_corr_ionC',
        'energy_adu_corr_ionD'
    ]
    for i, col in enumerate(energy_corr_cname_list):
        df[col] = corr_ion_energy[i]
    
    return None


def nodecor_crosstalk_correction(df):
    """ 
    Create new columns for the cross-talk corrected ionization channels.
    """        
    
    ion_energy = df[[
        'energy_adu_nodecor_ionA',
        'energy_adu_nodecor_ionB',
        'energy_adu_nodecor_ionC',
        'energy_adu_nodecor_ionD'
    ]]
    
    
    corr_ion_energy = np.dot(nodecor_crosstalk_correction_matrix, ion_energy.T)
    energy_corr_cname_list = [
        'energy_adu_corr_nodecor_ionA',
        'energy_adu_corr_nodecor_ionB',
        'energy_adu_corr_nodecor_ionC',
        'energy_adu_corr_nodecor_ionD'
    ]
    for i, col in enumerate(energy_corr_cname_list):
        df[col] = corr_ion_energy[i]
    
    return None


def calibration_heat(stream, df):
    """ 
    Create a new column for the calibrated energy of heat channel.
    """       
    calibration_factor = (
        10.37 / calibration_parameters[stream]['position_10kev_line_heat']
    )
    
    df['energy_heat'] = (
        df['energy_adu_heat'] * calibration_factor
    )
    
    return None


def calibration_ion(stream, df):
    """ 
    Create new columns for the calibrated energy of the ionization channels.
    """         
    for suffix in ion_channel_labels:
        
        position_cname = 'position_10kev_line_ion{}'.format(suffix)
        energy_adu_cname ='energy_adu_corr_ion{}'.format(suffix)
        energy_cname = 'energy_ion{}'.format(suffix)
        
        calibration_factor = (
            10.37 / calibration_parameters[position_cname]
        )
        
        df[energy_cname] = (
            df[energy_adu_cname] * calibration_factor
        )
        
    return None


def nodecor_calibration_ion(stream, df):
    """ 
    Create new columns for the calibrated energy of the ionization channels.
    """         
    for suffix in ion_channel_labels:
        
        position_cname = 'position_10kev_line_ion{}'.format(suffix)
        energy_adu_cname ='energy_adu_corr_nodecor_ion{}'.format(suffix)
        energy_cname = 'energy_nodecor_ion{}'.format(suffix)
        
        calibration_factor = (
            10.37 / calibration_parameters[position_cname]
        )
        
        df[energy_cname] = (
            df[energy_adu_cname] * calibration_factor
        )
        
    return None


def virtual_channels(df):
    """ 
    Create new columns for "virtual channels" which are combinations
    of the ionization channels:
        - energy_ion_total: A+B+C+D
        - energy_ion_collect: B+D
        - energy_ion_guard: A+C
    """    

    df['energy_ion_total'] = (
        df['energy_ionA'] + df['energy_ionB']
        + df['energy_ionC'] + df['energy_ionD']
    ) / 2
    
    df['energy_ion_bulk'] = (
        df['energy_ionB'] + df['energy_ionD']
    ) / 2
    
    df['energy_ion_guard'] = (
        df['energy_ionA'] + df['energy_ionC']
    ) / 2
    
    #hard coded polarization
    df['energy_ion_conservation'] = (
        - df['energy_ionA'] - df['energy_ionB']
        + df['energy_ionC'] + df['energy_ionD']
    ) / 2

    return None


def nodecor_virtual_channels(df):
    """ 
    Create new columns for "virtual channels" which are combinations
    of the ionization channels:
        - energy_nodecor_ion_total: A+B+C+D
        - energy_nodecor_ion_collect: B+D
        - energy_nodecor_ion_guard: A+C
    """    

    df['energy_nodecor_ion_total'] = (
        df['energy_nodecor_ionA'] + df['energy_nodecor_ionB']
        + df['energy_nodecor_ionC'] + df['energy_nodecor_ionD']
    ) / 2
    
    df['energy_nodecor_ion_bulk'] = (
        df['energy_nodecor_ionB'] + df['energy_nodecor_ionD']
    ) / 2
    
    df['energy_nodecor_ion_guard'] = (
        df['energy_nodecor_ionA'] + df['energy_nodecor_ionC']
    ) / 2
    
    #hard coded polarization
    df['energy_nodecor_ion_conservation'] = (
        - df['energy_nodecor_ionA'] - df['energy_nodecor_ionB']
        + df['energy_nodecor_ionC'] + df['energy_nodecor_ionD']
    ) / 2

    return None


def energy_recoil(ec, ei, V):
#    coeff = 1.6e-19 * V / 3
    coeff = V / 3
    return ec*(1+coeff) - ei*coeff


def quenching(ec, ei, V):
    er = energy_recoil(ec, ei, V)
    return ei/er


def lindhard(er):
    
    A = 72.63
    Z = 32
    
    k = 0.133 * Z**(2./3) * A**(-1./2)
    epsilon = 11.5 * er * Z**(-7./3)
    g = 3 * epsilon**0.15 + 0.7 * epsilon**0.6 + epsilon  
    
    Q = k*g/(1+k*g)
    
    return Q


def energy_heat_from_er_and_quenching(er, Q, V): 
    return er * (1 + Q*V/3) / (1 + V/3)


def energy_ion_from_er_and_quenching(er, Q, V=None): 
    return er * Q


def physical_quantities(df, voltage=2):
    """ 
    Create new columns for the recoil energy and quenching.
    """    
    for suffix in ('_total', '_bulk', '_guard'):
        
        energy_ion_cname = 'energy_ion{}'.format(suffix)
        recoil_energy_cname = 'recoil_energy{}'.format(suffix)
        quenching_cname = 'quenching{}'.format(suffix)
        
        df[recoil_energy_cname] = energy_recoil(
            df['energy_heat'],
            df[energy_ion_cname],
            voltage
        )
    
        df[quenching_cname] = quenching(
            df['energy_heat'],
            df[energy_ion_cname],
            voltage
        )

    return None


def pipeline_calibration(stream, df_quality):
    """
    Common to all types of events: data, noise, simulation
    """
    df_calibrated = pd.DataFrame()
    for col in df_quality.columns:
        df_calibrated[col] = df_quality[col]
        
    calibration_heat(stream, df_quality)    
        
    crosstalk_correction(df_quality)
    calibration_ion(stream, df_quality)
    
    #nodecor
    nodecor_crosstalk_correction(df_quality)
    nodecor_calibration_ion(stream, df_quality)

    virtual_channels(df_quality)
    physical_quantities(df_quality)    
    nodecor_virtual_channels(df_quality)

    return df_quality


def hdf5_calibration(fine_hdf5_path, output_hdf5_path):

    stream_list = pd.read_hdf(
        fine_hdf5_path,
        key='df',
        columns=['stream',]
    )['stream'].unique()
    
    # initializing the HDFstore (overwriting, be careful !)
    pd.DataFrame().to_hdf(
        output_hdf5_path,
        key='df', mode='w', format='table'
    )
    
    for stream in tqdm(stream_list):

        df_quality = pd.read_hdf(
            fine_hdf5_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        df_calibrated = pipeline_calibration(stream, df_quality)

        df_calibrated.to_hdf(
            output_hdf5_path,
            key='df',
            mode='a',
            format='table',
            append=True,
            min_itemsize=11,
            data_columns=True
        ) 

    return None
    

if __name__ == "__main__":

    analysis_dir = '/home/misiak/Analysis/neutron_background'
    
    ### DATA
    fine_data_path = '/'.join([analysis_dir, 'data_quality.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_calibrated.h5'])   
    hdf5_calibration(
        fine_data_path,
        output_data_path,
    )

    ### NOISE
    fine_noise_path = '/'.join([analysis_dir, 'noise_quality.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_calibrated.h5'])   
    
    hdf5_calibration(
        fine_noise_path,
        output_noise_path,
    )
    
    ### SIMULATION
    fine_simu_path = '/'.join([analysis_dir, 'simu_quality.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_calibrated.h5'])
    hdf5_calibration(
        fine_simu_path,
        output_simu_path,
    )
    