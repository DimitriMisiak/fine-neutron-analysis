#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# defining calibration parameters for each stream
# =============================================================================

# stream_name_list = np.loadtxt(
#     fname='/home/misiak/projects/fine_red80_analysis/stream_list.txt',
#     dtype=str
# )

calibration_parameters = dict()

calibration_parameters = {
    'tg18l005': 1176.0834163620705,
    'tg27l000': 1292.8822141898477,
    'tg28l000': 1292.8144790956226,
    'tg17l007': 1161.7974571908512,
    'tg19l010': 1197.0744026496347,
    'tg20l000': 1196.4545336077013,
    'tg21l000': 1234.4862677363194
}

# =============================================================================
# Function as usuam
# =============================================================================
ion_channel_labels = ('A', 'B', 'C', 'D')

def selection_cut(stream, df):
    
    selection_path = '/home/misiak/Analysis/NEUTRON/{0}_10kev_selection.npy'.format(stream)
    ind = np.load(selection_path)

    df_quality = df[df.quality_cut]
    
    df_reset = df_quality.reset_index(drop=True)
    
    selection_cut = np.in1d(np.array(df_reset.index), ind, assume_unique=True)
    
    df_10kev = df_quality[selection_cut]
    
    selection_cut = np.in1d(
        np.array(df.index),
        df_10kev.index,
        assume_unique=True
    )
    
    df['selection_cut'] = selection_cut
    
    return None


def calibration_heat(stream, df):
    """ 
    Create a new column for the calibrated energy of heat channel.
    """       
    calibration_factor = (
        10.37 / calibration_parameters[stream]
    )
    
    df['energy_heat'] = (
        df['energy_adu_heat'] * calibration_factor
    )
    
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

def recoil_energy_from_er_and_quenching(er, Q, V): 
    return er * (1 + Q*V/3) / (1 + V/3)

def energy_ion_from_er_and_quenching(er, Q, V=None): 
    return er * Q


def physical_quantities(df, voltage=2):
    """ 
    Create new columns for the recoil energy and quenching.
    """
    
    voltage = abs(df['polar_B'].unique()[0] - df['polar_D'].unique()[0])
    
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

def pipeline_heat_calib(stream, df_quality, selection_flag=False):
    """
    Common to all types of events: data, noise, simulation
    """
    df_calibrated = pd.DataFrame()
    for col in df_quality.columns:
        df_calibrated[col] = df_quality[col]
    
    if selection_flag:
        selection_cut(stream, df_quality)    
        
    calibration_heat(stream, df_quality)
    
    physical_quantities(df_quality)    

    return df_quality


def hdf5_heat_calib(fine_hdf5_path, output_hdf5_path, selection_flag=False):

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
        
        df_heat_calib = pipeline_heat_calib(stream, df_quality, selection_flag)

        df_heat_calib.to_hdf(
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

    analysis_dir = '/home/misiak/Analysis/NEUTRON'
    
    ### DATA
    fine_data_path = '/'.join([analysis_dir, 'data_ion_calib.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_heat_calib.h5'])
    hdf5_heat_calib(
        fine_data_path,
        output_data_path,
        selection_flag=True
    )

    ### NOISE
    fine_noise_path = '/'.join([analysis_dir, 'noise_ion_calib.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_heat_calib.h5'])   
    
    hdf5_heat_calib(
        fine_noise_path,
        output_noise_path,
        selection_flag=False
    )
    
    ### SIMU
    fine_simu_path = '/'.join([analysis_dir, 'simu_ion_calib.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_heat_calib.h5'])   
    
    hdf5_heat_calib(
        fine_simu_path,
        output_simu_path,
        selection_flag=False
    )  
        