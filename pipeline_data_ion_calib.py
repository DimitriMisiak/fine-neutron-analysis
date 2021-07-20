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

calibration_parameters = {    
    "position_10kev_line_ionA": 57.80,
    "position_10kev_line_ionB": 58.93,
    "position_10kev_line_ionC": 57.01,
    "position_10kev_line_ionD": 58.80,
}

### exception goes below


# =============================================================================
# Function as usuam
# =============================================================================
ion_channel_labels = ('A', 'B', 'C', 'D')



def calibration_ion(stream, df):
    """ 
    Create new columns for the calibrated energy of the ionization channels.
    """         
    for suffix in ion_channel_labels:
        
        position_cname = 'position_10kev_line_ion{}'.format(suffix)
        energy_adu_cname ='energy_adu_corr_ion{}'.format(suffix)
        energy_cname = 'energy_ion{}'.format(suffix)
        
        polar = df['polar_{}'.format(suffix)].unique()[0]
        
        calibration_factor = (
            -np.sign(polar) * 10.37 / calibration_parameters[position_cname]
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
        
        polar = df['polar_{}'.format(suffix)].unique()[0]
        
        calibration_factor = (
            -np.sign(polar) * 10.37 / calibration_parameters[position_cname]
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
    
    df['energy_ion_conservation'] = (
        - np.sign(df['polar_A'].unique()[0]) * df['energy_ionA']
        - np.sign(df['polar_B'].unique()[0]) * df['energy_ionB']
        - np.sign(df['polar_C'].unique()[0]) * df['energy_ionC']
        - np.sign(df['polar_D'].unique()[0]) * df['energy_ionD']
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
    
    df['energy_nodecor_ion_conservation'] = (
        - np.sign(df['polar_A'].unique()[0]) * df['energy_nodecor_ionA']
        - np.sign(df['polar_B'].unique()[0]) * df['energy_nodecor_ionB']
        - np.sign(df['polar_C'].unique()[0]) * df['energy_nodecor_ionC']
        - np.sign(df['polar_D'].unique()[0]) * df['energy_nodecor_ionD']
    ) / 2

    return None



def pipeline_ion_calib(stream, df_quality):
    """
    Common to all types of events: data, noise, simulation
    """
    df_calibrated = pd.DataFrame()
    for col in df_quality.columns:
        df_calibrated[col] = df_quality[col]
        
    calibration_ion(stream, df_quality)
    nodecor_calibration_ion(stream, df_quality)

    virtual_channels(df_quality)
    nodecor_virtual_channels(df_quality)
    
    return df_quality


def hdf5_ion_calib(fine_hdf5_path, output_hdf5_path):

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
        
        df_quality['polar_A'] = 1
        df_quality['polar_B'] = 1
        df_quality['polar_C'] = -1
        df_quality['polar_D'] = -1
        
        df_ion_calib = pipeline_ion_calib(stream, df_quality)

        df_ion_calib.to_hdf(
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
    fine_data_path = '/'.join([analysis_dir, 'data_xtalk.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_ion_calib.h5'])   
    hdf5_ion_calib(
        fine_data_path,
        output_data_path,
    )

    ### NOISE
    fine_noise_path = '/'.join([analysis_dir, 'noise_xtalk.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_ion_calib.h5'])   
    
    hdf5_ion_calib(
        fine_noise_path,
        output_noise_path,
    )

    ### simu
    fine_simu_path = '/'.join([analysis_dir, 'simu_xtalk.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_ion_calib.h5'])   
    
    hdf5_ion_calib(
        fine_simu_path,
        output_simu_path,
    )  
    
