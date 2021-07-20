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
# Crosstalk correction matrix
# =============================================================================
maxwell_detector = np.loadtxt('maxwell_red80.csv', comments='%', delimiter=',')
cabling_capacitance = 127e-12
maxwell_total = maxwell_detector + np.eye(4) * cabling_capacitance

xtalk_corr_matrix = maxwell_total / maxwell_total[0, 0]

# =============================================================================
# Function as usuam
# =============================================================================
ion_channel_labels = ('A', 'B', 'C', 'D')

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
    
    
    corr_ion_energy = np.dot(xtalk_corr_matrix, ion_energy.T)
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
    
    
    corr_ion_energy = np.dot(xtalk_corr_matrix, ion_energy.T)
    energy_corr_cname_list = [
        'energy_adu_corr_nodecor_ionA',
        'energy_adu_corr_nodecor_ionB',
        'energy_adu_corr_nodecor_ionC',
        'energy_adu_corr_nodecor_ionD'
    ]
    for i, col in enumerate(energy_corr_cname_list):
        df[col] = corr_ion_energy[i]
    
    return None


def pipeline_xtalk(stream, df_quality):
    """
    Common to all types of events: data, noise, simulation
    """
    df_calibrated = pd.DataFrame()
    for col in df_quality.columns:
        df_calibrated[col] = df_quality[col]
         
        
    crosstalk_correction(df_quality)
    
    #nodecor
    nodecor_crosstalk_correction(df_quality)

    return df_quality


def hdf5_xtalk_corr(fine_hdf5_path, output_hdf5_path):

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
    
    for stream in stream_list:

        df_quality = pd.read_hdf(
            fine_hdf5_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        df_xtalk = pipeline_xtalk(stream, df_quality)

        df_xtalk.to_hdf(
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
    fine_data_path = '/'.join([analysis_dir, 'data_quality.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_xtalk.h5'])   
    hdf5_xtalk_corr(
        fine_data_path,
        output_data_path,
    )

    ### NOISE
    fine_noise_path = '/'.join([analysis_dir, 'noise_quality.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_xtalk.h5'])   
    
    hdf5_xtalk_corr(
        fine_noise_path,
        output_noise_path,
    )

    ### SIMULATION
    fine_simu_path = '/'.join([analysis_dir, 'simu_quality.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_xtalk.h5'])   
    
    hdf5_xtalk_corr(
        fine_simu_path,
        output_simu_path,
    )
 
