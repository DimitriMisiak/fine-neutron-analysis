#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

quality_parameters = {
    "tg18l005": {
    	"glitch_time_cut": [ 
            [7.4, 7.6]
        ],
        "heat_chi2_threshold": 400,
    },
    "tg27l000": {
    	"glitch_time_cut": [ 
            [7, 11.3],
            [13.8, 14.1]
        ],
        "heat_chi2_threshold": 700,
    },
    "tg28l000": {
    	"glitch_time_cut": [ 
            [7.4, 8.05],
            [10, np.inf]
        ],
        "heat_chi2_threshold": 700,
    },
    "tg17l007": {
    	"glitch_time_cut": [],
        "heat_chi2_threshold": 400,
    },
    "tg19l010": {
    	"glitch_time_cut": [
            [8.3, 8.7]    
        ],
        "heat_chi2_threshold": 400,
        "position_10kev_line_heat": 1199,
    },
    "tg20l000": {
    	"glitch_time_cut": [],
        "heat_chi2_threshold": 400,
    },
    "tg21l000": {
    	"glitch_time_cut": [],
        "heat_chi2_threshold": 400,
    },
    "ion_chi2_threshold": 300,
    "offset_ion_threshold": 14000,
}

ion_channel_labels = ('A', 'B', 'C', 'D')


def glitch_time_cut(stream, df):
    """ 
    Create a new column with the "glitch time cut" truth array.
    """
    glitch_interval_list = quality_parameters[stream]['glitch_time_cut']
    
    if glitch_interval_list is None:
        df['glitch_time_cut'] = True
        
    else:
        
        timestamp = df['timestamp']
        
        truth_array = pd.Series(data=True, index=df.index)
        for inf, sup in glitch_interval_list:
            
            interval_truth = (timestamp < inf) | (timestamp > sup)
            truth_array = truth_array & interval_truth
            
        df['glitch_time_cut'] = truth_array
    
    return None
    

def maintenance_cut(df):
    """
    Create a new column with the "maintenance cut" truth array.
    """
    full_maintenance_duration = (
        (df.maintenance_duration + df.maintenance_cycle)/3600
    )
    remainder = df.timestamp % full_maintenance_duration
    
    df['maintenance_cut'] = (
        remainder > (df.maintenance_duration/3600)
    )
    
    return None


def reset_cut(df, tol=0.005):
    """
    Create a new column with the "reset_cut" truth array.
    """
    time_modulo_centered = df.time_modulo_reset -1.01 # seconds
    
    df['reset_cut'] = (
        abs(time_modulo_centered) > tol
    )
    
    return None

def heat_chi2_threshold_function(x2, e0):
    return x2 * ( 1 + (e0/2e3)**2 )

def heat_chi2_cut(stream, df):
    """ 
    Create a new column with the truth array of the 
    chi2 cut on the heat channel.
    """
    heat_chi2_threshold = quality_parameters[stream]['heat_chi2_threshold']
    chi2_heat = df['chi2_heat']
    energy_adu_heat = df['energy_adu_heat']
    
    # chi2_threshold = heat_chi2_threshold * ( 1 + (energy_adu_heat/2e3)**2 )
    chi2_threshold = heat_chi2_threshold_function(
        heat_chi2_threshold,
        energy_adu_heat
    )
        
    df['chi2_heat_cut'] = (chi2_heat < chi2_threshold)
    
    return None


def ion_chi2_threshold_function(x2, e0):
    return x2 * ( 1 + (e0/3e2)**2.2 )

def ion_chi2_cut(stream, df):
    """ 
    Create a new column with the truth array of the 
    chi2 cut on the ion channel.
    """
    ion_chi2_threshold = quality_parameters['ion_chi2_threshold']
    
    truth_array = pd.Series(data=True, index=df.index)
    for suffix in ion_channel_labels:
        chi2_column_name = 'chi2_ion{}'.format(suffix)
        chi2_ion = df[chi2_column_name]
        
        energy_column_name = 'energy_adu_ion{}'.format(suffix)
        energy_adu_ion = abs( df[energy_column_name] )
        
        # chi2_threshold = ion_chi2_threshold * ( 1 + (energy_adu_ion/3e2)**2.2 )
        chi2_threshold = ion_chi2_threshold_function(
            ion_chi2_threshold,
            energy_adu_ion
        )
        
        truth_array = truth_array & (chi2_ion < chi2_threshold)
        
    df['chi2_ion_cut'] = truth_array
    
    return None


def offset_ion_cut(df):
    """ 
    Create a new column with the truth array of the 
    chi2 cuts on the ion channels.
    """    

    offset_threshold = quality_parameters['offset_ion_threshold']
    
    truth_array = pd.Series(data=True, index=df.index)
    for suffix in ion_channel_labels:
        offset_column_name = 'offset_ion{}'.format(suffix)
        offset = abs( df[offset_column_name] )
        truth_array = truth_array & (offset < offset_threshold)
        
    df['offset_ion_cut'] = truth_array
    
    return None

    
def quality_cut(df):
    """ 
    Create a new column with the truth array of the 
    quality cut.
    """    

    quality_cut_components_columns = [
        'glitch_time_cut',
        'maintenance_cut',
        'reset_cut',
        'offset_ion_cut',
        'chi2_heat_cut',
        'chi2_ion_cut',
    ]
    
    truth_array = pd.Series(data=True, index=df.index)
    for col in quality_cut_components_columns:
        truth_array = truth_array & df[col]
        
    df['quality_cut'] = truth_array

    return None


def trigger_cut(
        df,
        nearest_data_trigger_allowed=5,
        furthest_simu_trigger_allowed=5
    ):
    """
    Create a new column for the trigger cut.
    This cut is specific to the simulated events.
    The tolerances should be given in milliseconds,
    and by defaults are both equals to 5ms.
    """
    df['trigger_cut'] = (
        ( abs(df['t_nearest_data_trigger']) > nearest_data_trigger_allowed )
        & ( abs(df['t_input_simu_trigger']) < furthest_simu_trigger_allowed )
    )


def pipeline_quality(stream, df_fine):
    """
    Common to all types of events: data, noise, simulation
    """
    # new dataframe initialization
    df_analysis = pd.DataFrame()
    for col in df_fine.columns:
        df_analysis[col] = df_fine[col]

    # valid running time of the detector
    glitch_time_cut(stream, df_analysis)
    
    # quality cuts
    maintenance_cut(df_analysis)
    reset_cut(df_analysis)
    heat_chi2_cut(stream, df_analysis)
    ion_chi2_cut(stream, df_analysis)
    offset_ion_cut(df_analysis)
    quality_cut(df_analysis)
    
    # check if simulation dataframe
    if ("simulation" in df_fine.columns):
        trigger_cut(df_analysis)
    
    return df_analysis


def hdf5_analysis(fine_hdf5_path, output_hdf5_path):

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

        df_fine = pd.read_hdf(
            fine_hdf5_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        df_analysis = pipeline_quality(stream, df_fine)

        df_analysis.to_hdf(
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
    fine_data_path = '/'.join([analysis_dir, 'data_reconstructed.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_quality.h5'])   
    hdf5_analysis(
        fine_data_path,
        output_data_path,
    )

    ### NOISE
    fine_noise_path = '/'.join([analysis_dir, 'noise_reconstructed.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_quality.h5'])   
    hdf5_analysis(
        fine_noise_path,
        output_noise_path,
    )
    
    ### SIMULATION
    fine_simu_path = '/'.join([analysis_dir, 'simu_reconstructed.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_quality.h5'])
    hdf5_analysis(
        fine_simu_path,
        output_simu_path,
    )
    