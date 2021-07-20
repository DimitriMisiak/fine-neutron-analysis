#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""


import pandas as pd


def extract_useful_columns_for_data(df):
    """
    Return a Dataframe with much less, but useful informations.
    Also, the name of the columns is now clearer.
    Dimitri likes it a lot to work with this :)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing unprocessed raw data freshly extracted from
        the ROOT files.

    Returns
    -------
    Panda Dataframe
    """
    # initialization of the fine DataFrame
    df_fine = pd.DataFrame()
    col_kept = [
        'detector',
        'source',
        'stream',
        'frequency',
        'maintenance_duration',
        'maintenance_cycle'
    ]
    for col in col_kept:
        df_fine[col] = df[col]

    # creating the timestamp column
    df_fine['timestamp'] = (
        df.MicroStp_filt_decor / (df.frequency * 3600)
        + df.NumPart_filt_decor
    )

    # creating the time_modulo_reset column
    df_fine['time_modulo_reset'] = df['Time_modulo_reset_filt_decor']

    # extracting the useful columns, and renaming them
    chi2_key_dict = {
        'chi2_heat': 'chi2_OF[0]_filt_decor',
        'chi2_ionA': 'chi2_OF[2]_filt_decor',
        'chi2_ionB': 'chi2_OF[3]_filt_decor',
        'chi2_ionC': 'chi2_OF[4]_filt_decor',
        'chi2_ionD': 'chi2_OF[5]_filt_decor',
    }

    offset_key_dict = {
        'offset_heat': 'Off[0]_raw',
        'offset_ionA': 'Off[2]_raw',
        'offset_ionB': 'Off[3]_raw',
        'offset_ionC': 'Off[4]_raw',
        'offset_ionD': 'Off[5]_raw',
    }

    slope_key_dict = {
         'slope_ionA': 'Slope_Ion[0]_raw',
         'slope_ionB': 'Slope_Ion[1]_raw',
         'slope_ionC': 'Slope_Ion[2]_raw',
         'slope_ionD': 'Slope_Ion[3]_raw',
    }

    energy_key_dict = {
        'energy_adu_heat': 'Energy_OF[0]_filt_decor',
        'energy_adu_ionA': 'Energy_OF[2]_filt_decor',
        'energy_adu_ionB': 'Energy_OF[3]_filt_decor',
        'energy_adu_ionC': 'Energy_OF[4]_filt_decor',
        'energy_adu_ionD': 'Energy_OF[5]_filt_decor',
        'energy_adu_nodecor_heat': 'Energy_OF[0]_filt',
        'energy_adu_nodecor_ionA': 'Energy_OF[2]_filt',
        'energy_adu_nodecor_ionB': 'Energy_OF[3]_filt',
        'energy_adu_nodecor_ionC': 'Energy_OF[4]_filt',
        'energy_adu_nodecor_ionD': 'Energy_OF[5]_filt',
    }

    # merging the previous directories together
    useful_key_dict = {
        **chi2_key_dict,
        **offset_key_dict,
        **slope_key_dict,
        **energy_key_dict,
    }

    for col, key in useful_key_dict.items():
        df_fine[col] = df[key]

    return df_fine


def extract_useful_columns_for_noise(df):
    """
    Same as the previous function, but with additional extracted columns
    for the pulse simulation.
    """
    
    df_fine = extract_useful_columns_for_data(df)
    
    # extracting the useful columns, and renaming them
    chi2_key_dict = {
        'chi2_heat': 'chi2_OF_t0[0]_filt_decor',
        'chi2_ionA': 'chi2_OF_t0[2]_filt_decor',
        'chi2_ionB': 'chi2_OF_t0[3]_filt_decor',
        'chi2_ionC': 'chi2_OF_t0[4]_filt_decor',
        'chi2_ionD': 'chi2_OF_t0[5]_filt_decor',
    }

    energy_key_dict = {
        'energy_adu_heat': 'Energy_OF_t0[0]_filt_decor',
        'energy_adu_ionA': 'Energy_OF_t0[2]_filt_decor',
        'energy_adu_ionB': 'Energy_OF_t0[3]_filt_decor',
        'energy_adu_ionC': 'Energy_OF_t0[4]_filt_decor',
        'energy_adu_ionD': 'Energy_OF_t0[5]_filt_decor',
        'energy_adu_nodecor_heat': 'Energy_OF_t0[0]_filt',
        'energy_adu_nodecor_ionA': 'Energy_OF_t0[2]_filt',
        'energy_adu_nodecor_ionB': 'Energy_OF_t0[3]_filt',
        'energy_adu_nodecor_ionC': 'Energy_OF_t0[4]_filt',
        'energy_adu_nodecor_ionD': 'Energy_OF_t0[5]_filt',
    }

    # merging the previous directories together
    noise_key_dict = {
        **chi2_key_dict,
        **energy_key_dict,
    }

    for col, key in noise_key_dict.items():
        df_fine[col] = df[key]

    return df_fine


def extract_useful_columns_for_simulation(df):
    """
    Same as the previous function, but with additional extracted columns
    for the pulse simulation.
    """
    
    df_fine = extract_useful_columns_for_data(df)
    
    df_fine['simulation'] = df['simulation']
    
    simu_key_dict = {
        't_input_simu_trigger': 'deltaT_simu_datasimu',
        't_nearest_data_trigger': 'deltaT_simu_data',
        'input_energy': 'Energy_In'
    }

    for col, key in simu_key_dict.items():
        df_fine[col] = df[key]

    return df_fine


def hdf5_processing(raw_hdf5_path, output_hdf5_path, extract_function):

    # reading the *large* raw DataFrame chunk by chunk
    df_iterator = pd.read_hdf(
        raw_hdf5_path,
        key='df', #change to df later
        chunksize=50000
    )

    # initializing the HDFstore (overwriting, be careful !)
    pd.DataFrame().to_hdf(output_hdf5_path, key='df', mode='w', format='table')

    # tqdm to follow the progression
    from tqdm import tqdm
    niter = df_iterator.nrows // df_iterator.chunksize + 1
    progress_bar = tqdm(total=niter)

    for df in df_iterator:
        
        df_fine = extract_function(df)
        
        # min_itemsize reserves enough size for str objects in the df
        # data_columns=True enables the query on-disk for all columns
        df_fine.to_hdf(output_hdf5_path, key='df', mode='a', format='table',
                       append=True, min_itemsize=11,  data_columns=True)
        
        progress_bar.update()
        
    progress_bar.close()    



if __name__ == "__main__":

    analysis_dir = '/home/misiak/Analysis/NEUTRON'
        
    raw_data_path =  '/'.join([analysis_dir, 'data_raw.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_reconstructed.h5'])

    # processing the experimental data
    hdf5_processing(
        raw_data_path,
        output_data_path,
        extract_useful_columns_for_data
    )

    raw_noise_path =  '/'.join([analysis_dir, 'noise_raw.h5'])
    output_noise_path = '/'.join([analysis_dir, 'noise_reconstructed.h5'])

    # processing the experimental data
    hdf5_processing(
        raw_noise_path,
        output_noise_path,
        extract_useful_columns_for_noise
    )

    raw_simu_path =  '/'.join([analysis_dir, 'simu_raw.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_reconstructed.h5'])
  
    # # processinf the pulse simulation
    hdf5_processing(
        raw_simu_path,
        output_simu_path,
        extract_useful_columns_for_simulation
    )
    