#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:09:10 2020

@author: misiak
"""

import os
import uproot
import pandas as pd
from tqdm import tqdm


stream_configuration = {
    "Background": [
    	"tg18l005",
    	"tg27l000",
    	"tg28l000"
    ],
    "Calibration": [
    	"tg17l007",
    	"tg19l010",
    	"tg20l000",
    	"tg21l000"
    ]
}

simulation_configuration = {
        'flat_ER' : 'Flat_Analytical_SimuOnly_0.0000_50.0000_ER',
        'flat_NR' : 'Flat_Analytical_SimuOnly_0.0000_50.0000_NR',
        'line_1keV' : 'Line_Analytical_SimuOnly_1.3000_ER',
        'line_10keV' : 'Line_Analytical_SimuOnly_10.3700_ER',
}

detector = "RED80"


def data_stream_info(directory):
    """
    Return the characteristics of each ROOT file partitions
    for the experimental data.
    """
    # detector, mode, stream, file path
    stream_info_list = list()
    
    for mode, stream_list in stream_configuration.items():
        
        for stream in stream_list:
            
            dir_path = '/'.join([directory, stream, detector])
            
            for fname in os.listdir(dir_path):
                
                if 'ProcessedData' in fname:
                    
                    file_path = '/'.join([dir_path, fname])
                    stream_info = [
                        detector,
                        mode,
                        stream,
                        file_path
                    ]
                    stream_info_list.append(stream_info)
                    
    return stream_info_list


def simu_stream_info(directory):
    """
    Return the characteristics of each ROOT file partitions
    for the pulse simulation.
    """
    # detector, mode, stream, simu, file path
    stream_info_list = list()
    # looping over the source configuration: Background or Calibration
    for mode, stream_list in stream_configuration.items():
        
        # looping over the streams
        for stream in stream_list:
            
            stream_dir = '/'.join([directory, stream, detector])
            
            # looping over the content of the directory
            for dir_name in os.listdir(stream_dir):
                
                # looping over the simulation type and their relative prefix
                for simu, prefix in simulation_configuration.items():
                    
                    if prefix not in dir_name:
                        continue
                    
                    simu_dir = '/'.join([stream_dir, dir_name])
                    for fname in os.listdir(simu_dir):
                        
                        if 'ProcessedData' in fname:
                            
                            file_path = '/'.join([simu_dir, fname])
                            stream_info = [
                                detector,
                                mode,
                                stream,
                                simu,
                                file_path
                            ]
                            stream_info_list.append(stream_info)
    
    return stream_info_list


def data_root_to_df(root):
    """
    For experimental data.
    Creates DataFrame from the data in the ROOT file.
    """
    # joining trees for triggering events
    df_raw = root['EventTree_trig_Normal_raw'].pandas.df()
    df_filt = root['EventTree_trig_Normal_filt'].pandas.df()
    df_filt_decor = root['EventTree_trig_Normal_filt_decor'].pandas.df()
    
    df_list = [df_raw, df_filt, df_filt_decor]
    suffix_list = ['_raw', '_filt', '_filt_decor']
    for df, suffix in zip(df_list, suffix_list):
        # decode_byte_column_name(df)
        df.rename(
            columns = {name:name+suffix for name in df.columns},
            inplace=True
        )
    
    df = df_raw.join([df_filt, df_filt_decor])
    
    # new columns for general info in the RunTree_Normal tree
    run_tree = root['RunTree_Normal']
    df['frequency'] = run_tree['f_max_heat'].array()[0]
    df['time_window'] = run_tree['TimeWindow_Heat'].array()[0]
    df['maintenance_cycle'] = run_tree['MaintenanceCycle'].array()[0]
    df['maintenance_duration'] = run_tree['MaintenanceDuration'].array()[0]
    df['reset_period'] = run_tree['Rt'].array()[0]
    
    polarization = run_tree['Polar_Ion'].array()[0]
    for polar, label in zip(polarization, ['A', 'B', 'C', 'D']):
        df['polar_'+label] = polar
    
    return df


def noise_root_to_df(root):


    df_raw_noise = root['EventTree_noise_Normal_raw'].pandas.df()
    df_filt_noise = root['EventTree_noise_Normal_filt'].pandas.df()    
    df_filt_decor_noise = root['EventTree_noise_Normal_filt_decor'].pandas.df()    

    df_list = [df_raw_noise, df_filt_noise, df_filt_decor_noise]
    suffix_list = ['_raw', '_filt', '_filt_decor']
    for df, suffix in zip(df_list, suffix_list):
        # decode_byte_column_name(df)
        df.rename(
            columns = {name:name+suffix for name in df.columns},
            inplace=True
        )
    
    df = df_raw_noise.join([df_filt_noise, df_filt_decor_noise])
    
    # new columns for general info in the RunTree_Normal tree
    run_tree = root['RunTree_Normal']
    df['frequency'] = run_tree['f_max_heat'].array()[0]
    df['time_window'] = run_tree['TimeWindow_Heat'].array()[0]
    df['maintenance_cycle'] = run_tree['MaintenanceCycle'].array()[0]
    df['maintenance_duration'] = run_tree['MaintenanceDuration'].array()[0]
    df['reset_period'] = run_tree['Rt'].array()[0]
    
    polarization = run_tree['Polar_Ion'].array()[0]
    for polar, label in zip(polarization, ['A', 'B', 'C', 'D']):
        df['polar_'+label] = polar
    
    return df


def simu_root_to_df(root):
    """
    For pulse simulation.
    Creates DataFrame from the data in the ROOT file.
    """    
    df = data_root_to_df(root)
    
    # specific to simulation data
    df_energy_input = root['tree_Input_SimuOnly_Energy'].pandas.df()
    df_tsimucoinc = root['tsimucoinc'].pandas.df()
    
    df = df.join([df_energy_input, df_tsimucoinc])
    
    return df


def data_root_to_hdf5(root_directory, save_path):

    stream_info_list = data_stream_info(root_directory)
    
    with pd.HDFStore(save_path, mode='w') as store:
        
        chunk_start = 0
        for detector, mode, stream, file_path in tqdm(stream_info_list):
            
            # opening root file
            root = uproot.open(file_path)
            
            # creating a DataFrame from the info in the RootDirectory
            df = data_root_to_df(root)
            
            ### HACK because of empty dataframes i think
            if not df.empty:
                try:
                    # converting bytes to str
                    str_df = df.select_dtypes('O')
                    str_df = str_df.stack().str.decode('utf-8').unstack()
                    for col in str_df:
                        df[col] = str_df[col]
                except:
                    print('Could not decode bytes. Be careful with this file:'
                          '\n{}'.format(file_path))
                
            # creating columns for more general info
            df['detector'] = detector
            df['source'] = mode
            df['stream'] = stream
            
            # correct indexing
            chunk_end = chunk_start + df.shape[0]
            df.index = range(chunk_start, chunk_end)
            chunk_start = chunk_end
            
            # df_list.append(df)
            store.append('df', df, min_itemsize=11)


def noise_root_to_hdf5(root_directory, save_path):

    stream_info_list = data_stream_info(root_directory)
    
    with pd.HDFStore(save_path, mode='w') as store:
        
        chunk_start = 0
        for detector, mode, stream, file_path in tqdm(stream_info_list):
            
            # opening root file
            root = uproot.open(file_path)
            
            # creating a DataFrame from the info in the RootDirectory
            df = noise_root_to_df(root)
            
            ### HACK because of empty dataframes i think
            if not df.empty:
                try:
                    # converting bytes to str
                    str_df = df.select_dtypes('O')
                    str_df = str_df.stack().str.decode('utf-8').unstack()
                    for col in str_df:
                        df[col] = str_df[col]
                except:
                    print('Could not decode bytes. Be careful with this file:'
                          '\n{}'.format(file_path))
                
            # creating columns for more general info
            df['detector'] = detector
            df['source'] = mode
            df['stream'] = stream
            
            # correct indexing
            chunk_end = chunk_start + df.shape[0]
            df.index = range(chunk_start, chunk_end)
            chunk_start = chunk_end
            
            # df_list.append(df)
            store.append('df', df, min_itemsize=11)



def simu_root_to_hdf5(root_path, save_path):
    """
    Build the HDF5 file for the pulse simulation.
    """
    stream_info_list = simu_stream_info(root_path)
    
    with pd.HDFStore(save_path, mode='w') as store:
        
        chunk_start = 0
        for detector, mode, stream, simu, file_path in tqdm(stream_info_list):
            
            # opening root file
            root = uproot.open(file_path)
            
            # creating a DataFrame from the info in the RootDirectory
            df = simu_root_to_df(root)
            
            ### HACK because of empty dataframes i think
            if not df.empty:
                try:
                    # converting bytes to str
                    str_df = df.select_dtypes('O')
                    str_df = str_df.stack().str.decode('utf-8').unstack()
                    for col in str_df:
                        df[col] = str_df[col]
                except:
                    print('Could not decode bytes. Be careful with this file:'
                          '\n{}'.format(file_path))
                
            # creating columns for more general info
            df['detector'] = detector
            df['source'] = mode
            df['stream'] = stream   
            df['simulation'] = simu

            # correct indexing
            chunk_end = chunk_start + df.shape[0]
            df.index = range(chunk_start, chunk_end)
            chunk_start = chunk_end
            
            # df_list.append(df)
            store.append('df', df, min_itemsize=11)


if __name__ == "__main__":
    
    output_directory = '/home/misiak/Analysis/NEUTRON'
    
    # DATA root files
    data_root_directory = '/home/misiak/Data/data_run57_neutron/Data'
    data_output_path = '/'.join([output_directory, 'data_raw.h5'])
    # data_root_to_hdf5(data_root_directory, data_output_path)

    # NOISE root files
    noise_output_path = '/'.join([output_directory, 'noise_raw.h5'])
    noise_root_to_hdf5(data_root_directory, noise_output_path)
    
    # SIMULATION root files
    simulation_root_directory = '/home/misiak/Data/data_run57_neutron/SimuCoinc'
    simulation_output_path = '/'.join([output_directory, 'simu_raw.h5'])
    # simu_root_to_hdf5(simulation_root_directory, simulation_output_path)
    