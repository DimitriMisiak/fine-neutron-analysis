#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:55:53 2020

@author: misiak
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_data_calibrated import (
    lindhard,
)


def charge_conservation_threshold(energy_heat):
    """
    Depends on the heat energy to take into consideration the possibility of
    charge trapping/detrapping.
    """
    nsigma = 2
    return (0.4120323 + 0.00165406 * energy_heat) * nsigma

def charge_conservation_cut(df):
    """ 
    Create a new column with the truth array of the 
    charge conservation cut.
    """    
    ion_conservation = df['energy_nodecor_ion_conservation']
    energy_heat = df['energy_heat']
    
    threshold = charge_conservation_threshold(energy_heat)
        
    df['charge_conservation_cut'] = abs(ion_conservation) < threshold
        
    return None


# def guard_threshold_for_bulk_cut(energy_ion):
#     # std_noise_blob_fid = 1
#     # std_10kev_fid = 1.5
#     std_noise_blob_fid = 0.29
#     std_10kev_fid = 0.5
#     alpha_fid = (std_10kev_fid**2 - std_noise_blob_fid**2)**0.5 / 10.37    
#     return (std_noise_blob_fid**2 + (alpha_fid*energy_ion)**2)**0.5


# def bulk_threshold_for_guard_cut(energy_ion):
#     # std_noise_blob_fid = 1
#     # std_10kev_fid = 1.5
#     std_noise_blob_fid = 0.29
#     std_10kev_fid = 0.5
#     alpha_fid = (std_10kev_fid**2 - std_noise_blob_fid**2)**0.5 / 10.37    
#     return (std_noise_blob_fid**2 + (alpha_fid*energy_ion)**2)**0.5

def ionization_baseline_resolution():
    std_noise_blob = 0.29
    return std_noise_blob


def fid_cuts(df):
    """ 
    Apply the so-called FID cuts, which is a way to discriminate events
    happening in the bulk (or in the guard) region from the others.
    Create new columns with the truth array for the bulk and guard events.
    """
    nsigma = 2
    
    collect_energy_threshold = ionization_baseline_resolution() * nsigma

    guard_energy_threshold = ionization_baseline_resolution() * nsigma
    
    df['bulk_cut'] = (
        ( abs(df['energy_ionA']) < collect_energy_threshold )
        & ( abs(df['energy_ionC']) < collect_energy_threshold )
    )

    df['guard_cut'] = (
        ( abs(df['energy_ionB']) < guard_energy_threshold )
        & ( abs(df['energy_ionD']) < guard_energy_threshold )
    )
    
    return None


def std_energy_ion(ec):
    """
    standard deviation on the ionization energy in function of the heat energy.
    """
    # std_nb = 0.254
    # std_nb = 0.200
    std_nb = 0.22
    #std_1kev = 0.272
    # std_10kev = 0.318
    std_10kev = 0.45
    alpha = (std_10kev**2 - std_nb**2)**0.5 / 10.37
    return ( std_nb**2 + (alpha*ec)**2 )**0.5


def band_cut(df):

    energy_heat = df['energy_heat']
    energy_ion = df['energy_ion_bulk']
    nsigma = 2
    ei_err = nsigma * std_energy_ion(energy_heat)
    ei_err_baseline = nsigma * std_energy_ion(np.zeros(energy_heat.shape))
    
    gamma_cut = ( abs(energy_ion - energy_heat) < ei_err )

    er_array = np.linspace(0, energy_heat.max(), int(1e4))
    dv=2
    ec_array = er_array * (1 + lindhard(er_array)*dv/3) / (1 + dv/3)
    ei_array = er_array * lindhard(er_array)
    energy_ion_lindhard = np.interp(energy_heat, ec_array, ei_array)
    
    neutron_cut = ( abs(energy_ion - energy_ion_lindhard) < ei_err )

    HO_cut = ( abs(energy_ion) < ei_err_baseline )

    df['gamma_cut'] = gamma_cut
    df['neutron_cut'] = neutron_cut
    df['HO_cut'] = HO_cut

    return None


def energy_cut(df, energy_bounds=[0.025, 50]):
    """
    Extra cut to select a specific range in energy.
    Quite handy to drop event with negative energy (not physical) and event 
    with a high energy (non-linearity becomes problematic).
    """
    inf, sup = energy_bounds
    
    df['energy_cut'] = (
        ( df['energy_heat'] > inf )
        & ( df['energy_heat'] < sup )
    )
    
    # add a condition on the ionization energy
    
    return None


def pipeline_science_common(df_calibrated):
    df_science = pd.DataFrame()
    for col in df_calibrated.columns:
        df_science[col] = df_calibrated[col]   
    
    charge_conservation_cut(df_science)
    fid_cuts(df_science)
    band_cut(df_science)
    energy_cut(df_science)    
    
    return df_science


def hdf5_science(fine_hdf5_path, output_hdf5_path):

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

        df_calibrated = pd.read_hdf(
            fine_hdf5_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        df_science = pipeline_science_common(df_calibrated)

        df_science.to_hdf(
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
    fine_data_path = '/'.join([analysis_dir, 'data_calibrated.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_science.h5'])   
    
    hdf5_science(
        fine_data_path,
        output_data_path,
    )
    
    ### SIMULATION
    fine_simu_path = '/'.join([analysis_dir, 'simu_calibrated.h5'])
    output_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])
    
    hdf5_science(
        fine_simu_path,
        output_simu_path,
    )
    