#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:12:45 2020

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from pipeline_data_heat_calib import (
    energy_heat_from_er_and_quenching,
    energy_ion_from_er_and_quenching,
    lindhard,
)


def sigma_extraction(df_analysis, df_data, df_noise):

    attr_list_common = [
        'energy_nodecor_ion_conservation',
    ]
    
    attr_list_specific = [
        'energy_heat',
        'energy_ionA',
        'energy_ionB',
        'energy_ionC',
        'energy_ionD',
        'energy_ion_total',
        'energy_ion_bulk',
        'energy_ion_guard',
    ]
    
    dfn_quality = df_noise[df_noise.quality_cut]    
    df_selection = df_data[df_data.selection_cut]
    
    for attr in attr_list_common:
        df_analysis['std_'+attr] = np.std(dfn_quality[attr])
        df_analysis['std_calib_'+attr] = np.std(df_selection[attr])

    for attr in attr_list_specific:
        df_analysis['std_'+attr] = np.std(dfn_quality[attr])

        blob_cut = (df_selection[attr] >= 10.37)
        df_blob = df_selection[blob_cut]
        sup = np.quantile(df_blob[attr], [0.68,])[0]
        
        df_analysis['std_calib_'+attr] = (sup - 10.37)
   
    return None


def sigma_function(energy_heat, sigma_baseline, sigma_calib):
    
    sigma_local = (
        sigma_baseline
        + (sigma_calib - sigma_baseline) * energy_heat / 10.37
    )
    
    return sigma_local
    
    
def charge_conservation_cut(df_analysis):
    """ 
    Create a new column with the truth array of the 
    charge conservation cut.
    """
    nsigma = 2
    
    sig0 = df_analysis['std_energy_nodecor_ion_conservation'].unique()[0]
    sig10 = df_analysis['std_calib_energy_nodecor_ion_conservation'].unique()[0]
    
    sigma = sigma_function(
        df_analysis.energy_heat,
        sig0,
        sig10
        )
    
    threshold = nsigma * sigma
    
    ion_conservation = df_analysis['energy_nodecor_ion_conservation']
    
    df_analysis['charge_conservation_cut'] = (
        abs(ion_conservation) < threshold
    )
        
    return None


def fid_cuts(df):
    """ 
    Apply the so-called FID cuts, which is a way to discriminate events
    happening in the bulk (or in the guard) region from the others.
    Create new columns with the truth array for the bulk and guard events.
    """
    nsigma = 2
    
    df['bulk_cut'] = (
        ( abs(df['energy_ionA']) < df['std_energy_ionA'] * nsigma )
        & ( abs(df['energy_ionC']) < df['std_energy_ionC'] * nsigma )
    )

    df['guard_cut'] = (
        ( abs(df['energy_ionB']) < df['std_energy_ionB'] * nsigma )
        & ( abs(df['energy_ionD']) < df['std_energy_ionD'] * nsigma )
    )
    
    return None


def band_cut(df):

    energy_heat = df['energy_heat']
    energy_ion = df['energy_ion_bulk']
    
    nsigma = 2
    
    sig0 = df['std_energy_heat'].unique()[0]
    sig10 = df['std_calib_energy_heat'].unique()[0]
    
    ei_err = nsigma * sigma_function(energy_heat, sig0, sig10)
    ei_err_baseline = nsigma * sig0
    
    gamma_cut = ( abs(energy_ion - energy_heat) < ei_err )

    er_array = np.linspace(0, energy_heat.max(), int(1e4))
    dv=2
    # ec_array = er_array * (1 + lindhard(er_array)*dv/3) / (1 + dv/3)
    ec_array = energy_heat_from_er_and_quenching(
        er_array,
        lindhard(er_array),
        dv
    )
    # ei_array = er_array * lindhard(er_array)
    ei_array = energy_ion_from_er_and_quenching(
        er_array,
        lindhard(er_array)
    )
    
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


def pipeline_science_common(df_calibrated, df_data, df_noise):
    df_science = pd.DataFrame()
    for col in df_calibrated.columns:
        df_science[col] = df_calibrated[col]   
    
    sigma_extraction(df_science, df_data, df_noise)
    charge_conservation_cut(df_science)
    fid_cuts(df_science)
    band_cut(df_science)
    energy_cut(df_science)    
    
    return df_science



def hdf5_science(fine_hdf5_path, fine_data_path, fine_noise_path, output_hdf5_path):

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

        df_noise = pd.read_hdf(
            fine_noise_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )

        df_data = pd.read_hdf(
            fine_data_path,
            key='df',
            where='stream = "{}"'.format(stream)
        )
        
        df_science = pipeline_science_common(df_calibrated, df_data, df_noise)

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

    
    analysis_dir = '/home/misiak/Analysis/NEUTRON'

    fine_noise_path = '/'.join([analysis_dir, 'noise_heat_calib.h5'])
  
    ### DATA
    fine_data_path = '/'.join([analysis_dir, 'data_heat_calib.h5'])
    output_data_path = '/'.join([analysis_dir, 'data_science.h5'])   
    
    hdf5_science(
        fine_data_path,
        fine_data_path,
        fine_noise_path,
        output_data_path,
    )
    
    # ### SIMULATION
    # fine_simu_path = '/'.join([analysis_dir, 'simu_heat_calib.h5'])
    # output_simu_path = '/'.join([analysis_dir, 'simu_science.h5'])
    
    # hdf5_science(
    #     fine_simu_path,
    #     fine_data_path,
    #     fine_noise_path,
    #     output_simu_path,
    # )
    