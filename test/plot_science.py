#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:35:22 2020

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



stream = 'tg09l000'

h5_path = '/home/misiak/Analysis/RED80/{}/data_heat_calib.h5'.format(stream)

df_analysis = pd.read_hdf(
    h5_path,
    key='df',
    where=(
        'stream = "{0}"'
    ).format(stream)
)

df_quality = df_analysis[df_analysis.quality_cut]
df_selection = df_analysis[df_analysis.selection_cut]

# =============================================================================
# SANITY PLOT
# =============================================================================
fig, ax = plt.subplots()

ax.plot(
        df_quality.energy_adu_heat,
        df_quality.energy_ion_total,
        ls='none',
        marker='.',
        color='b',
        alpha=0.3
)

ax.plot(
        df_selection.energy_adu_heat,
        df_selection.energy_ion_total,
        ls='none',
        marker='.',
        color='r'
)

