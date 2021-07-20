#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:54:15 2020

@author: misiak
"""

import os
import uproot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.close('all')

plt.rcParams['text.usetex']=True
plt.rcParams['font.size']=9
plt.rcParams['lines.linewidth']=1

data_dir = "/home/misiak/Data/data_run57_neutron/Data/tg18l005/RED80"

fp_processed = '/'.join([
    data_dir,
    "ProcessedData_tg18l005_S02_RED80_ChanTrig0.root"
])

fp_trigger = '/'.join([
    data_dir,
    "RootFiles/TriggerData_tg18l005_S02_RED80_ChanTrig0.root"
])



root_processed = uproot.open(fp_processed)
tree_processed = root_processed['EventTree_trig_Normal_filt_decor']
tree_normal = root_processed['RunTree_Normal']

# =============================================================================
# test
# =============================================================================
# plt.figure()

# A = tree_normal.array('OF_Filt_Decor')[0, :, 0]

# plt.plot(A)
    

# =============================================================================
# searching for a good pulse
# =============================================================================
energy_array = tree_processed.array("Energy_OF")
chi2_array = tree_processed.array("chi2_OF")[:, 0]

# plt.figure()
# plt.plot(energy_array[ind], chi2_array[ind], ls='none', marker='.')

good_condition = ( abs(energy_array[:,0] - 1175) < 25 ) & ( chi2_array - 275 < 25 )
good_indexes = np.nonzero(good_condition)[0]
# print(np.nonzero(ind)[0])
i0 = 7

guard_indexes = np.nonzero(
    good_condition 
    & (abs(energy_array[:, 2])>40) 
    & (abs(energy_array[:, 4])>40)
)[0]

custom_indexes = np.nonzero(
    good_condition 
    & (abs(energy_array[:, 2])>15) 
    & (abs(energy_array[:, 3])>15)
    & (abs(energy_array[:, 4])>15) 
    & (abs(energy_array[:, 5])>15)
)[0]


# =============================================================================
# PLOT
# =============================================================================
root_trigger = uproot.open(fp_trigger)
tree_trigger = root_trigger['tree']

fs = 400
dt = fs**-1
time_array = np.arange(0, 0.5, dt)

# A = tree_trigger.array('Trace_Heat_A_Raw')
# B = tree_trigger.array('Trace_Heat_A_Raw_Decor')
# C = tree_trigger.array('Trace_OF')

leaf_labels = [
    'Trace_Heat_A_Raw_Decor',
    'Trace_Ion_A_Raw_Decor',
    'Trace_Ion_B_Raw_Decor',
    'Trace_Ion_C_Raw_Decor',
    'Trace_Ion_D_Raw_Decor',
    # 'Trace_OF',
]

# leaf_labels = [
#     'Trace_Heat_A_Raw',
#     'Trace_Ion_A_Raw',
#     'Trace_Ion_B_Raw',
#     'Trace_Ion_C_Raw',
#     'Trace_Ion_D_Raw',
#     'Trace_OF',
# ]

ndim = len(leaf_labels)

ind_list = [7, 1454, 1080, 2672]

fig, axes = plt.subplots(
    nrows=ndim,
    ncols=len(ind_list),
    figsize=(6.3,7),
    sharex='col',
    sharey='row'
)

color_list = ['slateblue', 'orangered', 'forestgreen', 'orange']

for i in range(ndim):
    
    
    for j,ind in enumerate(ind_list):
        ax = axes[i, j]
        
        signal_array = tree_trigger.array(leaf_labels[i])[ind]
        
        signal_array -= np.mean(signal_array[80:100])
        
        ax.plot(
                time_array,
                signal_array,
                color=color_list[j]
        )
        

### plot formatting

axes[0,0].set_ylabel(r'$V_{heat}$ / ADU')
axes[1,0].set_ylabel(r'$V_{A}$ / ADU')
axes[2,0].set_ylabel(r'$V_{B}$ / ADU')
axes[3,0].set_ylabel(r'$V_{C}$ / ADU')
axes[4,0].set_ylabel(r'$V_{D}$ / ADU')
# axes[5].set_ylabel(r'OF Trace / ADU')

axes[0, 0].yaxis.set_major_locator(mticker.MultipleLocator(500))
axes[0, 0].yaxis.set_minor_locator(mticker.MultipleLocator(100))
for i in range(ndim-1):
    axes[i+1, 0].yaxis.set_major_locator(mticker.MultipleLocator(20))
    axes[i+1, 0].yaxis.set_minor_locator(mticker.MultipleLocator(10))
    
for j in range(len(ind_list)):
    axes[-1, j].set_xlabel(r'Time / s')
    axes[-1, j].set_xlim(time_array[0], time_array[-1])

    # axes[-1, j].xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    axes[-1, j].xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    axes[-1, j].xaxis.set_minor_locator(mticker.MultipleLocator(0.05))

for i in range(ndim):
    for j in range(len(ind_list)-1):
        axes[i,j].spines['right'].set_linewidth(2)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

for ax in np.ravel(axes):
    ax.grid(True, alpha=0.5, which='major')
    ax.grid(True, alpha=0.1, which='minor')

axes[0,1].legend(
    title='Exemples of 10.37keV calibration events from the stream tg18l005 in detector RED80:',
    handles=[
        plt.Line2D([], [],color='slateblue'),
        plt.Line2D([], [],color='orangered'),
        plt.Line2D([], [],color='forestgreen'),
        plt.Line2D([], [],color='gold'),
    ],
    labels=[
        'Bulk event',
        'Guard event',
        'Guard-Collect\nevent',
        'Frontier event'
    ],
    loc='lower center',
    bbox_to_anchor=(1, 1,),
    ncol=4,
    frameon=False
)

### Figure adjustments
fig.align_ylabels(fig.axes)    
fig.tight_layout()
fig.subplots_adjust(hspace=.0, wspace=0.)

fig.savefig('/home/misiak/Analysis/NEUTRON/thesis_plots/pulse_raw.pdf')