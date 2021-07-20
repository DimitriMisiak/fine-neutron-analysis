#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:09:11 2020

@author: misiak
"""


import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt


# P = st.poisson(100)

# samples = P.rvs(int(1e6))

# # plt.plot(
# #     samples,
# #     ls='none',
# #     marker='o'
# # )

# plt.hist(
#     1/samples,
# )

var_list = list()

x_data = np.linspace(0, 100, 100)

from tqdm import tqdm
for i in tqdm(x_data):
    P = st.poisson(i)
    samples = P.rvs(int(1e6))
    var_list.append(
        (1/samples).var()
    )
    
plt.plot(
    x_data,
    var_list,
    label='exp'
)

plt.plot(
    x_data,
    x_data**-1,
    label='1/lambda'
)
    
plt.legend()
plt.grid()