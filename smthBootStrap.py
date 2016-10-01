from __future__ import division
"""
Created on Fri Sep 16 11:27:29 2016

@author: markiemt
"""

#-----------------------------------------------------------------------------#
'''
Smoothed Bootstrap Part 2
'''

#Normal necessary inputs
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from astroML.resample import bootstrap
from astroML.utils import check_random_state

#Smoothed Bootstrap Function
def smthBoot(data, n_bootstraps, user_stat, kwargs=None, random_state=None):
    if kwargs is None:
        kwargs = {}
        
    data = np.asarray(data)
    n_samples = data.size
    rng = check_random_state(random_state)
        
    sample_std = np.std(data,ddof=1)
    noise_std = sample_std/(np.sqrt(n_samples))
    
    ind = rng.randint(n_samples, size=(n_bootstraps, n_samples))
    noise_data = npr.normal(0.0, noise_std, size=(n_bootstraps, n_samples))
    
    
    stat_smthBoot = user_stat(data[ind] + noise_data, **kwargs)
        
    return stat_smthBoot

# input constants
sgm = 1.0
n_boot = 1000
n_runs = 1000
n_samples = 5
user_stat = np.std
smthStat = np.zeros(n_runs)
bootStat = np.zeros(n_runs)
computed_std = np.zeros(n_runs)

# create data for bootstrapping
for i in range(n_runs):
    data = npr.normal(0.0, sgm, 5)
    # Histograms of each statistic output for bootStat and smthStat
    #plt.clf
    #plt.figure(i)
    #plt.hist(smthBoot(data, n_boot, user_stat, kwargs=dict(axis=1, ddof=1)), bins=50, color='b', histtype='step')
    #plt.hist(bootstrap(data, n_boot, user_stat, kwargs=dict(axis=1, ddof=1)), bins=50, color='r', histtype='step')
    smthStat[i] = np.median(smthBoot(data, n_boot, user_stat, kwargs=dict(axis=1, ddof=1)))
    #Regular bootstrap for comparison
    bootStat[i] = np.median(bootstrap(data, n_boot, user_stat, kwargs=dict(axis=1, ddof=1)))
    #Computed standard deviation per data set
    computed_std[i] = np.std(data, ddof=1)
    

# Plotting histograms of all median std calculated by each method
plt.figure(0)
plt.clf
plt.hist(smthStat, bins = 50, color='b', histtype='step')
plt.hist(bootStat, bins = 50, color='r', histtype='step')
plt.hist(computed_std, bins = 50, color='g', histtype='step')
plt.legend(['Smooth Bootstrap','Bootstrap','Standard Deviation'])

# Plotting the ratios of smthStat to Std. and bootStat to Std.
plt.figure(1)
plt.clf
plt.hist(smthStat/computed_std, bins=50, color='b', ls='-', histtype='step')
plt.hist(bootStat/computed_std, bins=50, color='r', histtype='step')
plt.legend(['Smooth Bootstrap/Standard Deviation','Bootstrap/Standard Deviation'], fontsize='small', loc='upper left')