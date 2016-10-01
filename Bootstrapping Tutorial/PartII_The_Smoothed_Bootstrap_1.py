# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:21:22 2016

@author: markiemt

The Smoothed Bootstrap:

Part II of the bootstrapping tutorial.
"""

#Normal necessary inputs
from __future__ import division
import numpy as np
import numpy.random as npr
#from matplotlib import pyplot as plt
from astroML.resample import bootstrap

#-----------------------------------------------------------------------------#
'''
Smoothed Bootstrap Part II: 1
'''
#Input constants
sigma = 1
n = 1000 #number of bootstraps

#Create input data using a normal distribution
data = npr.normal(0, sigma, 5)

# Calculate sigma from np.std
sgmComputed = np.std(data, ddof=1)

# Bootstrap sigma
user_Stat = np.std
sgmBootArr = bootstrap(data, n, user_Stat, kwargs=dict(axis=1, ddof=1))
#plt.figure(0)
#plt.hist(sgmBootArr,bins=50)
sgmBoot = np.median(sgmBootArr)

print "Sigma_True = %f" %sigma
print "Sigma_Comp. = %f" %sgmComputed
print "Sigma_Boot = %f" %sgmBoot

