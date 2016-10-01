"""
Zombie Activity: Part 2

Author: Mark Tierney
Last Edited: 9/23/16

Frequentist vs. Bayesian approach activity using the undead.
"""
# Import usual packages
import numpy as np
import matplotlib.pyplot as plt
#import numpy.random as npr

# Read in zombie data and plot as stars
data = np.loadtxt(r"percentzombie.txt")
time = data[:,0]
percHuman = 100 - data[:,1]
plt.figure(0)
plt.clf()
plt.plot(time,percHuman,'b*',markersize=10)
plt.xlabel("Time")
plt.ylabel("% Humans")
plt.xlim([-14,0])