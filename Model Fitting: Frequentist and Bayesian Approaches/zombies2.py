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

# Setup us grids for parameter space based on zombies 1.
slopePoss=np.arange(-20.,0,.2)
yintPoss=np.arange(0, 100, 10)

# What values to range over?
#   Knowing that our slope was negative and between zero and twenty and that
#   our y-int. can be positivie or negative we can set these ranges.

# Compute posterior distributions
gridsize = 20/.2
lnpostprob=np.zeros([100,10])
for i in xrange(100):
    for j in xrange(10):
        modelvals = slopePoss[i]*time+yintPoss[j]
        resids = (percHuman - modelvals)
        chisq = np.sum(resids**2 / 3**2)
        priorval = 1  # uniform prior
        lnpostprob[i,j] = (-1./2.)*chisq + np.log(priorval)      
        
#   The prior used for a flat posterior distribution is the uniform prior 1
        
# Marginalized Posterior Distribution
#   Marginalized over the time value (axis-1) because we want to determine
#   the percentage of humans today, therefore we want that distribution. The
#   yint parameter tells us nothing beyond the first day.
plt.figure(1)
plt.clf()
postprob=np.exp(lnpostprob)
marginalizedpprob_slope = np.sum(postprob,axis=1) / np.sum(postprob)
plt.plot(slopePoss,marginalizedpprob_slope,'r*',markersize=10)
plt.xlabel("Slope")
plt.ylabel("marginalized posterior distribution of slope")

#   The most likely % humans left today is ~2%.

# New prior: We are not yet a zombie.
yintPossNew=np.arange(0.1, 100.1, 10)
#   Now it is impossible for our intercept value to start lower than 0.1%

# Compute posterior distributions
gridsize = 20/.2
lnpostprob=np.zeros([100,10])
for i in xrange(100):
    for j in xrange(10):
        modelvals = slopePoss[i]*time+yintPossNew[j]
        resids = (percHuman - modelvals)
        chisq = np.sum(resids**2 / 3**2)
        priorval = 1  # uniform prior
        lnpostprob[i,j] = (-1./2.)*chisq + np.log(priorval)
        
postprob=np.exp(lnpostprob)
marginalizedpprob_slope = np.sum(postprob,axis=1) / np.sum(postprob)
plt.plot(slopePoss,marginalizedpprob_slope,'g*',markersize=10)

# How does the bayesian analysis compare to the MLE in the first ex.?
#   The bayesian analysis provides more realistic and logical values, such 
#   that we don't have to using reasoning to exclude data that is unrealistic.
#   It also allows for more detailed priors so we can enforce constraints on
#   model that we know must be true. Finally it does not use a fitting program
#   to determine a good fit from residuals.