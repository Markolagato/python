"""
zombies2
Mark Tierney
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

# data
zombieData = np.loadtxt("percentzombie.txt")
time = zombieData[:,0]
zombiePercent = zombieData[:,1]
humanPercent = 100 - zombiePercent

# plot initial data
plt.figure(0)
plt.clf()
plt.plot(time, humanPercent, 'b*', label="# of Humans")

# bayesian prior grids
slope = np.linspace(-14, -4, 1001)
inter = np.linspace(-5, 20, 251)

# priors and posterior prob
posterior = np.zeros((len(slope), len(inter)))
for i in range(len(slope)):
    for j in range(len(inter)):
        estHumanPercent = time * slope[i] + inter[j]
        resids = humanPercent - estHumanPercent
        chiSq = np.sum((resids)**2/(3.0)**2)
        priorUnif=(1.+slope[i]**2)**(-3/2.)
        posterior[i, j] = -1. * chiSq/2. + np.log(priorUnif)
#   use the same equation for the prior as before to create an
#   uninformed slope prior

# marginalize
margin = np.sum(np.exp(posterior), axis=0)/np.sum(np.exp(posterior))
#   marginalized over axis=0 because we care about the intercept
#   not the slope
plt.figure(1)
plt.clf()
plt.plot(inter, margin, 'r*')
#   at one sigma there are approximately 3% of humans left

# new prior
slope2 = np.linspace(-14, -4, 1001)
inter2 = np.linspace(1, 20, 191)

# priors and posterior prob
posterior2 = np.zeros((len(slope2), len(inter2)))
for i in range(len(slope2)):
    for j in range(len(inter2)):
        estHumanPercent2 = time * slope2[i] + inter2[j]
        resids2 = humanPercent - estHumanPercent2
        chiSq2 = np.sum((resids2)**2/(3.0)**2)
        priorUnif2 = (1. + slope2[i]**2)**(-3/2.)
        posterior2[i, j] = -1. * chiSq2/2. + np.log(priorUnif2)
#   use the same equation for the prior as before to create an
#   uninformed slope prior

# marginalize
margin2 = np.sum(np.exp(posterior2), axis=0)/np.sum(np.exp(posterior2))
#   marginalized over axis=0 because we care about the intercept
#   not the slope
plt.figure(1)
plt.plot(inter2, margin2, 'g.')
#   The probability of humans still being alive is much larger
#   with the new prior.

#   The Bayesian analysis has a much larger bandwidth than the MLE
#   approach we saw before, however the distribution is similar
#   and they both estimate a similar value for the intercept.



