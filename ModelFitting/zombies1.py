"""
zombies1
Mark Tierney
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

# zombie data
zombieData = np.loadtxt("percentzombie.txt")
time = zombieData[:,0]
zombiePercent = zombieData[:,1]
humanPercent = 100 - zombiePercent

# plot initial data
plt.figure(0)
plt.clf()
plt.plot(time, humanPercent, 'b*', label="# of Humans")

# MLE Slope/y-intercept
polyVals = np.polyfit(time, humanPercent, 1)
print("MLE Slope = %f" %polyVals[0])
print("y-intercept = %f\n" %polyVals[1])
x = np.linspace(-15, -4, 12)
y = np.polyval([polyVals[0],polyVals[1]], x)
plt.plot(x, y, 'g')
#   No, based on this slope I am not yet a zombie

# minimize resids in the x direction
polyVals2 = np.polyfit(humanPercent, time, 1) 
print("MLE Slope = %f" %(1./polyVals2[0]))
print("y-intercept = %f\n" %(-1.0 * polyVals2[1]/polyVals2[0]))
y2 = x * (1./polyVals2[0]) - (polyVals2[1]/polyVals2[0])
plt.plot(x, y2, 'r')
#   This slope is slightly more optimistic, so I am still not a zombie

# determining the chi^2 value
resid = humanPercent - np.polyval([polyVals[0],polyVals[1]], time)
plt.figure(1)
plt.clf()
plt.plot(time, resid, 'g*')
redChiSq = np.sum(resid**2/(3.0**2)) * (1./(len(humanPercent) - 2))
print("Reduced Chi Sq. (3.0) = %f" %redChiSq)
#   Not bad for the size of some of the resids, this is a rough linear fit

# new error = %5
redChiSq5 = np.sum(resid**2/(5.0**2)) * (1./(len(humanPercent) - 2))
print("Reduced Chi Sq. (5.0) = %f" %redChiSq5)
#   much smaller chi sq. means a much closer fit so smaller error was underestimate

# new error = %1
redChiSq1 = np.sum(resid**2/(1.0**2)) * (1./(len(humanPercent) - 2))
print("Reduced Chi Sq. (1.0) = %f\n" %redChiSq1)
#   much worse chi sq., probably not an overestimate

# higher order fit
poly2nd = np.polyfit(time, humanPercent, 2)
print("MLE A = %f" %poly2nd[0])
print("MLE B = %f" %poly2nd[1])
print("MLE C = %f\n" %poly2nd[2])
plt.figure(0)
plt.plot(time, np.polyval([poly2nd[0], poly2nd[1], poly2nd[2]], time), 'k*')
resid2 = humanPercent - np.polyval([poly2nd[0], poly2nd[1], poly2nd[2]], time)
plt.figure(1)
plt.plot(time, resid2, 'k.')
#   less spread on resids for the higher order fit

#resids for higher order
redChiSq2nd = np.sum(resid2**2/(5.0**2)) * (1./(len(humanPercent) - 3))
print("Reduced Chi Sq. 2nd Order (5.0) = %f" %redChiSq2nd)
#   Much better than any of the other residuals minimization we saw,
#   this data is likely higher order. Also means based on the plot that
#   I am already a zombie, dangit!




