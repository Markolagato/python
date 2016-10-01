"""
Zombie Activity: Part 1

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

# Calculate MLE slope & y-int plotted as green line
slopeEst=(np.mean(time)*np.mean(percHuman)-np.mean(time*percHuman)) / \
   (np.mean(time)**2 -np.mean(time**2))
yintEst=np.mean(percHuman)-slopeEst*np.mean(time)

print("analytical MLE slope = %0.7f" %slopeEst)
print("analytical MLE y-intercept = %0.7f" %yintEst)

# Overplot the MLE ("best fit") solution
yfitvals=time*slopeEst+yintEst
plt.plot(time,yfitvals,'g')

# What does the y-int mean? Am I a zombie?
#   The y-int means that there is a chance that I am not yet a zombie because
#   total zombification should occur four days from now.

# Using polyfit to minimize the residual in the time-direction
pfit = np.polyfit(percHuman, time, 1)
slopeEst_Xres = 1./pfit[0]
yintEst_Xres = pfit[1]*(-1.*slopeEst_Xres)
yfitvals_Xres = time*slopeEst_Xres + yintEst_Xres
plt.plot(time, yfitvals_Xres, 'r')

yintEst_Xres=np.mean(percHuman)-slopeEst_Xres*np.mean(time)
print("analytical MLE y-intercept (min. X res) = %0.7f" %yintEst_Xres)

# Did this change my conclusion about being a zombie?
#   Yes, this slightly changed my conclusion about being a zombie,.if complete
#   zombification occured three days ago. Using the minimalized residuals in 
#   the y variable gives the most accurate prediction of complete zombification
#   because I am still alive, however, if there is a non-linear model that
#   matches our data it appears that maybe zombification will have occured 
#   before today.

# Uncertainty Measurements
plt.figure(1)
plt.clf()
upperRes = percHuman + 0.03 * percHuman
plt.plot(time, percHuman, 'b*', time, upperRes, 'g*')
chiSqr = np.sum((percHuman - yfitvals)**2/3**2)
red_chiSqr = chiSqr / 9;
print("reduced chi^2 value = %0.7f" %red_chiSqr)

# Is model a good fit for the data?
#   No, our chi sqr value is very large, and even visually it is easy to tell
#   that a linear model is not a good fit for this data. If 3% is an over
#   estimate then our chiSqr value will be smaller and the opposite for am
#   under estimate.

# Increasing the order of the fit
plt.figure(0)
pfit_higher = np.polyfit(time,percHuman,2)
yvalfits_higher = (time**2)*pfit_higher[0] + (time)*pfit_higher[1] + pfit_higher[2]
plt.plot(time, yvalfits_higher, 'k*')
resfit_higher = np.polyfit(time, upperRes, 2)
res_high = resfit_higher[0]*time**2 + resfit_higher[1]*time + resfit_higher[2]
plt.figure(1)
plt.plot(time, res_high, 'k*')

#   The residuals in the higher fit become a bit larger but converge
#   on the plot faster than the other lower fits.

chiSqr_higher = np.sum((percHuman - yvalfits_higher)**2/3**2)
red_chiSqr_higher = chiSqr_higher / 9;
print("reduced chi^2 value for higher fit = %0.7f" %red_chiSqr_higher)

# The chiSqr value for the higher fits is much closer to 1 than the lower
# fit, so we trust this model much more. It does however, sadly indicate that I
# am very likely a zombie.
