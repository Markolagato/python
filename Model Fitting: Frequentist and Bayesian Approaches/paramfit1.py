"""
Part I activity in Parameter Fitting Tutorial
Modified by Kathleen Eckert from an activity written by Sheila Kannappan
June 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

# Generating fake data set to start with:
alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma (amplitude of errors)

narr=100 # number of data points
xvals = np.arange(narr) + 1. # xvals range from 1-51
yvals = alphatrue*xvals + betatrue + npr.normal(0,errs,narr) # yvals 
# What aspect of a real data set does npr.normal emulate here?
#   The outliers of a data set and some sort of actual distribution.
# What assumption is made here that is key to the least squares approach?
#   The assumption is that the uncertainty in both xvals and yvals is the same,
#   since yvals is calculated using xvals.

# Plot fake data
plt.figure(1) 
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)
plt.xlabel("x-values")
plt.ylabel("y-values")

# Determine slope & y-intercept using least squares analytic solution 

alphaest=(np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
   (np.mean(xvals)**2 -np.mean(xvals**2)) #  from derivation
betaest=np.mean(yvals)-alphaest*np.mean(xvals) # calculate estimate of y-intercept from derivation
# Why must we use alphaest rather than alphatrue in the above formula?
#   Because normally we would not know alphatrue.

# The MLE values of the slope and y-intercept are equivalent to the least
# squares fit results.
print("analytical MLE slope = %0.7f" %alphaest)
print("analytical MLE y-intercept = %0.7f" %betaest)

# Overplot the MLE ("best fit") solution
yfitvals=xvals*alphaest+betaest
plt.plot(xvals,yfitvals,'r')

# Compute analytic uncertainties on slope and y-intercept 

alphaunc = np.sqrt(np.sum((yvals - (alphaest*xvals+betaest))**2) / ((narr-2.)*(np.sum((xvals-np.mean(xvals))**2))))
betaunc = np.sqrt((np.sum((yvals - (alphaest*xvals+betaest))**2) / (narr-2.)) * ((1./narr) + (np.mean(xvals)**2)/np.sum((xvals-np.mean(xvals))**2)) )

print("analytical MLE uncertainty on alpha is %0.7f" % (alphaunc))
print("analytical MLE uncertainty on beta is %0.7f" % (betaunc))

print("fractional uncertainty on alpha is %0.7f" % (alphaunc/alphaest))
print("fractional uncertainty on beta is %0.7f" % (betaunc/betaest))
# Which parameter has larger fractional uncertainty?
#   betaest has a much larger fractional uncertainty.

# Solution using python solver np.polyfit
# third parameter is order of fit, 1 for linear
pfit = np.polyfit(xvals, yvals, 1)  # returns coeff. of highest order term first

print("               ") # put in some whitespace to make easier to read
print("np.polyfit MLE slope = %0.7f" %pfit[0])
print("np.polyfit MLE y-intercept = %0.7f" %pfit[1])

# Do you get the same result as in analytical case?
#    Yes, exactly the same result.
# Note that most problems do not have analytical solutions

# Can also obtain errors from the diagonal terms of the covariance
# matrix, which is the inverse of the Hessian matrix and
# can be computed in np.polyfit by setting cov='True'

pfit, covp = np.polyfit(xvals, yvals, 1, cov='True')  # returns coeff. of highest order term first
# setting cov='True' returns the covariance matrix
# how do we get the errors from it?
print("slope is %0.7f +- %0.7f" % (pfit[0], np.sqrt(covp[0,0])))
print("intercept is %0.7f +- %0.7f" % (pfit[1], np.sqrt(covp[1,1])))

# Are those errors the same as in analytical solution?
#   No, they are slightly larger for both the slope and the y-int.
# What happens to the uncertainties if you increase/decrease the number of points used in the fit (try N=100, N=10) ?
#   For fewer data points (10) the uncertainty in the y-int and the slope increased very significantly. 
#   For more data points (100) both the y-int and slope uncertainties decreased.
# What happens to the percentage difference between the analytical and numerical methods for computing the uncertanties if you increase/decrease the number of points (try N=100, N=10)?
anal_num_uncSlopeDiff = np.sqrt(covp[0,0])-(alphaunc/alphaest)
anal_num_uncYintDiff = np.sqrt(covp[1,1])-(betaunc/betaest)
print("Percentage difference between analytic/numeric slope unc. = %0.7f" %anal_num_uncSlopeDiff)
print("Percentage difference between analytic/numeric y-int unc. = %0.7f" %anal_num_uncYintDiff)
#   For fewer points (10) the difference between the analytic and numerical uncertainty methods for both slope and y-int is larger than
#   with our original 50 points by a factor of around 10 for the slope and 3 for the y-int. For more data points (100) the difference 
#   is smaller for both than the original number by a factor of about 1/4 for slope and 3/5 for the y-int.



