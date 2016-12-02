"""
This is a template code for ASTR 503/703. It is intended to illustrate
standard imports and header information, while providing practice in
debugging, speed optimization, and spotting bad habits in programming.

This code runs but is deeply flawed. Perform the four tasks below to
learn something about both programming and the Central Limit Theorem.

Author: Sheila Kannappan
Created: August 2016
"""

# standard imports and naming conventions; uncomment as needed
import numpy as np              # basic numerical analysis
import matplotlib.pyplot as plt # plotting
#import scipy as sp              # extended scientific function
import scipy.stats as stats     # statistical functions
#import numpy.random as npr      # random number generation
#import astropy as ap            # core astronomy library
#import astroML as ml            # machine learning for astronomy
#import astroML.datasets as mld  # datasets
#import pymc                     # bayesian package with MCMC
import pdb                      # python debugger
import time                     # python timekeeper
#plt.ion()                       # use if working in ipython under linux

# if any package above does not import properly, then you need to
# revisit your anaconda installation

"""
Just for fun, this code will explore the Central Limit Theorem by comparing 
Poisson distributions with Gaussian distributions.

An example of a Poisson process is counting the # of people who use the gym per
hour where the count is run for nhr = different times. We assume the underlying
average users per hour U is fixed but the counts have "Poisson fluctuations"
so N = nhr x U specifies the *expected* count (the mean of the theoretical
Poisson distribution) not the observed count Nobs, whose possible values have 
different probabilities following a Poisson distribution with mean N. (In fact
nhr may need to be very large for N to exactly equal Nobs, because Nobs is by
definition an integer whereas U is by definition a real number.)

Statistical theory tells us that for a Poisson process, the observed count Nobs 
fluctuates around the true theoretical mean N with a 68% confidence interval of 
+-sqrt(N) for "large N". The sleight of hand of statistics is to use the
observed data to estimate N as Nobs and likewise estimate the 68% confidence 
interval as +-sqrt(Nobs). Thus the estimated fractional error in the count is 
fracerr = sqrt(Nobs)/Nobs = 1/sqrt(Nobs). The 1/sqrt explains why we get a 
better estimate of U by running the count for 10 hours rather than 1 hour. 
However in the exercise below we will not use data but simply compare the 
theoretical distributions while increasing N (or equivalently, increasing nhr).

According to the Central Limit Theorem, as we increase N, the Poisson 
distribution should start to look like a Gaussian. Therefore we will plot the
Poisson distribution for increasing N and overplot Gaussians with the same mean
N and 68% confidence interval +-sqrt(N), to see how quickly the Poisson shape
approaches a Gaussian shape (i.e., when are we in the "large N" limit).
"""
#pdb.set_trace()
    
def gaussfunc(xvals_gauss, mean_gauss, sigma_gauss):
    y = np.exp(-1.*(((xvals_gauss-mean_gauss)**2) / (2.* sigma_gauss**2)))
#   moved sigma outside of sqrt to avoid extra math and used "**(1./2.)"
#   which is faster than np.sqrt()
    y = y/(sigma_gauss * (2. * np.pi)**(1./2.))
    return y
    
#   Moved function to before main code for clarity
def poissonfunc(xvals_pois, mean_pois):
#   Removed for loop to speed up code
    prob = stats.poisson.pmf(xvals_pois, mean_pois)
    return prob

users = 8. # underlying rate of gym users per hour
num_count = np.array([6, 36, 216, 1296]) # total number of people counted (powers of 6)
nhr = num_count/users # time to count this many people
#   Changed n and N to num_count and count for clarity/fix pdb problems
#labelarr = ["count for %s hr" % ihr for ihr in nhr]
pdb.set_trace()
plt.figure(0)
for i in xrange(0, len(num_count)):
    
    # plot probabilities of count values for range around mean
    mean = num_count[i]
    maxval = 2*mean
    xvals=np.arange(0, maxval)
    prob = poissonfunc(xvals, mean)
    plt.plot(xvals, prob, 'r', lw=3)
    plt.xlabel("count value")
    plt.ylabel("probability")
    plt.xscale("log")
    sel = np.where(prob == max(prob))
    count = xvals[sel[0][0]]
    probval = prob[sel[0][0]]
    label = "count for %s hr" % (nhr[i])
    plt.text(count, probval, label)
# plot Gaussian distribution with matching mean and sigma
#   Used time.clock() to determine that **(1/2) is on average
#   faster than using np.sqrt()
    sigma=mean**(1./2.)
    y = gaussfunc(xvals, mean, sigma)
    plt.plot(xvals, y, 'b')


"""
Task 1: Many times a code runs fine, but the output may be wrong; you 
step through it line by line to make sure it's doing what you think it 
should be doing. There are several errors in the program above. Try 
to find them using pdb.set_trace() as described in the tutorial here:
https://pythonconquerstheuniverse.wordpress.com/category/python-debugger/
Check the size and contents of the variables at each step to determine 
whether they make sense. Useful commands include print, len(), and 
np.size(). WATCH OUT: the very first bug you need to find is one that
makes pdb not even work properly -- why does this code mess up how
the next line ("n") command works in pdb?

Task 2: We don't always want to optimize code speed -- sometimes it's
just not important -- but you should be in the habit of avoiding 
silly things that slow your code down, like unnecessary loops or math
operations. Use time.clock() to measure the time taken by the whole code,
and each part of the code, above and try to find inefficiencies. When 
you find a slow step, ask yourself whether it could be faster, and 
whether it matters (is it the rate-limiting step?). For now, fix it 
even if it's not the rate-limiting step, just for practice. Overall,
you should be able to speed up this code by about a factor of 10.

Task 3: Some things in the code above represent poor programming practice,
even though they do not affect speed and are not bugs. Note examples and
correct them.

Task 4: Once you've got the code fixed up, you can play with the zoom in the
plot window to see how closely the Poisson and Gaussian distributions match 
each other for each value of N. If they match well, does that mean the 
fractional error in the observed count must be small? Explain.

Yes, the better fit between the distributions does mean that the fractional 
error is getting smaller since it is 1/sqrt(Nobs) and Nobs increases
every hour, the longer the better.
"""
