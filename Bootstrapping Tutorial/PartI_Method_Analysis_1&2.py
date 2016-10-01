"""
Correlation estimates
---------------------
Figure 3.24.

Bootstrap estimates of the distribution of Pearson's, Spearman's, and Kendall's
correlation coefficients based on 2000 resamplings of the 1000 points shown
in figure 3.23. The true values are shown by the dashed lines. It is clear
that Pearson's correlation coefficient is not robust to contamination.
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general

'''
Bootstrapping Tutorial Part I: Method Analysis
    Answers to #1 at the bottom of this code
'''

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.stats.random import bivariate_normal

# percent sign must be escaped if usetex=True
import matplotlib
if matplotlib.rcParams.get('text.usetex'):
    pct = '\%'
else:
    pct = '%'

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# Set parameters for the distributions
Nbootstraps = 2000
N = 1000

sigma1 = 2.0
sigma2 = 1.0
mu = (10.0, 10.0)
alpha_deg = 45.0
alpha = alpha_deg * np.pi / 180
f = 0.01

#------------------------------------------------------------
# Create a function to compute the statistics.  Since this
#CHG: Moved up in program, (i.e. declare functions before main code)
def compute_results(N, Nbootstraps):
    results = np.zeros((3, 2, Nbootstraps))

    for k in range(Nbootstraps):
        ind = np.random.randint(N, size=N)
        for j, data in enumerate([X, X_outlier]):
            x = data[ind, 0]
            y = data[ind, 1]
            for i, statistic in enumerate([stats.pearsonr,
                                           stats.spearmanr,
                                           stats.kendalltau]):
                results[i, j, k] = statistic(x, y)[0]

    return results

#------------------------------------------------------------
# sample the distribution
# without outliers and with outliers
np.random.seed(0)
X = bivariate_normal(mu, sigma1, sigma2, alpha, N)

X_outlier = X.copy()
X_outlier[:int(f * N)] = bivariate_normal(mu, 2, 5,
                                      45 * np.pi / 180., int(f * N))

# true values of rho (pearson/spearman r) and tau
# tau value comes from Eq. 41 of arXiv:1011.2009
rho_true = 0.6
tau_true = 2 / np.pi * np.arcsin(rho_true)


#------------------------------------------------------------
# Calculate results for given data set inputs
results = compute_results(N, Nbootstraps)

# Plot the results in a three-panel plot
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.25)

histargs = (dict(alpha=0.5, label='No Outliers'),
            dict(alpha=0.8, label='%i%s Outliers' % (int(f * 100), pct)))

distributions = ['Pearson-r', 'Spearman-r', r'Kendall-$\tau$']
xlabels = ['r_p', 'r_s', r'\tau']\

for i in range(3):
    ax = fig.add_subplot(311 + i)
    for j in range(2):
        ax.hist(results[i, j], 40, histtype='stepfilled', fc='gray',
                normed=True, **histargs[j])

    if i == 0:
        ax.legend(loc=2)

    ylim = ax.get_ylim()
    if i < 2:
        ax.plot([rho_true, rho_true], ylim, '--k', lw=1)
        ax.set_xlim(0.34, 0.701)
    else:
        ax.plot([tau_true, tau_true], ylim, '--k', lw=1)
        ax.set_xlim(0.31, 0.48)
    ax.set_ylim(ylim)

    ax.text(0.98, 0.95, distributions[i], ha='right', va='top', 
            transform=ax.transAxes, bbox=dict(fc='w', ec='w'))

    ax.set_xlabel('$%s$' % xlabels[i])
    ax.set_ylabel('$N(%s)$' % xlabels[i])

plt.show()

'''
1.) The source code for astroML's bootstrap function uses an array of data provided
by the users and then randomy generates a number of indices based on the number of
data points and also based on how many bootstrapped samples you wish to compute.
It then sets a new array of bootstrapped samples equal to samples from your data
provided, with replacement, meaning that though it is drawing from one data set, 
it will return one with differing values of each data point. The number of times
you wish this operation to occur is based on your input (they reccommend 1000-10000).
It then returns the bootstrapped function.
    The figure 3.24 code samples from a bivariate distribution with outliers and without
by changing one of the sigmas in the bivariate sampling. We then select our indices to 
draw from using a random integer, then for each data point we draw a new sample (with replacements)
from our data set with outliers. We also draw the statistics for each of these points from
the spearman, pearson, and kendall tau values of each selected data point. Those results
are returned and then this is repeated for the number of bootstrappings you desire.
    The two methods are different in that first takes your input data, turns it into an array,
selects the random indices as an array of size=(#ofBootstraps, #ofSamples), and then does
one array operation and returns the statistics to the user as an array. The second method uses for
loops to iterate for the desired number of bootstraps the user wants to perform, creating
an array to store the results in for the three statistics, selecting the random indices for 
that iteration, using those to get the new sampled data points and using those new sampled data points
to generate the statistics for that bootstrap. One big difference is the use of for
loops which slows down execution time, but is necessary for the second method because
we are using two different inputs (X_out, X) and because we want to retrieve three
statistics for each input for Nbootstraps iterations. The main difference is that 
for the astroML code the bootstrap can only take in user data with one statistic, while
this code uses a bivariate input and can bootstrap over two variables.
'''
