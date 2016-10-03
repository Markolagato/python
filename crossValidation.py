"""
Cross Validation

Author: Mark Tierney
Last Edited: 10/1/16

Using cross validation to determine ideal bandwidth for kernel density estimates.
"""
# Import usual packages
import numpy as np
import astroML as astro
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astroML.plotting import hist
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu

# import U-R data from Eco survey
data = np.genfromtxt("ECO_DR1_withradec.csv", delimiter=",", dtype=None, names=True)
ur_color = data['MODELU_RCORR']

binWidth_knuth = astro.density_estimation.knuth_bin_width(ur_color) # determine knuth bin widths for full sample

# Set up regular array of bandwidths
bdwidth = np.linspace(.01, binWidth_knuth, 10)

# create random training set (%50), test (25%), and cross val. (25%) sets
mask = np.zeros(np.size(ur_color), dtype=bool)
mask[np.random.choice(np.size(ur_color),size=(np.size(ur_color))/2,replace=False)]=True
ur_training = ur_color[mask]
ur_other = ur_color[~mask]
mask2 = np.zeros((np.size(ur_color)-1)/2, dtype=bool)
mask2[np.random.choice((np.size(ur_color)-1)/2,size=(np.size(ur_color)-1)/4,replace=False)]=True
ur_crossVal = ur_other[mask2]
ur_test = ur_other[~mask2]

# Kolmogroov-Smirnov to test whether random samples provide similar distributions
print('Kolmogorov-Smirnov for training set and other set p-value: %0.3f' % ks_2samp(ur_training, ur_other)[1])
print('Kolmogorov-Smirnov for test and cross val. sets p-value: %0.3f' % ks_2samp(ur_crossVal, ur_test)[1])

sum_log_like = np.zeros(len(bdwidth))

for i in range(len(bdwidth)):
    kde_train = KernelDensity(kernel='gaussian', bandwidth=bdwidth[i]).fit(ur_training[:,np.newaxis])
    kde_samples = kde_train.score_samples(ur_training[:,np.newaxis])
    
    kde_crossVal = KernelDensity(kernel='gaussian', bandwidth=.5).fit(kde_samples[:,np.newaxis])
    kde_samples_crossVal = kde_train.score_samples(ur_crossVal[:,np.newaxis])

    sum_log_like[i] = (1./len(ur_color))*np.sum(np.log(np.exp(kde_samples_crossVal)))

opt_band_ind = (sum_log_like.tolist()).index(np.max(sum_log_like))
opt_bdwidth = bdwidth[opt_band_ind]

print('Optimum bandwidth by cross-val: %0.3f' % opt_bdwidth)

# to assess error, calculating KDE for test and training sets
kde_train_err = KernelDensity(kernel='gaussian', bandwidth=opt_bdwidth).fit(ur_training[:,np.newaxis])
kde_test_err = KernelDensity(kernel='gaussian', bandwidth=opt_bdwidth).fit(ur_test[:,np.newaxis])
kde_samples_train_err = kde_train.score_samples(ur_training[:,np.newaxis])
kde_samples_test_err = kde_train.score_samples(ur_test[:,np.newaxis])

plt.figure(0)
plt.clf()
plt.plot(ur_training[:,np.newaxis],np.exp(kde_samples_train_err),'r.',label='KDE Opt. Bandwidth (Training)')
plt.plot(ur_test[:,np.newaxis],np.exp(kde_samples_test_err),'b.',label='KDE Opt. Bandwidth (Test)')
plt.title('Error Assessment Plot')
plt.legend(loc='upper left')

# Swapping Sets to test variation
sum_log_like = np.zeros(len(bdwidth))

for i in range(len(bdwidth)):
    kde_train = KernelDensity(kernel='gaussian', bandwidth=bdwidth[i]).fit(ur_other[:,np.newaxis])
    kde_samples = kde_train.score_samples(ur_other[:,np.newaxis])
    
    kde_crossVal = KernelDensity(kernel='gaussian', bandwidth=.5).fit(kde_samples[:,np.newaxis])
    kde_samples_crossVal = kde_train.score_samples(ur_test[:,np.newaxis])

    sum_log_like[i] = (1./len(ur_color))*np.sum(np.log(np.exp(kde_samples_crossVal)))

opt_band_ind = (sum_log_like.tolist()).index(np.max(sum_log_like))
opt_bdwidth = bdwidth[opt_band_ind]

print('Optimum bandwidth by cross-val: %0.3f' % opt_bdwidth)

# to assess error, calculating KDE for test and training sets
kde_train_err = KernelDensity(kernel='gaussian', bandwidth=opt_bdwidth).fit(ur_other[:,np.newaxis])
kde_test_err = KernelDensity(kernel='gaussian', bandwidth=opt_bdwidth).fit(ur_crossVal[:,np.newaxis])
kde_samples_train_err = kde_train.score_samples(ur_other[:,np.newaxis])
kde_samples_test_err = kde_train.score_samples(ur_crossVal[:,np.newaxis])


plt.plot(ur_other[:,np.newaxis],np.exp(kde_samples_train_err),'g.',label='KDE Opt. Bandwidth (Training Flipped)')
plt.plot(ur_test[:,np.newaxis],np.exp(kde_samples_test_err),'k.',label='KDE Opt. Bandwidth (Test Flipped)')
plt.title('Error Assessment Plot')
plt.legend(loc='upper left')

# There is not much variation between the graphs when the random samples that compose
# our test, cross validation, and training sets are all swapped. This is desired as 
# they should all be drawing from the same initial distribution and we'd expect the 
# training set to produce the same density esimate with little variation depending 
# on its training set.



