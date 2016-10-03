"""
Histograms, Kernel Density Estimation, and Hypothesis Tests for Comparing Distributions

Author: Mark Tierney
Last Edited: 9/23/16

Frequentist vs. Bayesian approach activity using the undead.
"""
# Import usual packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astroML.plotting import hist
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
#import numpy.random as npr

# import data from csv
data = np.genfromtxt("ECO_DR1_withradec.csv", delimiter=",", dtype=None, names=True)
name = data['NAME']
radeg = data['RADEG']
decdeg = data['DEDEG']
grpcz = data['GRPCZ']
cz = data['CZ']

# define X,Y, and Z variables in cartesian
radeg_eq = radeg*np.cos(decdeg*(np.pi/180))
hubConst = 70
Z = cz/hubConst
X = Z*2*np.pi*(radeg_eq/360)
Y = Z*2*np.pi*(decdeg/360)
coord = np.transpose(np.array((X,Y,Z)))

Z_nopec = grpcz/hubConst
X_nopec = Z_nopec*2*np.pi*(radeg_eq/360)
Y_nopec = Z_nopec*2*np.pi*(decdeg/360)
coord_nopec = np.transpose(np.array((X_nopec,Y_nopec,Z_nopec)))

# KDE and Histograms for non-group corrected cz's

kdt = cKDTree(coord)
dist,inds = kdt.query(coord, k=2)
dist = dist[:,1]
inds = inds[:,1]

plt.figure(0)
plt.clf()
plt.hist(dist,bins='scott',histtype='step',normed=True,label='Scott')
plt.hist(dist,bins='fd',histtype='step',normed=True,label='Freedman-Diac')
hist(dist, bins='knuth', histtype='step',normed=True,label='Knuth')
hist(dist, bins='blocks', histtype='step',normed=True,label='Bayes. Blocks')
plt.legend()
# Scott's rule 

# Block spikes?

kde = KernelDensity(kernel='gaussian', bandwidth=.05).fit(dist[:,np.newaxis])
kde_samples = kde.score_samples(dist[:,np.newaxis])
plt.plot(dist[:,np.newaxis],np.exp(kde_samples),'k.',label='kde (.05)')

kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(dist[:,np.newaxis])
kde_samples = kde.score_samples(dist[:,np.newaxis])
plt.plot(dist[:,np.newaxis],np.exp(kde_samples),'g.',label='kde (.5)')

kde = KernelDensity(kernel='gaussian', bandwidth=.75).fit(dist[:,np.newaxis])
kde_samples = kde.score_samples(dist[:,np.newaxis])
plt.plot(dist[:,np.newaxis],np.exp(kde_samples),'y.',label='kde (.75)')

plt.title('Non-group cz')
plt.legend()
plt.show()

# KDE and Histogramd for group corrected (nopec) cz's
kdt_nopec = cKDTree(coord_nopec)
dist_nopec,inds_nopec = kdt_nopec.query(coord_nopec, k=2)
dist_nopec = dist_nopec[:,1]
inds_nopec = inds_nopec[:,1]

plt.figure(1)
plt.clf()
plt.hist(dist_nopec,bins='scott',histtype='step',normed=True,label='Scott')
plt.hist(dist_nopec,bins='fd',histtype='step',normed=True,label='Freedman-Diac')
hist(dist_nopec, bins='knuth', histtype='step',normed=True,label='Knuth')
hist(dist_nopec, bins='blocks', histtype='step',normed=True,label='Bayes. Blocks')
plt.legend()
'''
# Scott's rule?
# In the book plot scott's rule for a non-guassian does not demonstrate the distribution well
# however in our case it matched the non-guassian distribution as well as other binning 
# methods. 

# Block spikes?

kde_nopec = KernelDensity(kernel='gaussian', bandwidth=.05).fit(dist_nopec[:,np.newaxis])
kde_samples_nopec = kde_nopec.score_samples(dist_nopec[:,np.newaxis])
plt.plot(dist_nopec[:,np.newaxis],np.exp(kde_samples_nopec),'k.',label='kde_nopec (.05)')

kde_nopec = KernelDensity(kernel='gaussian', bandwidth=.5).fit(dist_nopec[:,np.newaxis])
kde_samples_nopec = kde_nopec.score_samples(dist_nopec[:,np.newaxis])
plt.plot(dist_nopec[:,np.newaxis],np.exp(kde_samples_nopec),'g.',label='kde_nopec (.5)')

kde_nopec = KernelDensity(kernel='gaussian', bandwidth=.75).fit(dist_nopec[:,np.newaxis])
kde_samples_nopec = kde_nopec.score_samples(dist_nopec[:,np.newaxis])
plt.plot(dist_nopec[:,np.newaxis],np.exp(kde_samples_nopec),'y.',label='kde_nopec (.75)')

plt.title('Group cz')
plt.legend()
plt.show()

# Subdividing samples
# compute the Y_nopec division for dec = 15(deg)

#Y_nopec_division = Z_nopec*2*np.pi*(15/360)
ind_divNorth = np.where(decdeg > 15)
ind_divSouth = np.where(decdeg < 15)
Y_nopec_North = Y_nopec[ind_divNorth]
Y_nopec_South = Y_nopec[ind_divSouth]
X_nopec_North = X_nopec[ind_divNorth]
X_nopec_South = X_nopec[ind_divSouth]
Z_nopec_North = Z_nopec[ind_divNorth]
Z_nopec_South = Z_nopec[ind_divSouth]

coord_nopec_North = np.transpose(np.array((X_nopec_North,Y_nopec_North,Z_nopec_North)))
coord_nopec_South = np.transpose(np.array((X_nopec_South,Y_nopec_South,Z_nopec_South)))

kdt_nopec_North = cKDTree(coord_nopec_North)
dist_nopec_North,inds_nopec_North = kdt_nopec_North.query(coord_nopec_North, k=2)
dist_nopec_North = dist_nopec_North[:,1]
inds_nopec_North = inds_nopec_North[:,1]

plt.figure(2)
plt.clf()
#plt.hist(dist_nopec_North,bins='scott',histtype='step',normed=True,label='Scott')
#plt.hist(dist_nopec_North,bins='fd',histtype='step',normed=True,label='Freedman-Diac')
hist(dist_nopec_North, bins='knuth', histtype='step',normed=True,label='Knuth Northern')
#hist(dist_nopec_North, bins='blocks', histtype='step',normed=True,label='Bayes. Blocks')

kdt_nopec_South = cKDTree(coord_nopec_South)
dist_nopec_South,inds_nopec_South = kdt_nopec_South.query(coord_nopec_South, k=2)
dist_nopec_South = dist_nopec_South[:,1]
inds_nopec_South = inds_nopec_South[:,1]


#plt.hist(dist_nopec_South,bins='scott',histtype='step',normed=True,label='Scott')
#plt.hist(dist_nopec_South,bins='fd',histtype='step',normed=True,label='Freedman-Diac')
hist(dist_nopec_South, bins='knuth', histtype='step',normed=True,label='Knuth Southern')
#hist(dist_nopec_South, bins='blocks', histtype='step',normed=True,label='Bayes. Blocks')

plt.title('Northern vs. Southern distributions of neighbor distances')
plt.legend()

print('Kolmogorov-Smirnov p-value: %0.3e' %ks_2samp(dist_nopec_North, dist_nopec_South)[1])
print('Mann-Whitney p-value: %0.3e\n' %mannwhitneyu(dist_nopec_North, dist_nopec_South)[1])

# The tiny p-values mean we can reject the null hypothesis that these came from the same
# data set. Since these are drawn from a data set divided into north and south that means
# that the idea of cosmic homogeneity is not completely accurate. The exclusion zone we 
# see is possibly an actual difference in galaxy cluster densities in these two regions
# of the sky. Therefore the two subsamples are not fair to compare, more meaningful
# data could be gathered from the divided hemisphere distributions.

# Subdividing
mask = np.zeros(np.size(data), dtype=bool)
mask[np.random.choice(np.size(data),size=np.size(data)/2,replace=False)]=True

coord_nopec_rand1 = np.transpose(np.array((X_nopec[mask],Y_nopec[mask],Z_nopec[mask])))
coord_nopec_rand2 = np.transpose(np.array((X_nopec[~mask],Y_nopec[~mask],Z_nopec[~mask])))

kdt_nopec_rand1 = cKDTree(coord_nopec_rand1)
dist_nopec_rand1,inds_nopec_rand1 = kdt_nopec_rand1.query(coord_nopec_rand1, k=2)
dist_nopec_rand1 = dist_nopec_rand1[:,1]
inds_nopec_rand1 = inds_nopec_rand1[:,1]

kdt_nopec_rand2 = cKDTree(coord_nopec_rand2)
dist_nopec_rand2,inds_nopec_rand2 = kdt_nopec_rand2.query(coord_nopec_rand2, k=2)
dist_nopec_rand2 = dist_nopec_rand2[:,1]
inds_nopec_rand2 = inds_nopec_rand2[:,1]

print('Kolmogorov-Smirnov for random p-value: %0.3f' % ks_2samp(dist_nopec_rand1, dist_nopec_rand2)[1])
print('Mann-Whitney for random p-value: %0.3f' % mannwhitneyu(dist_nopec_rand1, dist_nopec_rand2)[1])


input = np.load("crossvalidationflag.npz")
mask = input['mask']


'''





