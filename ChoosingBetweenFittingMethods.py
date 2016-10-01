"""
Choosing Between Fitting Methods and Scatter Estimators
 in the Frequentist Paradigm
 
 Author: Mark Tierney
 Date: 9/26/16
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from operator import add

# Construct Data Sets
# No errors or Biases
X = np.linspace(1,10,100)
Y = np.linspace(20,40,100)

# Add in ten random systematic errors along with a normal gaussian scatter.
Y_gaussScat = map(add,Y,npr.normal(scale=1.0,size=(100,1)))
systemErr = np.zeros([np.size(Y),1])
for x in xrange(10):
    i = x+npr.randint(0,89,size=1)[0]
    systemErr[i] = systemErr[i] + 3*npr.rand(1)[0]
Y = map(add,Y_gaussScat,systemErr)


# Plot X vs. Y
plt.figure(0)
plt.clf()
plt.plot(X,Y,'r')

# Forward Fits
slopeEst=(np.mean(X)*np.mean(Y)-np.mean(X*Y)) / \
   (np.mean(X)**2 -np.mean(X**2))
yintEst=np.mean(Y)-slopeEst*np.mean(X)

Y_forward = slopeEst*X + yintEst

plt.figure(0)
plt.plot(X, Y_forward,'b')