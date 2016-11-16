"""
Choosing Between Fitting Methods and Scatter Estimators
 in the Frequentist Paradigm
 
 Author: Mark Tierney
 Date: 9/26/16
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
#from operator import add

# Construct Data Sets
# No errors or Biases
X = np.linspace(1,10,100)
Y = np.linspace(20,40,100)

# Add in ten random systematic errors along with a normal gaussian scatter.
Y_gaussScat = Y+npr.normal(scale=1.0,size=(100))
systemErr = np.zeros([np.size(Y)])
for x in xrange(10):
    i = x+npr.randint(0,89,size=1)[0]
    systemErr[i] = systemErr[i] + 3*npr.rand(1)[0]
Y = Y_gaussScat+systemErr


# Plot X vs. Y
plt.figure(0)
plt.clf()
plt.plot(X,Y,'r')

# Forward Fits
slopeEst_for= np.polyfit(X, Y, 1)[0]
yintEst_for= np.polyfit(X, Y, 1)[1]

Y_forward = slopeEst_for*X + yintEst_for

plt.figure(0)
plt.plot(X, Y_forward,'b')

# Inverse Fits

slopeEst_inv = 1./(np.polyfit(Y, X, 1)[0])
yintEst_inv = -1.*np.polyfit(Y, X, 1)[1]/np.polyfit(Y, X, 1)[0]

Y_inverse = slopeEst_inv*X + yintEst_inv

plt.figure(0)
plt.plot(X, Y_inverse,'g')

# Plot scatter
plt.plot(X, Y, 'k')
plt.legend(('True','Forward','Inverse','True+Scatter'), loc='lower right')


# Bisector Fits
slopeEst_bis = (1./(slopeEst_for+slopeEst_inv))*(slopeEst_for*slopeEst_inv - 1. + np.sqrt((1.+slopeEst_for**2)*(1.+slopeEst_inv**2)))
yintEst_bis = np.mean(Y) - slopeEst_bis*np.mean(X)
Y_bisector = slopeEst_bis*X+yintEst_bis

plt.plot(X ,Y_bisector)
plt.legend(('True','Forward','Inverse','True+Scatter','Bisector'), loc='lower right')

# RMS and Biweight for each
RMS_for = np.sqrt(np.mean((Y_forward-Y)**2))
RMS_inv = np.sqrt(np.mean((Y_inverse-Y)**2))
RMS_bis = np.sqrt(np.mean((Y_bisector-Y)**2))

resmed = (((Y_forward-Y)-np.median(Y_forward-Y)))/(8.5*np.median(np.abs((Y_forward-Y)-np.median(Y_forward-Y))))
bi_for = np.sqrt(100)*np.sqrt(np.sum(((Y_forward-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

resmed = (((Y_inverse-Y)-np.median(Y_inverse-Y)))/(8.5*np.median(np.abs((Y_inverse-Y)-np.median(Y_inverse-Y))))
bi_inv = np.sqrt(100)*np.sqrt(np.sum(((Y_inverse-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))    

resmed = (((Y_bisector-Y)-np.median(Y_bisector-Y)))/(8.5*np.median(np.abs((Y_bisector-Y)-np.median(Y_bisector-Y))))
bi_bis = np.sqrt(100)*np.sqrt(np.sum(((Y_bisector-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

print "RMS_forward = %f and biweight_forward = %f" % (RMS_for,bi_for)
print "RMS_inverse = %f and biweight_inverse = %f" % (RMS_inv,bi_inv)
print "RMS_bisector = %f and biweight_bisector = %f" % (RMS_bis,bi_bis)

# Forward fit has lowest RMS meaning that the forward fit of the data represents
# the best minimization of the residuals. That makes sense, since all of the scatter
# we added was in the Y direction.

# Plot X vs. Y
plt.figure(1)
plt.clf()
plt.plot(X,Y,'r')

# X Scatter
X = X + npr.normal(0,3.0,100)

# Forward Fits
slopeEst_for= np.polyfit(X, Y, 1)[0]
yintEst_for= np.polyfit(X, Y, 1)[1]

Y_forward = slopeEst_for*X + yintEst_for

plt.figure(1)
plt.plot(X, Y_forward,'b')

# Inverse Fits

slopeEst_inv = 1./(np.polyfit(Y, X, 1)[0])
yintEst_inv = -1.*np.polyfit(Y, X, 1)[1]/np.polyfit(Y, X, 1)[0]

Y_inverse = slopeEst_inv*X + yintEst_inv

plt.figure(1)
plt.plot(X, Y_inverse,'g')

# Plot scatter
plt.plot(X, Y, 'k')
plt.legend(('True','Forward','Inverse','True+Scatter'), loc='lower right')


# Bisector Fits
slopeEst_bis = (1./(slopeEst_for+slopeEst_inv))*(slopeEst_for*slopeEst_inv - 1. + np.sqrt((1.+slopeEst_for**2)*(1.+slopeEst_inv**2)))
yintEst_bis = np.mean(Y) - slopeEst_bis*np.mean(X)
Y_bisector = slopeEst_bis*X+yintEst_bis

plt.plot(X ,Y_bisector)
plt.legend(('True','Forward','Inverse','True+Scatter','Bisector'), loc='lower right')

# RMS and Biweight for each
RMS_for = np.sqrt(np.mean((Y_forward-Y)**2))
RMS_inv = np.sqrt(np.mean((Y_inverse-Y)**2))
RMS_bis = np.sqrt(np.mean((Y_bisector-Y)**2))

resmed = (((Y_forward-Y)-np.median(Y_forward-Y)))/(8.5*np.median(np.abs((Y_forward-Y)-np.median(Y_forward-Y))))
bi_for = np.sqrt(100)*np.sqrt(np.sum(((Y_forward-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

resmed = (((Y_inverse-Y)-np.median(Y_inverse-Y)))/(8.5*np.median(np.abs((Y_inverse-Y)-np.median(Y_inverse-Y))))
bi_inv = np.sqrt(100)*np.sqrt(np.sum(((Y_inverse-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))    

resmed = (((Y_bisector-Y)-np.median(Y_bisector-Y)))/(8.5*np.median(np.abs((Y_bisector-Y)-np.median(Y_bisector-Y))))
bi_bis = np.sqrt(100)*np.sqrt(np.sum(((Y_bisector-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

print "X_scat RMS_forward = %f and biweight_forward = %f" % (RMS_for,bi_for)
print "X_scat RMS_inverse = %f and biweight_inverse = %f" % (RMS_inv,bi_inv)
print "X_scat RMS_bisector = %f and biweight_bisector = %f" % (RMS_bis,bi_bis)
# From looking at the plots it looks like the bisector has the best reduction of
# the residuals. With the true slope however, it looks like the inverse slope is closer.
# The better way to compute the RMS now would be to use the inverse method because
# there is more scatter in the X direction so it needs to be minimized more than the
# Y which is what the inverse method does.

# X direction comparison
slopeEst_xinv= np.polyfit(Y, X, 1)[0]
yintEst_xinv= np.polyfit(Y, X, 1)[1]

Y_xinverse = slopeEst_xinv*X + yintEst_xinv

plt.figure(2)
plt.plot(X, Y_xinverse,'b')

slopeEst_xfor = 1./(np.polyfit(X, Y, 1)[0])
yintEst_xfor = -1.*np.polyfit(X, Y, 1)[1]/np.polyfit(X, Y, 1)[0]

Y_xforward = slopeEst_xfor*X + yintEst_xfor

plt.figure(2)
plt.plot(X, Y_xforward,'g')

# Plot scatter
plt.plot(X, Y, 'k')
plt.legend(('True','Forward','Inverse','True+Scatter'), loc='lower right')


# Bisector Fits
slopeEst_xbis = (1./(slopeEst_xfor+slopeEst_xinv))*(slopeEst_xfor*slopeEst_xinv - 1. + np.sqrt((1.+slopeEst_xfor**2)*(1.+slopeEst_xinv**2)))
yintEst_xbis = np.mean(Y) - slopeEst_xbis*np.mean(X)
Y_xbisector = slopeEst_xbis*X+yintEst_xbis

# RMS and Biweight for each
RMS_xfor = np.sqrt(np.mean((Y_xforward-Y)**2))
RMS_xinv = np.sqrt(np.mean((Y_xinverse-Y)**2))
RMS_xbis = np.sqrt(np.mean((Y_xbisector-Y)**2))

resmed = (((Y_xforward-Y)-np.median(Y_xforward-Y)))/(8.5*np.median(np.abs((Y_xforward-Y)-np.median(Y_xforward-Y))))
bi_xfor = np.sqrt(100)*np.sqrt(np.sum(((Y_xforward-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

resmed = (((Y_xinverse-Y)-np.median(Y_xinverse-Y)))/(8.5*np.median(np.abs((Y_xinverse-Y)-np.median(Y_xinverse-Y))))
bi_xinv = np.sqrt(100)*np.sqrt(np.sum(((Y_xinverse-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))    

resmed = (((Y_xbisector-Y)-np.median(Y_xbisector-Y)))/(8.5*np.median(np.abs((Y_xbisector-Y)-np.median(Y_xbisector-Y))))
bi_xbis = np.sqrt(100)*np.sqrt(np.sum(((Y_xbisector-Y)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

print "X_scat RMS_forward = %f and biweight_forward = %f" % (RMS_xfor,bi_xfor)
print "X_scat RMS_inverse = %f and biweight_inverse = %f" % (RMS_xinv,bi_xinv)
print "X_scat RMS_bisector = %f and biweight_bisector = %f" % (RMS_xbis,bi_xbis)
# Inverse fit looks much more likely.
# Other two methods not as good as inverse and similar because neither are minimizing
# the residuals fully of the most common error source

# Selection Bias
select = np.where(X > 3.)
X_sel = X[select]
Y_sel = Y[select]

# Fits for all three again
# Plot X vs. Y
plt.figure(3)
plt.clf()
plt.plot(X_sel,Y_sel,'r')

# Forward Fits
slopeEst_for= np.polyfit(X_sel, Y_sel, 1)[0]
yintEst_for= np.polyfit(X_sel, Y_sel, 1)[1]

Y_forward = slopeEst_for*X_sel + yintEst_for

plt.figure(3)
plt.plot(X_sel, Y_forward,'b')

# Inverse Fits

slopeEst_inv = 1./(np.polyfit(Y_sel, X_sel, 1)[0])
yintEst_inv = -1.*np.polyfit(Y_sel, X_sel, 1)[1]/np.polyfit(Y_sel, X_sel, 1)[0]

Y_inverse = slopeEst_inv*X_sel + yintEst_inv

plt.figure(3)
plt.plot(X_sel, Y_inverse,'g')

# Plot scatter
plt.plot(X_sel, Y_sel, 'k')
plt.legend(('True','Forward','Inverse','True+Scatter'), loc='lower right')


# Bisector Fits
slopeEst_bis = (1./(slopeEst_for+slopeEst_inv))*(slopeEst_for*slopeEst_inv - 1. + np.sqrt((1.+slopeEst_for**2)*(1.+slopeEst_inv**2)))
yintEst_bis = np.mean(Y_sel) - slopeEst_bis*np.mean(X_sel)
Y_bisector = slopeEst_bis*X_sel+yintEst_bis

plt.plot(X_sel ,Y_bisector)
plt.legend(('True','Forward','Inverse','True+Scatter','Bisector'), loc='lower right')

# RMS and Biweight for each
RMS_for = np.sqrt(np.mean((Y_forward-Y_sel)**2))
RMS_inv = np.sqrt(np.mean((Y_inverse-Y_sel)**2))
RMS_bis = np.sqrt(np.mean((Y_bisector-Y_sel)**2))

resmed = (((Y_forward-Y_sel)-np.median(Y_forward-Y_sel)))/(8.5*np.median(np.abs((Y_forward-Y_sel)-np.median(Y_forward-Y_sel))))
bi_for = np.sqrt(100)*np.sqrt(np.sum(((Y_forward-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

resmed = (((Y_inverse-Y_sel)-np.median(Y_inverse-Y_sel)))/(8.5*np.median(np.abs((Y_inverse-Y_sel)-np.median(Y_inverse-Y_sel))))
bi_inv = np.sqrt(100)*np.sqrt(np.sum(((Y_inverse-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))    

resmed = (((Y_bisector-Y_sel)-np.median(Y_bisector-Y_sel)))/(8.5*np.median(np.abs((Y_bisector-Y_sel)-np.median(Y_bisector-Y_sel))))
bi_bis = np.sqrt(100)*np.sqrt(np.sum(((Y_bisector-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

print "RMS_forward = %f and biweight_forward = %f" % (RMS_for,bi_for)
print "RMS_inverse = %f and biweight_inverse = %f" % (RMS_inv,bi_inv)
print "RMS_bisector = %f and biweight_bisector = %f" % (RMS_bis,bi_bis)

# X direction comparison again
slopeEst_xinv= np.polyfit(Y_sel, X_sel, 1)[0]
yintEst_xinv= np.polyfit(Y_sel, X_sel, 1)[1]

Y_xinverse = slopeEst_xinv*X_sel + yintEst_xinv

plt.figure(4)
plt.plot(X_sel, Y_xinverse,'b')

slopeEst_xfor = 1./(np.polyfit(X_sel, Y_sel, 1)[0])
yintEst_xfor = -1.*np.polyfit(X_sel, Y_sel, 1)[1]/np.polyfit(X_sel, Y_sel, 1)[0]

Y_xforward = slopeEst_xfor*X_sel + yintEst_xfor

plt.figure(4)
plt.plot(X_sel, Y_xforward,'g')

# Plot scatter
plt.plot(X_sel, Y_sel, 'k')
plt.legend(('True','Forward','Inverse','True+Scatter'), loc='lower right')


# Bisector Fits
slopeEst_xbis = (1./(slopeEst_xfor+slopeEst_xinv))*(slopeEst_xfor*slopeEst_xinv - 1. + np.sqrt((1.+slopeEst_xfor**2)*(1.+slopeEst_xinv**2)))
yintEst_xbis = np.mean(Y_sel) - slopeEst_xbis*np.mean(X_sel)
Y_xbisector = slopeEst_xbis*X_sel+yintEst_xbis

# RMS and Biweight for each
RMS_xfor = np.sqrt(np.mean((Y_xforward-Y_sel)**2))
RMS_xinv = np.sqrt(np.mean((Y_xinverse-Y_sel)**2))
RMS_xbis = np.sqrt(np.mean((Y_xbisector-Y_sel)**2))

resmed = (((Y_xforward-Y_sel)-np.median(Y_xforward-Y_sel)))/(8.5*np.median(np.abs((Y_xforward-Y_sel)-np.median(Y_xforward-Y_sel))))
bi_xfor = np.sqrt(100)*np.sqrt(np.sum(((Y_xforward-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

resmed = (((Y_xinverse-Y_sel)-np.median(Y_xinverse-Y_sel)))/(8.5*np.median(np.abs((Y_xinverse-Y_sel)-np.median(Y_xinverse-Y_sel))))
bi_xinv = np.sqrt(100)*np.sqrt(np.sum(((Y_xinverse-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))    

resmed = (((Y_xbisector-Y_sel)-np.median(Y_xbisector-Y_sel)))/(8.5*np.median(np.abs((Y_xbisector-Y_sel)-np.median(Y_xbisector-Y_sel))))
bi_xbis = np.sqrt(100)*np.sqrt(np.sum(((Y_xbisector-Y_sel)**2*(1-resmed**2)**4)))/np.abs(np.sum(((1-resmed**2)*(1-5*resmed**2))))

print "X_scat RMS_forward = %f and biweight_forward = %f" % (RMS_xfor,bi_xfor)
print "X_scat RMS_inverse = %f and biweight_inverse = %f" % (RMS_xinv,bi_xinv)
print "X_scat RMS_bisector = %f and biweight_bisector = %f" % (RMS_xbis,bi_xbis)

# Very high RMS and biweights with bisector being the lowest. The selection bias
# throws off the error modeling. We could do the same above code for Y, with similar
# results as for X.














