# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
import matplotlib.pyplot as plt
import sys;sys.dont_write_bytecode = True;
from pyDOE import *
# Import Scikit_Learn Modules
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SE
from sklearn.gaussian_process.kernels import Matern as M
from sklearn.gaussian_process.kernels import ConstantKernel as sig_n
# Import Homemade Modules
from acq_funcs import *
################################################################################
print 'Seeding...'
#np.random.seed(10);                                      # Plant Random Seed
print 'Sampling points are being generated...'
# Random Sample Generation
bound   = np.array([[0,1]]);
X       = lhs(1,samples=4)
print X
#X       = np.random.uniform(bound[:,0],bound[:,1], (4,1));            # Random Sample Points  [nx1]
print X
y       = np.exp(X) + np.sin(3.*X) + 0.7*X + np.cos(4.*X);      # Random Sample Targets [nx1]
# Grid Creation
print 'Grid is being generated...'
num     = 100;
x       = np.linspace(bound[:,0],bound[:,1],num).reshape(-1,1); # Grid Points to Predict [nx1]
y_plt   = np.exp(x) + np.sin(3.*x) + 0.7*x + np.cos(4.*x)
print 'Initializing GP Regressor...'
# Initialize the Gaussian Process Regressor
kernel  = sig_n(1.)**2 * SE(length_scale = [0.2])
GPR     = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 3);
# Imposing Sampling Points for Inference
print 'Imposing sampling points...'
for i in range(0,3):
    GPR.fit(X,y);
    mean, std = GPR.predict(x,return_std=True)

    #EI = Expected_Improvement(x,X,GPR)
    #index = np.where(EI == np.amax(EI))
    KG = Knowledge_Gradient(x,X,y,GPR,kernel)
    index = np.where(KG == np.amax(KG))
    xnew = x[index]
    X = np.vstack((X,xnew))
    ynew = np.exp(xnew) + np.sin(3.*xnew) + 0.7*xnew + np.cos(4.*xnew);
    y = np.vstack((y,ynew))

GPR.fit(X,y)
    #KG = Knowledge_Gradient(x,X,y,GPR,kernel)
plt.figure(0)
plt.plot(x,y_plt)
plt.scatter(X,y)
plt.plot(x,mean)
plt.figure(1)
plt.plot(x,KG)
#plt.plot(x,EI)
plt.show()
