# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
import sys;sys.dont_write_bytecode = True;
# Import Scikit_Learn Modules
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SE
from sklearn.gaussian_process.kernels import ConstantKernel as sig_n
# Import Homemade Modules
from GP_Plot_Reg import *
################################################################################
print 'Seeding...'
np.random.seed(10);                                      # Plant Random Seed
print 'Sampling points are being generated...'
# Random Sample Generation
bound   = np.array([[-1,1]]);
X       = np.random.uniform(bound[:,0],bound[:,1], (3,1));            # Random Sample Points  [nx1]
y       = np.exp(X) + np.sin(3.*X) + 0.7*X + np.random.randn(1);      # Random Sample Targets [nx1]
# Grid Creation
print 'Grid is being generated...'
num     = 50;
x       = np.linspace(bound[:,0],bound[:,1],num).reshape(-1,1); # Grid Points to Predict [nx1]
print 'Initializing GP Regressor...'
# Initialize the Gaussian Process Regressor
kernel  = sig_n(1.)**2 * SE(length_scale = [5]);
GPR     = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20);
# Imposing Sampling Points for Inference
print 'Imposing sampling points...'
GPR.fit(X,y);
# Prediction Based on Observation
print 'Prediction...'
mean,std = GPR.predict(x, return_std=True);
# Plot using Homemade Module
print 'Plotting the results...'
GP_2D_Surrogate(X,y,x,mean,std)
plt.show()
