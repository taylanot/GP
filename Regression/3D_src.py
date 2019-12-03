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
# START-Creating Sample Data
# NOTE: REMOVE THIS FOR REAL APPLICATION
################################################################################
print 'Seeding...'
np.random.seed(2);                                       # Plant Random Seed
print 'Sampling points are being generated...'
# Random Sample Generation
bound   = np.array([[-4,4]]);
X       = np.random.uniform(bound[:,0],bound[:,1], (10,2));  # Random Sample Points  [nx2]
y       = np.sin(X[:,0]+X[:,1]) + np.exp(X[:,0]-X[:,1]); # Random Sample Targets [nx2]
################################################################################
# START-Creating Sample Data
# NOTE: REMOVE THIS FOR REAL APPLICATION
################################################################################
# Grid Creation
print 'Grid is being generated...'
num     = 50;                                            # Grid Density
x       = np.linspace(bound[:,0],bound[:,1],num);            # Change your Bound for the Application
x1, x2  = np.meshgrid(x,x);                              # [n,n], [n,n]
print 'Initializing GP Regressor...'
# Initialize the Gaussian Process Regressor
kernel  = sig_n(1.)**2 * SE(length_scale = [5,5]);
GPR     = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20);
# Imposing Sampling Points for Inference
print 'Imposing sampling points...'
GPR.fit(X,y);
# Prepare Inputs for Prediction
################################################################################
# NOTE;  in order to prevent getting into the regressor I stacked all
#        all the possible point combinations in x1x2 as a row and passed
#        passed it once. Then we change the shape of the mean and std.dev
#        taken from regressor to match the x1,x2 size.
################################################################################
x1x2 = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)));
# Prediction Based on Observation
print 'Prediction...'
mean,std = GPR.predict(x1x2, return_std=True);
# Output Manipulation (mentioned in the NOTE!)
mean = mean.reshape(x1.shape); std = std.reshape(x1.shape);
# Plot using Homemade Module
print 'Plotting the results...'
GP_3D_Surrogate(X,y,x1,x2,mean,std)
plt.show()
