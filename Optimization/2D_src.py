# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
import sys;sys.dont_write_bytecode = True;
# Import Scikit_Learn Modules
from sklearn.gaussian_process           import GaussianProcessRegressor
from sklearn.gaussian_process.kernels   import RBF as SE
from sklearn.gaussian_process.kernels   import Matern as M
from sklearn.gaussian_process.kernels   import ConstantKernel as sig_n
# Import Homemade Modules
from acq_funcs      import *
from proposal       import *
from GP_Plot_Opt    import *
################################################################################
################################################################################
# NOTE: if you are having problems with convergence try turning of your seeding,
# to get away from local optimum.
################################################################################
#print 'Seeding...'
# np.random.seed(2);                                      # Plant Random Seed
print 'Sampling points are being generated...'
# Random Sample Generation
bound   = np.array([[-1,2]]);
################################################################################
# NOTE: by changing below X to some constant X you get something same for all
# the runs now it will be arbitrary starting point.
################################################################################
X       = np.random.uniform(bound[:,0],bound[:,1],(2,1))
y       = -np.sin(3*X) - X**2 + 0.7*X ;         # Random Sample Targets [nx1]
#plt.plot(X,y,'o')
# Grid Creation
print 'Grid is being generated...'
num     = 100;
x       = np.linspace(bound[:,0],bound[:,1],num); # Grid Points to Predict [nx1]
y_plt   = -np.sin(3*x) - x**2 + 0.7*x
#plt.plot(x,y_plt)
# Initialize the Gaussian Process Regressor
################################################################################
# NOTE: Chose your kernel wisely it has some serious effects...
################################################################################
print 'Initializing the GP Regressor...'
kernel  = sig_n(1.)**2 * SE(length_scale = 1.)
GPR     = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 30);
# Information regarding optimization: increase to increase accuracy of surragate
################################################################################
# NOTE: As it founds optimum it stays there which causes some convergence issues
# you can avoid it by playing with xi values to let the model explore more or
# keep your data numbers low...
################################################################################
max_it  =  6
for iter in range (1,max_it):
    print 'Imposing sampling points for iteration...',iter
    GPR.fit(X,y);
    print 'Proposing new point for iteration...     ',iter
    # Proposal of new point by minimizing acquisation function
    X_next  = prop (Expected_Improvement,X,GPR,bound);
    y_next  = -np.sin(3*X_next) - np.power(X_next,2) + 0.7*X_next
    # Plotting...
    GP_2D_Opt(X,y,x,GPR,Expected_Improvement(x,X,GPR),X_next,y_next)
    X       = np.vstack((X,X_next));
    y       = np.vstack((y,y_next));
plt.show()
