# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
from scipy.stats import norm
# Import Scikit_Learn Modules
from sklearn.gaussian_process           import GaussianProcessRegressor
from sklearn.gaussian_process.kernels   import RBF as SE
from sklearn.gaussian_process.kernels   import Matern as M
from sklearn.gaussian_process.kernels   import ConstantKernel as sig_n
################################################################################
# EXPECTED IMPROVEMENT(EI) ACQUISATION FUNCTION
################################################################################
# x : values you want the Expected Imp. to be calculated [mxd]
# X : observations                                       [nxd]
# y : observed targets                                   [nx1]
# xi: trade-off term for exploration and exploitation, default = 0 -> balanced
################################################################################
def Expected_Improvement(x, X, GPR, xi=0.1):
    # Posteriror mean and std. for GPR
    mean, std = GPR.predict(x,return_std=True);
    std = std.reshape(-1,1)
    # Best value of the target so far...
    best_samp = np.max(GPR.predict(X))
    # Z calculation for normal distribution functions
    imp = mean - best_samp - xi;
    Z   = imp / std
    # Expected Improvement calculation
    EI  =  imp * norm.cdf(Z) + std * norm.pdf(Z);
    EI[std==0.] = 0.
    return EI;
################################################################################
# PROBABILITY OF IMPROVEMENT(PI) ACQUISATION FUNCTION
################################################################################
# x : values you want the Expected Imp. to be calculated [mxd]
# X : observations                                       [nxd]
# y : observed targets                                   [nx1]
# xi: trade-off term for exploration and exploitation, default = 0 -> balanced
################################################################################
def Probability_Improvement(x, X, GPR, xi=0.1):
    # Posteriror mean and std. for GPR
    mean, std = GPR.predict(x,return_std=True);
    std = std.reshape(-1,1)
    # Best value of the target so far...
    best_samp = np.max(GPR.predict(X))
    # Z calculation for normal distribution functions
    imp = mean - best_samp - xi;
    Z   = imp / std
    # Probability of Improvement calculation
    PI  = norm.cdf(Z)
    return PI;
################################################################################
# KNOWLEGE GRADIENT (KG) ACQUISATION FUNCTION
################################################################################
# x : values you want the Expected Imp. to be calculated [mxd]
# X : observations                                       [nxd]
# xi: trade-off term for exploration and exploitation, default = 0 -> balanced
################################################################################
def Knowledge_Gradient(x,X,y,GPR,kernel):
    mean,std = GPR.predict(x,return_std=True)
    mean_max = np.max(mean)
    #kernel  = sig_n(1.)**2 * SE(length_scale=1.)
    eval     = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 5);
    KG       = np.zeros(mean.size)
    for iter in range(mean.size):
        X_try = np.vstack((X,x[iter]))
        y_try = np.vstack((y,mean[iter]))
        eval.fit(X_try,y_try)
        m_try, s_try = eval.predict(x,return_std=True)
        KG[iter] =  np.max(m_try) - mean_max
    return KG
