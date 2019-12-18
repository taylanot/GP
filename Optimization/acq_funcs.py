# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
from scipy.stats import norm
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
