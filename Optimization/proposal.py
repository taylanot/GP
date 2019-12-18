# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import numpy as np
from scipy.optimize import minimize
# Import Homemade Modules
from acq_funcs import *
################################################################################
# New Point Proposal Algorithm:
# Minimize the selected acquisation function
# acq   : acquisation function
# X     : sample points
# GPR   : regressor
# NOTE: x is grid points we are looking for the function [mxd]
################################################################################
################################################################################
# NOTE: if you are observing other peaks in your the plots of acq functions
# increase your start points. You can search the domain by means of starting
# searching every point in the domain.
################################################################################
def prop(acq, X, GPR, bound, n_restarts = 300):
    dim     = X.shape[1];
    min_val = 1.;
    min_x   = None;
    ################################################################################
    # NOTE: Silly solution for my problem, but you should provide the boundaries
    # for all dimensions... you could implement a more elegent solutoin for the
    # boundary creation...
    ################################################################################
    if (dim == 1):
        bounds = bound;
    else:
        for i in range(1,dim):
            bounds = np.vstack((bound,bound));
    def obj(x):
        return -acq(x.reshape(-1,dim), X, GPR);
    for x0 in np.random.uniform(bound[:,0], bound[:,1], size=(n_restarts,dim)):
        res = minimize(obj, x0=x0, bounds=bounds, method='L-BFGS-B',options={'maxls':50})
        # res.fun   : y value of the minimized function
        # res.x     : X value of the minimized function as array [1xdim]
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x
    return min_x.reshape(-1,dim)
