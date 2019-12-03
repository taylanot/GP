# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import matplotlib.pyplot as plt
def GP_2D_Opt(X, y, x, GPR, y_acq, X_next,y_next):
    mean, std   = GPR.predict(x,return_std=True);std.reshape(-1,1);
    fig, ax     = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    # Create the first axis -> Surragete Function
    ax[0].fill_between(x.ravel(), mean.ravel() + 2*std, mean.ravel() - 2*std,
                    alpha = 0.1,color='m');
    ax[0].plot(x,mean,label='Surrogate Function',color = 'm')
    ax[0].plot(X,y,'kx',mew=3, label="Observation");
    ax[0].set_ylabel('Surragete Function')
    ax[0].axvline(x=X_next,ls=':');
    ax[0].legend()
    # Point Proposition
    ax[0].plot(X_next,y_next,'x',mew=3,color='maroon')
    # Create the second -> Acquisation Function
    ax[1].set_ylabel('Acqusation Function');
    ax[1].plot(x,y_acq,label='Acqusation Function',color='midnightblue');
    ax[1].axvline(x=X_next,ls=':');
    ax[1].legend()
