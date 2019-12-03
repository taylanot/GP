# (C) Ozgur Taylan TURAN 2019, Nov. (Delft University of Technology)
# Import General Modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 3D Plotting Option for GP
def GP_3D_Surrogate(X,y,x1,x2,mean,std=None):
    fig = plt.figure(figsize=(8,5));
    ax  = fig.add_subplot(111,projection='3d');
    ax.scatter(X[:,0],X[:,1],y,color='r');
    if std is None:
        surf = ax.plot_surface(x1, x2, mean, cmap=plt.cm.CMRmap, linewidth=0, antialiased=True, alpha=0.5);
    else:
        surf = ax.plot_surface(x1, x2, mean-2*std, cmap='binary', linewidth=0, antialiased=True, alpha=0.3);
        surf = ax.plot_surface(x1, x2, mean+2*std, cmap='binary', linewidth=0, antialiased=True, alpha=0.3);
        surf = ax.plot_surface(x1, x2, mean, cmap=plt.cm.CMRmap, linewidth=0, antialiased=True, alpha=0.5);
    ax.set_xlabel('x1');ax.set_ylabel('x2');ax.set_zlabel('x3')
    fig.colorbar(surf)
# 2D Plotting Option for GP
def GP_2D_Surrogate(X, y, x, mean, std):
    plt.fill_between(x.ravel(),mean.ravel() + 2 * std,mean.ravel() - 2 * std,alpha=0.1,color='m');
    plt.plot(x,mean,label='Surrogate-Function',color='m')
    plt.plot(X,y,'o',color='orange',alpha=1,label='Sample Points')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('2D GP Regression')
    plt.legend()
