import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dataset

def plot_dataset(X,  y, exp_config, fig_dir):
    plt.plot(X, y, 'rx', label="true")
    plt.title(f"Data scenario {exp_config.dataset.scenario}")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()

def plot_front(X_train,Y_train,X_test,Y_test,exp_config, fig_dir):
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = np.arange(0,1,0.02)
    x2 = np.arange(0,1,0.02)
    x1,x2 = np.meshgrid(x1,x2)
    z = dataset.function(x1,x2)
    plt.title("This is graph of total data")
    ax.plot_surface(x1,x2,z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')


    plt.plot(X_train[:,0],X_train[:,1], Y_train, 'rx', label="train")
    plt.plot(X_test[:,0],X_test[:,1], Y_test, 'bx', label="test")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()