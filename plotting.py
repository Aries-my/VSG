import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
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

def plot_genx(X_train, gen_x, def_x, length):
    '''
    synthetic data
    the graph of original x points and generated x points in two-dimensional grid
    :param X_train:original x points
    :param gen_x:enerated x points
    :length:the length of sample
    :return: none
    '''
    x1_space = length[0]
    x2_space = length[1]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(2, (10, 6))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """设置刻度"""
    ax.set_ylim(-3, 3)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim([-5, 8])
    ax.set_xticks([-5,5,1])

    plt.title('显示中文标题')
    plt.xlabel("x1")
    plt.ylabel("x2")
    x = np.arange(0, len(list1)) + 1
    x[0] = 1
    my_x_ticks = np.arange(1, 14, 1)
    plt.xticks(my_x_ticks)
    plt.plot(x, list1, label='list1', marker="o", markersize=10)  # marker设置标记形状 markersize设置标记大小
    plt.plot(x, list2, label='list2', marker="x", markersize=8)
    plt.legend()
    plt.grid()  # 添加网格
    plt.show()

    return