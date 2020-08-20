import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib import cm
import dataset,sample


def plot_dataset(X,  y, exp_config, fig_dir):
    plt.plot(X, y, 'rx', label="true")
    plt.title(f"Data scenario {exp_config.dataset.scenario}")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()


def plot_front(X_train, Y_train, X_test, Y_test, exp_config, fig_dir):
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = np.arange(0, 1, 0.02)
    x2 = np.arange(0, 1, 0.02)
    x1,x2 = np.meshgrid(x1, x2)
    z = dataset.function(x1, x2)
    plt.title("This is graph of total data")
    ax.plot_surface(x1, x2, z, rstride=1, cstride=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')

    plt.plot(X_train[:, 0], X_train[:, 1], Y_train, 'bo', label="train samples")
    plt.plot(X_test[:, 0], X_test[:, 1], Y_test, 'ro', label="test samples")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()


def plot_genx(X_train, gen_x, del_x, length, n_sample):
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
    x_min = np.amin(X_train, axis=0)
    x1_min = x_min[0]
    x2_min = x_min[1]

    ax.set_xlim([x1_min-length[0] / 2, (x1_min-length[0])+n_sample[0]*length[0]])
    x1 = np.arange(x1_min - length[0] / 2, ((x1_min-length[0])+n_sample[0]*length[0]), x1_space, "float")
    ax.set_xticks(x1)
    ax.set_ylim([x2_min-length[1] / 2, (x2_min-length[1])+n_sample[1]*length[1]])
    x2 = np.arange(x2_min - length[1] / 2, ((x2_min-length[1])+n_sample[1]*length[1]), x2_space, "float")
    ax.set_yticks(x2)

    # 设置网格样式
    ax.grid(True, linestyle='-', color="grey")

    plt.title('显示中文标题')
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", label="train")
    plt.scatter(gen_x[:, 0], gen_x[:, 1], marker="x", label="gen")
    plt.scatter(del_x[:, 0], del_x[:, 1], marker='v', label="delete")

    plt.show()

    return