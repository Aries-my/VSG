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


def plot_genx(X_train, gen_x, length, n_sample, exp_config, fig_dir, xlimit, string):
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
    fig = plt.figure(2, (10, 10))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """设置刻度"""
    x_min = np.amin(X_train, axis=0)

    ax.set_xlim(0, 1)
    ax.set_xticks(xlimit[0])
    ax.set_ylim(0, 1)
    ax.set_yticks(xlimit[1])

    # 设置网格样式
    ax.grid(True, linestyle='-', color="grey")

    plt.title('显示中文标题')
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", label="train")
    plt.scatter(gen_x[:, 0], gen_x[:, 1], marker="o", label="gen")

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/"+string)

    plt.show()

    return


def plot_qr(del_point, exp_config, fig_dir, string):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(2, (10, 10))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """设置刻度"""

    plt.title('显示中文标题')
    plt.xlabel("no.")
    plt.ylabel("y")

    vir_y = []
    true_y = []
    no = []

    for index in range(len(del_point)):
        point = del_point[index]
        vir_y.append(point.y)
        true_y.append(point.true)
        no.append(index)

    plt.plot(no, vir_y, marker="o", label="vir_y")
    plt.plot(no, true_y, marker="^", label="true_y")

    plt.legend(loc='upper left')

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/" + string)

    plt.show()

    return


def plot_erro(vir_point, exp_config, fig_dir, string):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(2, (10, 10))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """设置刻度"""

    plt.title('显示中文标题')
    plt.xlabel("no.")
    plt.ylabel("erro")

    erro_y = []
    no = []

    for index in range(len(vir_point)):
        point = vir_point[index]
        erro_y.append(point.y - point.true)
        no.append(index)

    plt.plot(no, erro_y, marker="o", label="erro")

    plt.legend(loc='upper left')

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/" + string)

    plt.show()

    return