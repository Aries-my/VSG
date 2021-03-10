import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib import cm
import dataset,sample
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
    ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('f(x0,x1)')

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
    #ax.grid(True, linestyle='-', color="black")

    plt.title("")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", label="Original samples", color="red")
    plt.scatter(gen_x[:, 0], gen_x[:, 1], marker="s", label="Generated samples", color="green")

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

    plt.title(string)
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

    plt.plot(no, vir_y, marker="s", label="Virtual y", color="green")
    plt.plot(no, true_y, marker="o", label="True y", color="red")

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

    plt.title(string)
    plt.xlabel("no.")
    plt.ylabel("erro")

    erro_y = []
    no = []

    for index in range(len(vir_point)):
        point = vir_point[index]
        erro_y.append(point.y - point.true)
        no.append(index)

    plt.plot(no, erro_y, marker="o", label="erro", color="blue")

    plt.legend(loc='upper left')

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/" + string)

    plt.show()

    return


def plot_erro2(test_list, exp_config, fig_dir, string):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(2, (10, 10))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """设置刻度"""

    plt.title(string)
    plt.xlabel("no.")
    plt.ylabel("erro")

    erro_y_old = []
    erro_y_new = []
    no = []

    for index in range(len(test_list)):
        erro_y_old.append(test_list[index].erro_old)
        erro_y_new.append(test_list[index].erro_new)
        no.append(index)

    plt.plot(no, erro_y_old, marker="o", label="erro_before")
    plt.plot(no, erro_y_new, marker="s", label="erro_before")

    plt.legend(loc='upper left')

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/" + string)

    plt.show()

    return



def plot_sample(X_train, exp_config, fig_dir, xlimit, string):
    '''
    synthetic data
    the graph of original x points and generated x points in two-dimensional grid
    :param X_train:original x points
    :param gen_x:enerated x points
    :length:the length of sample
    :return: none
    '''
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
    #ax.grid(True, linestyle='-', color="black")

    plt.title("")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", label="Original samples", color="red")

    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/"+string)

    plt.show()

    return

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def plots(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, fig_dir="", save_fig=False):
    plt.plot(d_loss_err, label="Discriminator Loss")
    plt.plot(d_loss_true, label="Discriminator Loss - True")
    plt.plot(d_loss_fake, label="Discriminator Loss - Fake")
    plt.plot(g_loss_err, label="Generator Loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Loss")
    if save_fig:
        plt.savefig(f"{fig_dir}/gan_loss.png")
    plt.show()


def plot_scatter_density(x, y, title=""):
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_den_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_den_y = fig.add_subplot(gs[1, 1], sharey=ax)

    def scatter_kde():
        ax.set_xlabel(r"$x_1$"), ax.set_ylabel(r"$x_2$")
        ax_den_x.tick_params(axis="x", labelbottom=False)
        ax_den_y.tick_params(axis="y", labelleft=False)

        ax.scatter(x, y, s=10, c="#D02090", marker="o")  #mc=m
        sns.distplot(x, vertical=False, ax=ax_den_x)
        sns.distplot(y, vertical=True, ax=ax_den_y)

        ax.set_xlim([0, 1.25]), ax.set_ylim([0, 1.25])
        ax.set_aspect("equal", "box")

    scatter_kde()
    fig.suptitle(title)
    plt.show()

def _plot_surface(ax, function):
    x_range = np.arange(0, 1, 0.01)
    y_range = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(x_range, y_range)
    Z = function(X, Y)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.Blues(norm(Z))
    surf = ax.plot_surface(X, Y, Z,
                           rstride=5, cstride=5, facecolors=colors, shade=False, zorder=1)
    surf.set_facecolor((0, 0, 0, 0))
    return ax


def _magical_sinus(x, y):
    """
    Create a noise-free single-valued benchmarking function:
                 z = f(x, y)
    derived from sinus function. It feeds two variables and
    returns a single value for each given pair of inputs(x, y).
    """
    z =1.335*(1.6*(1-x))+np.exp(2*x-1)*np.sin(4*np.pi*(x-0.6)**2)+np.exp(3*(y-0.5))*np.sin(3*np.pi*(y-0.9)**2)

    return z


def plot_3d(X_train, X_test, y_train, y_test, X_del, Y_del, exp_config, fig_dir, save_fig=False):
    fig = plt.figure(figsize=(6, 5))
    ax1 = Axes3D(fig)
    # ax1.set_title(f"Data scenario {exp_config.dataset.scenario}")

    ax1 = _plot_surface(ax1, dataset.function)
    # 画数据点
    ax1.scatter(X_train[:,0], X_train[:,1], y_train, c='r', label="Training Samples", s=10, marker='o')
    ax1.scatter(X_test[:,0], X_test[:,1], y_test, color='yellow', label="Virtual Samples", s=12, marker='^')
    ax1.scatter(X_del[:, 0], X_del[:, 1], Y_del, color='green', label="Deleted Samples", s=12, marker='s')

    ax1.set_zlim(0, 1.05 * max(y_train.max(), y_train.max()))
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter(r'%.02f'))
    ax1.set_xlabel(r"$x_1$"), ax1.set_ylabel(r"$x_2$"), ax1.set_zlabel(r"$f(x_1,x_2)$")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()

