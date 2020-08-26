import SampleCharacter
import numpy as np


class XLim:
    def __init__(self, dim, xl, xu, xc):
        self.dim = dim
        self.slist = []
        self.xl = xl
        self.xu = xu
        self.xc = xc
        self.ori_num = 0
        self.gen_num = 0
        self.del_num = 0
        self.checked_num = 0
        self.uncheck_num = 0
        self.checked = 0
        self.xlist = []


def con_sample(con_list, X_train, gen_x_point, del_x_points, sample_list, n_sample, length):
    x_min = np.amin(X_train, axis=0)
    for index in range(len(n_sample)):
        x_list = []
        for i in range(n_sample[index]):
            xl = XLim(index, x_min[index]+i*length[index] - length[index] / 2,
                      x_min[index] + i * length[index] + length[index] / 2,
                      x_min[index]+i*length[index])
            x_list.append(xl)
            xl_attri(xl, X_train, gen_x_point, del_x_points)
            add_sample(xl, sample_list)
        con_list.append(x_list)


def add_sample(xlim, sample_list):
    for sample in sample_list:
        if in_limit(sample.center, xlim):
            xlim.slist.append(sample)
    return


def xl_attri(xl, X_train, gen_x_point, del_x_points):
    for point in X_train:
        if in_limit(point, xl):
            xl.ori_num += 1
    for point in gen_x_point:
        if in_limit(point, xl):
            xl.gen_num += 1
    for point in del_x_points:
        if in_limit(point, xl):
            xl.del_num += 1
    xl.uncheck_num = xl.gen_num
    return


def in_limit(point, xlim):
    dim = xlim.dim
    if point[dim] > xlim.xl and point[dim] < xlim.xu:
        return 1
    return 0


def con_s(gen_sample_point, length, n_sample, sample_list, y_pre):
    for index in range(len(gen_sample_point)):
        point = gen_sample_point[index]
        sc = SampleCharacter.SampleCharacter(len(n_sample), point, y_pre[index])
        for i in range(len(n_sample)):
            xl = point[i] - length[i] / 2
            xu = point[i] + length[i] / 2
            xlim = []
            xlim.append(xl)
            xlim.append(xu)
            sc.lim.append(xlim)
        sample_list.append(sc)
    return


def sample_attri(sample_list, X_train, gen_x_point, del_x_points, Y_train):
    for sample in sample_list:
        for index in range(len(X_train)):
            point = X_train[index]
            if in_sample(point, sample):
                sample.ori_num += 1
                sample.ori_xlist.append(point)
                sample.ori_ylist.append(Y_train[index])
                sample.checked = 1
        for point in gen_x_point:
            if in_sample(point, sample):
                sample.gen_num += 1
        for point in del_x_points:
            if in_sample(point, sample):
                sample.del_num += 1
    return


def in_sample(point, sample):
    i = 0
    for index in range(len(point)):
        if point[index] > sample.lim[index][0] and point[index] < sample.lim[index][1]:
            i = 1
        else:
            return 0
    return 1