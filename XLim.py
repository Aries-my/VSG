import SampleCharacter
import numpy as np
import random
import copy


class XLim:
    def __init__(self, dim, xl, xu):
        self.dim = dim
        self.slist = []
        self.xl = xl
        self.xu = xu
        self.ori_num = 0
        self.gen_num = 0
        self.del_num = 0
        self.checked_num = 0
        self.uncheck_num = 0
        self.checked = 0
        self.xlist = []


def con_sample(xlim_list, length, x_min, dim, n_sample):
    for index in range(dim):
        x_list = []
        for i in range(n_sample[index]):
            xl = XLim(index, x_min[index] + i * length[index], x_min[index] + (i + 1) * length[index])
            x_list.append(xl)
        xlim_list.append(x_list)


def sample_feature(xlim_list, sample_list, xvalue):
    for index in range(len(xlim_list)):
        for i in range(len(xlim_list[index])):
            xl = xlim_list[index][i]
            dim = xl.dim
            for x in xvalue[dim]:
                if xl.xl < x < xl.xu:
                    xl.xlist.append(x)
            if len(xl.xlist) == 0:
                x_insert = random.uniform(xl.xl, xl.xu)
                xl.xlist.append(x_insert)
                xvalue[index].append(x_insert)
            add_sample(xl, sample_list)


def add_xvalue(xlim_list):
    for index in range(len(xlim_list)):
        for xl in xlim_list[index]:
            for sample in xl.slist:
                sample.xlist.append(xl.xlist)
    return


def r_blank(sample_list):
    blank = 0
    for sample in sample_list:
        if sample.gen_num == 0 and sample.ori_num == 0:
            blank += 1
    return blank


def sample_filling(xlim_list, sample_list, n_sample, dim):
    l = copy.deepcopy(n_sample)
    blank = r_blank(sample_list)
    while True:
        idim = np.argmax(l)
        for index in range(len(xlim_list[idim])):
            xl = xlim_list[idim][index]
            for sample_index in range(len(xl.slist)):
                sample = xl.slist[sample_index]
                if sample.gen_num == 0 and sample.ori_num == 0:
                    r = get_sample(xl, sample_index)
                    if r >= 0:
                        i = len(xl.slist[r].gen_xlist) + len(xl.slist[r].ori_xlist)
                        if i > 0:
                            t = random.randint(0, i - 1)
                        else:
                            t = 0
                        x_insert = random.uniform(sample.lim[idim][0], sample.lim[idim][1])
                        x_gen = xl.slist[r].gen_xlist[t]
                        x_gen[idim] = x_insert
                        sample.gen_num += 1
                        blank -= 1
        l[idim] = 0
        if blank == 0:
            break


def get_sample(xl, s_index):
    i = s_index + 1
    u = -1
    l = -1
    while i < len(xl.slist):
        if xl.slist[i].gen_num != 0 or xl.slist[i].ori_num != 0:
            u = i
            break
        i += 1
    i = s_index - 1
    while i > -1:
        if xl.slist[i].gen_num != 0 or xl.slist[i].ori_num != 0:
            l = i
            break
        i -= 1
    if u != -1 and l != -1:
        if abs(u - s_index) < abs(l - s_index):
            return u
        else:
            return l
    elif u == -1 and l != -1:
        return l
    elif u != -1 and l == -1:
        return u
    else:
        return -1


def get_x(xlim, sample):
    xl = sample.lim[xlim.dim][0]
    xu = sample.lim[xlim.dim][1]
    xvalue = []
    for x in xlim.xlist:
        if xl < x < xu:
            xvalue.append(x)
    return xvalue


def add_sample(xlim, sample_list):
    for sample in sample_list:
        if in_limit(sample.center, xlim):
            xlim.slist.append(sample)
    return


def xl_attri(xlim_list, X_train, gen_x_cross):
    for index in range(len(xlim_list)):
        for xl in xlim_list[index]:
            for point in X_train:
                if in_limit(point, xl):
                    xl.ori_num += 1
            for point in gen_x_cross:
                if in_limit(point, xl):
                    xl.gen_num += 1
            xl.uncheck_num = xl.gen_num
    return


def in_limit(point, xlim):
    dim = xlim.dim
    if xlim.xl <= point[dim] < xlim.xu:
        return 1
    return 0


def con_s(gen_sample_point, sample_list, dim, xlimit):
    for point in gen_sample_point:
        sc = SampleCharacter.SampleCharacter(dim, point)
        for i in range(dim):
            xl, xu = r_lu(xlimit, point[i], i)
            xlim = []
            xlim.append(xl)
            xlim.append(xu)
            sc.lim.append(xlim)
        sample_list.append(sc)
    return


def r_lu(xlimit, x, dim):
    for i in range(len(xlimit[dim])):
        if xlimit[dim][i] <= x < xlimit[dim][i+1]:
            return xlimit[dim][i], xlimit[dim][i+1]


def sample_attri(sample_list, X_train, gen_x_cross, Y_train):
    for sample in sample_list:
        for index in range(len(X_train)):
            point = X_train[index]
            if in_sample(point, sample):
                sample.ori_num += 1
                sample.ori_xlist.append(point)
                sample.ori_ylist.append(Y_train[index])
        for point in gen_x_cross:
            if in_sample(point, sample):
                sample.gen_num += 1
                sample.gen_xlist.append(point)
    return


def in_sample(point, sample):
    for index in range(len(point)):
        if sample.lim[index][0] <= point[index] < sample.lim[index][1]:
            r = 1
        else:
            return 0
    return 1


def add_y(sample_list, gpr):
    for sample in sample_list:
        for index in range(len(sample.gen_xlist)):
            point = sample.gen_xlist[index]
            point = [point]
            point = np.array(point)
            sample.gen_ylist.append(gpr.predict(point)[0])
            sample.checked_list.append(0)
    return