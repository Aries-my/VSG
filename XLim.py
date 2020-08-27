import SampleCharacter


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


def con_sample(con_list, X_train, gen_x_point, sample_list, length, x_min, dim):
    for index in range(dim):
        x_list = []
        for i in range(dim):
            xl = XLim(index, x_min[index] + i * length[index], x_min[index] + (i + 1) * length[index])
            x_list.append(xl)
            xl_attri(xl, X_train, gen_x_point)
            add_sample(xl, sample_list)
        con_list.append(x_list)


def add_sample(xlim, sample_list):
    for sample in sample_list:
        if in_limit(sample.center, xlim):
            xlim.slist.append(sample)
    return


def xl_attri(xl, X_train, gen_x_point):
    for point in X_train:
        if in_limit(point, xl):
            xl.ori_num += 1
    for point in gen_x_point:
        if in_limit(point, xl):
            xl.gen_num += 1
    xl.uncheck_num = xl.gen_num
    return


def in_limit(point, xlim):
    dim = xlim.dim
    if xlim.xl < point[dim] < xlim.xu:
        return 1
    return 0


def con_s(gen_sample_point, length, sample_list, dim):
    for point in gen_sample_point:
        sc = SampleCharacter.SampleCharacter(dim, point)
        for i in range(dim):
            xl = point[i] - length[i] / 2
            xu = point[i] + length[i] / 2
            xlim = []
            xlim.append(xl)
            xlim.append(xu)
            sc.lim.append(xlim)
        sample_list.append(sc)
    return


def sample_attri(sample_list, X_train, gen_x_poin, del_x_points, Y_train):
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