import SampleCharacter
import numpy as np


class XLim:
    def __init__(self, num, xl, xu, xc):
        self.slist = []
        self.num = num
        self.xl = xl
        self.xu = xu
        self.xc = xc
        self.ori_num = 0
        self.gen_num = 0
        self.del_num = 0


def con_sample(con_list, X_train, gen_x_point, del_point, sample_list, n_sample, length):
    x_min = np.amin(X_train, axis=0)
    for index in range(len(n_sample)):
        x_list = []
        for i in range(n_sample[index]):
            xl = XLim(0, x_min[index]+i*length[index] - length[index] / 2,
                      x_min[index] + i * length[index] + length[index] / 2,
                      x_min[index]+i*length[index])
            x_list.append(xl)
        con_list.append(x_list)

def con_s(gen_sample_point, length, n_sample, sample_list):
    for point in gen_sample_point:
        sc = SampleCharacter.SampleCharacter(length, point)
        for i in range(len(n_sample)):
            xl = point[i] - length[i] / 2
            xu = point[i] + length[i] / 2
            xlim = []
            xlim.append(xl)
            xlim.append(xu)
            sc.lim.append(xlim)
        sample_list.append(sc)
    return