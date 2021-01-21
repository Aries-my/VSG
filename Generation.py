import sample
import numpy as np
import copy


def gen_x_sample(sample_list, X_train, gen_x, max_dist, dim, discard_list, x_com, x_max, x_min, E_dist):
    q = -1
    for sample1 in sample_list:
        q += 1
        print(q)
        xvlist = sample1.xlist
        list = copy.deepcopy(xvlist)
        point_list = sample.gen_product(list, x_com, E_dist, max_dist)
        sample.cross_point_del(point_list, X_train)
        if len(point_list) != 0:
            p_dist_list = []
            i = 0
            select_list = []
            n = sample1.ori_num
            c(point_list, dim, x_com, p_dist_list)
            if n == 0:
                index_ = min_index(p_dist_list, 1)
                for i in index_:
                    sample1.gen_xlist.append(point_list[i])
                    select_list.append(point_list[i])
            else:
                num = np.ceil(2 * n)
                if len(point_list) <= num:
                    sample1.gen_xlist = point_list
                    for xo in point_list:
                        select_list.append(xo)
                else:
                    index_list = min_index(p_dist_list, num)
                    for i in index_list:
                        sample1.gen_xlist.append(point_list[i])
                        select_list.append(point_list[i])
            for xo in select_list:
                gen_x.append(xo)
        else:
            p = []
            for i in range(len(xvlist)):
                d = x_max[i] - x_min[i]
                x_index = 0
                for j in range(len(xvlist[i])):
                    if abs(xvlist[i][j] - (x_max[i] - x_min[i]) / 2) < d:
                        x_index = j
                        d = xvlist[i][j] - (x_max[i] - x_min[i])
                p.append(xvlist[i][x_index])

            gen_x.append(p)
            discard_list.append(p)
    return

def gen_x_sample2(sample_list, gen_x, dim, x_max, x_min, n_sample, X_train):
    q = -1
    x_center = center(x_max, x_min, dim)
    for si in range(len(sample_list)):
        q += 1
        #print(q)
        sample1 = sample_list[si]
        xvlist = sample1.xlist
        n = sample1.ori_num
        list = copy.deepcopy(xvlist)
        point_list = []
        '''for index in range(dim):
            xl = xvlist[index]
            if n_sample[index] != 1:
                p_list.append(xl)
        point_list1 = sample.gen_product2(p_list)
        get_point_list(point_list1, point_list, n_sample, list, dim)
        center_dist(x_center, point_list, p_dist_list)'''
        p = get_closed_x(sample1.center, X_train)
        n = copy.deepcopy(n_sample)
        idim = np.argmax(n)
        p[idim] = list[idim][0]
        while in_list(p, gen_x) or in_list(p, X_train):
            n[idim] = 1
            idim = np.argmax(n)
            if n[idim] == 1:
                break
            p = get_closed_x(sample1.center, X_train)
            xi = get_closed_x_dim(sample1.center, idim, list[idim])
            p[idim] = xi
        point_list.append(p)
        gen_x.append(p)
        '''if n == 0:
            p = get_closed_x(x_center, X_train)
            idim = np.argmax(n_sample)
            p[idim] = list[idim][0]
            point_list.append(p)
        else:
            num = np.ceil(2 * n)
            index_list = min_index(p_dist_list, num)
            for i in index_list:
                sample1.gen_xlist.append(point_list[i])
                gen_x.append(point_list[i])'''
    return


def in_list(list1, l):
    list = copy.deepcopy(l)
    for i in range(len(list)):
        l = list[i]
        m = 0
        for j in range(len(list1)):
            if list1[j] == l[j]:
                m += 1
            else:
                break
        if m == len(list1):
            return 1
    return 0



def get_closed_x(center, X_train):
    d = 10000000000
    j = 0
    x = copy.deepcopy(X_train)
    for i in range(len(x)):
        d1 = sample.x_dist(center, x[i])
        if d1 < d:
            d = d1
            j = i
    return x[j]



def get_closed_x_dim(center, idim, list):
    d = 10000000000
    j = 0
    for i in range(len(list)):
        d1 = list[i] - center[idim]
        if d1 < d:
            j = i
            d = d1
    return list[j]



def center(x_max, x_min, dim):
    x = []
    for i in range(dim):
        xi = (x_max[i] + x_min[i]) / 2
        x.append(xi)
    return x


def center_dist(x_center, point_list, p_dist_list):
    for point in point_list:
        dist = sample.x_dist(point, x_center)
        p_dist_list.append(dist)


def get_point_list(point_list1, point_list, n_sample, xvlist, dim):
    for i in range(len(point_list1)):
        p = []
        j = 0
        for idim in range(dim):
            if n_sample[idim] == 1:
                p.append(inter_x(xvlist[idim]))
            else:
                p.append(point_list1[i][j])
                j += 1
        point_list.append(p)


def inter_x(list):
    L = copy.deepcopy(list)
    L = sorted(L)
    inter_x = L[len(L) // 2]
    return inter_x


def min_index(list, num):
    i = num
    index_list = []
    list1 = copy.deepcopy(list)
    list1 = np.array(list1)
    while i > 0:
        index = np.argmin(list1)
        index_list.append(index)
        list1[index] = np.max(list1) + 1
        i -= 1
    return index_list


def E_dist(x_com):
    sum_x = np.sum(np.square(x_com), 1)
    dist = np.add(np.add(-2 * np.dot(x_com, x_com.T), sum_x).T, sum_x)
    dist = np.array(dist)
    dist = np.around(dist, decimals=3)
    return dist


def get_x_com(X_train, dim, x_value, x_value_ori):
    x_min = np.amin(X_train, axis=0)
    x_max = np.amax(X_train, axis=0)
    x_com = copy.deepcopy(X_train)
    x_add = []
    for index in range(dim):
        for x in x_value[index]:
            if x not in x_value_ori[index]:
                point = []
                for i in range(dim):
                    if i == index:
                        point.append(x)
                    else:
                        point.append((x_min[index] + x_max[index]) / 2)
                x_add.append(point)
                point = np.array(point)
                x_com = np.vstack((x_com, point))

    return x_com, x_add


def c(point_list, dim, x_com, p_dist_list):
    for point in point_list:
        xlist = []
        for idim in range(dim):
            x = point[idim]
            for X in x_com:
                if X[idim] == x:
                    xlist.append(X)
                    break
        d = 0
        for xi in xlist:
            for xj in xlist:
                d1 = sample.x_dist(xi, xj)
                if d1 > d:
                    d = d1
        p_dist_list.append(d)
    return


def gen_x_in_sample(x_list, i_list, sample1, dim, list):
    center = sample1.center
    for idim in range(dim):
        point_list = x_list[idim]
        index = 0
        d = 10000000
        for i in range(len(point_list)):
            p = point_list[i]
            d1 = sample.x_dist(center, p)
            if d1 < d:
                index = i
                d = d1
        x_list[idim] = [point_list[index]]
        i_list[idim] = [i_list[idim][index]]
        list[idim] = [list[idim][index]]
    return


def get_xlist(xvlist, x_com, dim):
    x_list = []
    i_list = []
    for idim in range(dim):
        x_l = []
        i_l = []
        for i in range(len(xvlist[idim])):
            for j in range(len(x_com)):
                if xvlist[idim][i] == x_com[j][idim]:
                    x_l.append(x_com[j])
                    i_l.append(j)
                    break
        x_list.append(x_l)
        i_list.append(i_l)
    return x_list, i_list


def check_dist(x_list, i_list, max_dist, E_dist, dim):
    idim = 0
    while idim < dim - 1:
        x1 = idim
        x2 = idim + 1
        while  x2 < dim:
            i1 = i_list[x1][0]
            i2 = i_list[x2][0]
            d = E_dist[i1][i2]
            if d > max_dist / 7:
                f = 1
                return 0
            else:
                x2 += 1
        idim += 1
    return 1

