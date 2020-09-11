import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import math
import copy


def get_sta_reg_cov(X_train, Y_train):
    X_normal = preprocessing.scale(X_train)
    Y_normal = preprocessing.scale(Y_train)
    model_1 = LinearRegression()
    model_1.fit(X_train, Y_train)
    print("Unstandardized regression coefficient: ")
    print(np.around(model_1.coef_, decimals=5))
    print("normal coefficient: ")
    print(np.around(model_1.intercept_, decimals=5))
    model_2 = LinearRegression()
    model_2.fit(X_normal, Y_normal)
    print("Standardized regression coefficient: ")
    model_2_coef = np.around(model_2.coef_, decimals=5)
    print(model_2_coef)
    print("normal coefficient: ")
    print(np.around(model_2.intercept_, decimals=5))
    return model_2_coef


def get_importance(coe, Y_train):

    # 得到每列的标准差,是一维数组
    y_std = np.std(Y_train)
    print("y std:")
    print(np.around(y_std, decimals=5))
    imp = []
    for value in coe:
        i = abs(value / y_std)
        imp.append(i)

    imp = np.array(imp)
    imp = np.around(imp, decimals=5)
    print("The importance for every dimension:")
    print(imp)
    return imp


def get_min(nplist):
    '''
    get minist value in nplist where all values are not nagetive
    :param nplist:
    :return:
    '''
    min = nplist[1]
    for value in nplist:
        if value != 0:
            if value < min:
                min = value
    return min


def fit_length(L, length):
    num = 1
    n_sample = []
    for index in range(len(L)):
        n = math.ceil(L[index] / length[index])
        num *= n
        n_sample.append(n)
        print("The number of diversions of the " + str(index) + "th dimension is: " + str(n))
        print(n_sample[index])

    print("总的样方分割数为：")
    print(num)
    return num, n_sample


def get_sample_length(X_train, imp):
    length = []
    #get Euclidean distance in x domain
    sum_x = np.sum(np.square(X_train), 1)
    dist = np.add(np.add(-2 * np.dot(X_train, X_train.T), sum_x).T, sum_x)
    dist = np.array(dist)
    dist = np.around(dist, decimals=3)
    print("Euclidean distance in x domain：")
    #print(dist)

    np.set_printoptions(suppress=True)
    dist = dist.flatten()
    m_dist = get_min(dist)
    m_imp = max(imp)
    max_dist = max(dist)

    print("minist dist:")
    print(m_dist)
    print("maxist imp:")
    print(m_imp)

    for index in range(len(imp)):
        l = m_dist * m_imp / imp[index]
        length.append(l)

    length = np.array(length)
    length = np.around(length, decimals=5)
    print("The original length of the smaple: ")
    print(length)
    return length, max_dist


def get_x_len(x_min, x_max):
    L = []

    for index in range(len(x_min)):
        minx = x_min[index]
        maxx = x_max[index]
        print("The value area of x in the sample is between "
              + str(minx) + " and " + str(maxx) + "in the dimension of No. " + str(index))
        l = maxx - minx
        L.append(l)

    L = np.array(L)
    print("The full length of every dimension:")
    print(L)
    return L


def divide_sample(length, L):
    num, n_sample = fit_length(L, length)
    while num > 130:
        for index in range(len(length)):
            length[index] = length[index] * 1.2
        num, n_sample = fit_length(L, length)

    print("分割数：")
    n_sample = np.array(n_sample)
    print(n_sample)
    print("样方的大小：")
    print(length)
    return n_sample, length


def gen_x_center(dim, length, n_sample, x_min):
    X = []
    for index in range(dim):
        mi = x_min[index]
        print("第"+str(index)+"维度，最小的x为"+str(mi))
        i = 0
        x = []
        while i < n_sample[index]:
            a = mi + (i + 1 / 2) * length[index]
            a = np.around(a, decimals=5)
            x.append(a)
            i += 1
        X.append(x)
        print("第"+str(index)+"维度的中心值有：")
        print(x)
    return X


def gen_two_product(list1, list2):
    res_list = []
    for index1 in range(len(list1)):
        for index2 in range(len(list2)):
            l = []
            l = copy.deepcopy(list1[index1])
            l.append(list2[index2])
            res_list.append(l)
    return res_list


def gen_product(list_of_list):
    list = copy.deepcopy(list_of_list)
    list1 = list[0]
    for index in range(len(list1)):
        i = list1[index]
        list1[index] = []
        list1[index].append(i)
    for tmp_list in list_of_list[1:]:
        list2 = tmp_list
        two_res_list = gen_two_product(list1, list2)
        list1 = two_res_list
    return list1


def cross_point_del(gen_x_cross, X_train):
    for point in gen_x_cross:
        for x in X_train:
            if same_point(point, x) == 1:
                gen_x_cross.remove(point)
    return


def same_point(point1, point2):
    for i in range(len(point1)):
        if point1[i] != point2[i]:
            return 0
    return 1


def sample_point_num (X_train, length, point):
    for x in X_train:
        r = 0
        for index in range(len(length)):
            low = point[index] - length[index] / 2
            up = point[index] + length[index] / 2
            if low < x[index] < up:
                r += 1
        if r == len(length):
            return bool(1)
    return bool(0)


def gen_true_x(X_train, point_list, length):
    i = 0
    while i < len(point_list):
        r = sample_point_num(X_train, length, point_list[i])
        if r:
            point_list = np.delete(point_list, i, axis=0)
        else:
            i += 1
    return point_list

def del_x(gen_sample_point, gen_x_point):
    '''
    find the deleted x points
    :param gen_sample_point:
    :param gen_x_point:
    :return:
    '''
    del_x_point = []
    for p in range(len(gen_sample_point)):
        if_exist = 0
        for q in range(len(gen_x_point)):
            if compare_num(gen_sample_point[p],gen_x_point[q]) == 1:
                if_exist = 1
                break
        if if_exist == 0:
            del_x_point.append(gen_sample_point[p])
    del_x_point = np.array(del_x_point)
    return del_x_point

def compare_num(list1, list2):
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return 0
    return 1


def gen_is_list(X_train, length, is_fliter):
    '''
    genarate the 0-1 graph of the samples in X
    :param X_train:
    :param point_list:
    :param length:
    :param is_fliter:
    :return:
    '''
    x_min = np.amin(X_train, axis=0)
    x1_min = x_min[0]
    x2_min = x_min[1]
    for p in range(len(is_fliter[0])):
        for q in range(len(is_fliter[1])):
            x1_l = x1_min - length[0] / 2 + p * length[0]
            x1_u = x1_min - length[0] / 2 + (p + 1) * length[0]
            x2_l = x2_min - length[1] / 2 + q * length[1]
            x2_u = x2_min - length[1] / 2 + (q + 1) * length[1]
            for x in X_train:
                if x[0] > x1_l and x[0] < x1_u and x[1] > x2_l and x[1] < x2_u:
                    is_fliter[p][q] = 1
                    break
    return is_fliter


def point_filiter(gen_x_cross, X_train, max_dist, x_value, x_value_ori, dim):
    x_min = np.amin(X_train, axis=0)
    x_max = np.amax(X_train, axis=0)
    x_com = copy.deepcopy(X_train)
    for index in range(dim):
        for x in x_value[index]:
             if x not in x_value_ori[index]:
                point = []
                for i in range(dim):
                    if i == index:
                        point.append(x)
                    else:
                        point.append((x_min[index] + x_max[index]) / 2)
                point = np.array(point)
                x_com = np.vstack((x_com, point))
    i = 0
    #x_com = np.array(x_com)
    while i < len(gen_x_cross):
        point = gen_x_cross[i]
        x_list = []
        for index in range(dim):
            xi = point[index]
            for x in x_com:
                if xi == x[index]:
                    x_list.append(x)
                    break
        dist = 0
        for xi in x_list:
            for xj in x_list:
                if x_dist(xi, xj) > dist:
                    dist = x_dist(xi, xj)
        if dist > max_dist / 5:
            del gen_x_cross[i]
        else:
            i += 1
    return


def x_dist(list1, list2):
    dist = 0
    for index in range(len(list1)):
        dist += pow(list1[index] - list2[index], 2)
    dist = math.sqrt(dist)
    return dist