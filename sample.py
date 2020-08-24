import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import math
import copy


def get_sta_reg_cov(X_train, Y_train):
    X_normal = preprocessing.scale(X_train)
    Y_normal = preprocessing.scale(Y_train)
    model_1 = LinearRegression()
    model_1.fit(X_train,Y_train)
    print("Unstandardized regression coefficient: ")
    print(np.around(model_1.coef_, decimals=5))
    print("normal coefficient: ")
    print(np.around(model_1.intercept_, decimals=5))
    model_2 = LinearRegression()
    model_2.fit(X_normal,Y_normal)
    print("Standardized regression coefficient: ")
    model_2_coef = np.around(model_2.coef_, decimals=5)
    print(model_2_coef)
    print("normal coefficient: ")
    print(np.around(model_2.intercept_, decimals=5))
    return model_2_coef


def get_importance(coe,Y_train):

    # 得到每列的标准差,是一维数组
    y_std = np.std(Y_train)
    print("y std:")
    print(np.around(y_std, decimals=5))
    imp = []
    for value in coe:
        i = abs(value/y_std)
        imp.append(i)

    imp = np.array(imp)
    imp = np.around(imp, decimals=5)
    print("The importance for every dimension:")
    print(imp)
    return imp


def get_min(nplist):
    '''
    get minist value in nplist where values are all not nagetive
    :param nplist:
    :return:
    '''
    min = nplist[1]
    for value in nplist:
        if value != 0:
            if value <min:
                min = value
    return min


def get_ab(nplist):
    list = []
    for value in nplist:
        if value < 0:
            value = 0 - value
        list.append(value)
    list = np.array(list)
    return list


def fit_length(L,length):
    num = 1
    n_sample = []
    for index in range(len(L)):
        n = math.floor(L[index] / length[index])
        num = num * n
        n_sample.append(n)
        print("第" + str(index) + "维的分割数是：" + str(n))
        print(n_sample[index])

    print("总的样方分割数为：")
    print(num)
    return num, n_sample


def get_sample_length(X_train,imp):
    imp = get_ab(imp)
    length = []
    sum_x = np.sum(np.square(X_train), 1)
    dist = np.add(np.add(-2 * np.dot(X_train, X_train.T), sum_x).T, sum_x)
    dist = np.array(dist)
    dist = np.around(dist, decimals=3)
    dist = dist.flatten()
    np.set_printoptions(suppress=True)

    m_dist = get_min(dist)
    m_imp = min(imp)

    print("欧氏距离：")
    print(dist)
    print("min dist:")
    print(m_dist)
    print("min imp:")
    print(m_imp)

    for index in range(len(imp)):
        l = m_dist * imp[index] / m_imp
        length.append(l)

    length = np.array(length)
    length = np.around(length, decimals=5)
    print("The original length of the smaple: ")
    print(length)
    return length


def get_x_len(X_train):
    L = []
    X = X_train.T
    print("X_train.T:")
    print(X)

    for index in range(len(X)):
        mi = min(X[index])
        ma = max(X[index])
        print("第" + str(index) + "行最大值：" + str(ma) + "   最小值：" + str(mi))
        l = ma - mi
        L.append(l)

    L = np.array(L)
    print("The length of every diversion:")
    print(L)
    return L


def divide_sample(X_train, length):
    L = []
    X = X_train.T
    for index in range(len(X)):
        mi = min(X[index])
        ma = max(X[index])
        print("第" + str(index) + "行最大值：" + str(ma) + "   最小值：" + str(mi))
        l = ma - mi + length[index]
        L.append(l)

    L = np.array(L)
    num, n_sample = fit_length(L, length)
    while num > 200:
        for index in range(len(length)):
            L[index] = L[index] - length[index]
            length[index] = length[index] * 1.5
            L[index] = L[index] + length[index]
        num, n_sample= fit_length(L, length)

    print("分割数：")
    n_sample = np.array(n_sample)
    print(n_sample)
    print("样方的大小：")
    print(length)
    return n_sample, length


def gen_x_center(X_train, length, n_sample):
    X_T = X_train.T
    X = []
    for index in range(len(X_T)):
        mi = min(X_T[index])
        print("第"+str(index)+"维度，最小的x为"+str(mi))
        i = 0
        x = []
        print("index= "+str(index))
        while i < n_sample[index]:
            a = mi+i*length[index]
            x.append(a)
            i += 1
        X.append(x)
        print("第"+str(index)+"维度的x生成的值有：")
        print(x)
    print("生成的x值：")
    print(X)
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
    list1 = list_of_list[0]
    for index in range(len(list1)):
        i = list1[index]
        list1[index] = []
        list1[index].append(i)
    for tmp_list in list_of_list[1:]:
        list2 = tmp_list
        two_res_list = gen_two_product(list1, list2)
        list1 = two_res_list
    return list1


def sample_point_num (X_train, length, point):
    for x in X_train:
        r = 0
        for index in range(len(length)):
            low = point[index] - length[index] / 2
            up = point[index] + length[index] / 2
            if x[index] > low and x[index] < up:
                r += 1
        if r == len(length):
            return bool(1)
    return bool(0)


def gen_true_x(X_train, point_list, length):
    i = 0
    while i <len(point_list):
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

