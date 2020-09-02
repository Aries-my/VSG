import numpy as np
import XLim
import copy


def qr_selection(xlim_list, models, vir_xpoint, vir_ypoint, y_quantile, gen_x_point, ols, x_value_ori, x_value,
                 n_sample, X_train, Y_train, sample_list, point_list):
    n = copy.deepcopy(n_sample)

    idim = np.argmax(n)
    ni = np.amax(n)
    for x in x_value_ori[idim]:
        index = find_orix(idim, x, X_train)
        si, xi = confir_sample(X_train, index, sample_list, xlim_list, idim)
        p_list = []
        l_sort(point_list, X_train, idim, p_list)
        l = len(p_list)
        while l > 0:
            i = confir_point(p_list, x, idim)



def confir_point(p_list, x, dim):
    iid = 0
    l = -1
    u = -1
    m = 0
    for index in range(len(p_list)):
        if p_list[index][dim] > x:
            iid = index
            m = index
            break
    i = iid
    while i < len(p_list):
        if p_list[i].checked == 0:
            u = i
        else:
            i += 1
    i = iid - 1
    while i > -1:
        if p_list[i].checked == 0:
            l = i
        else:
            i -= 1
    if l == -1:
        if u == -1:
            return -1
        else:
            return u
    else:
        if u == -1:
            return l
        else:
            if abs(l - m) < abs(u - m):
                return l
            else:
                return u


def l_sort(point_list, X_train, dim, p_list):
    for point in point_list:
        r = -1
        for i in range(len(X_train)):
            if i != dim and point.x[i] != X_train[i]:
                    r = 0
                    break
        if r == -1:
            if len(p_list) == 0:
                p_list.append(point)
            else:
                f = -1
                for j in range(len(p_list)):
                    if point.x[dim] < p_list[j].x[dim]:
                        p_list.insert(j, point)
                        f = 0
                        break
                if f == 0:
                    p_list.append(point)
    return


def confir_sample(X_train, index, sample_list, xlim_list, dim):
    x = X_train[index]
    si = 0
    xi = 0
    for si in range(len(sample_list)):
        if XLim.in_sample(x, sample_list[si]):
            break
    for xi in range(len(xlim_list[dim])):
        if XLim.in_limit(x, xlim_list[dim][xi]):
            break
    return si, xi


def find_orix(dim, xv, X_train):
    for index in range(len(X_train)):
        if xv == X_train[index][dim]:
            return index
    return -1

def qr(sample, vir_xpoint, vir_ypoint, models, checked_ylist, checked_xlist, dim, y_quantile, ols):
    closed_y_index, closed_y_n = closed_y(sample.y_pre, checked_ylist)
    closed_x_n = checked_xlist[closed_y_index][dim]
    y_pre = sample.y_pre
    x = sample.center[dim]
    q_index = quan(y_pre, y_quantile)
    if q_index < len(y_quantile):
        b = models[q_index].param[dim]
    else:
        b = ols.param[dim+1]
    if b > 0 and (closed_y_n - y_pre) * (closed_x_n - x) > 0 or b < 0 and (closed_y_n - y_pre) * (
                closed_x_n - x) < 0:
        sample.checked = 1
        sample.checked_num = 1
        vir_xpoint.append(sample.center)
        vir_ypoint.append(sample.y_pre)
        checked_xlist.append(sample.center)
        checked_ylist.append(sample.y_pre)
        print("Accepted point:")
        print(str(sample.center) + "\t" + str(sample.y_pre))
    else:
        sample.checked = 1
        print("Bad point:")
        print(str(sample.center) + "\t" + str(sample.y_pre))
    return


def quan(y_pre, y_quantile):
    for i in range(len(y_quantile)):
        if y_pre < y_quantile[i]:
            return i
    return len(y_quantile)


def closed_y(pre_y, checked_ylist):
    index = 0
    y = max(checked_ylist)
    for i in range(len(checked_ylist)):
        if abs(checked_ylist[i] - pre_y) < abs(y - pre_y) :
            y = checked_ylist[i]
            index = i
    return index, y


def checked_list(xlim, checked_xlist, checked_ylist):
    for index in range(len(xlim.slist)):
        sample = xlim.slist[index]
        if sample.ori_num > 0:
            for x in sample.ori_xlist:
                checked_xlist.append(x)
            for y in sample.ori_ylist:
                checked_ylist.append(y)
        if sample.checked_num > 0:
            for i in range(len(sample.checked_list)):
                if sample.checked_list[i] == 1:
                    checked_xlist.append(sample.gen_xlist[i])
                    checked_ylist.append(sample.gen_ylist[i])
    return


def chose_xlim(xlim_list):
    d = 0
    l = 0
    r = 0
    for dim in range(len(xlim_list)):
        for index in range(len(xlim_list[dim])):
            xlim = xlim_list[dim][index]
            r1 = (xlim.checked_num + xlim.ori_num) / (xlim.checked_num + xlim.uncheck_num + xlim.ori_num)
            if r1 > r and xlim.checked == 0:
                r = r1
                d = dim
                l = index
    return d, l