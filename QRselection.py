import numpy as np
import XLim, Point
import copy


def qr_selection(xlim_list, models, vir_xpoint, vir_ypoint, y_quantile, ols, x_value_ori,
                 n_sample, X_train, Y_train, sample_list, point_list, x_value):
    n = copy.deepcopy(n_sample)
    idim = np.argmin(n)
    ni = np.amin(n)
    while ni !=10000:
        for x in x_value_ori[idim]:
            index = find_orix(idim, x, X_train)
            y = Y_train[index]
            p_list = []
            ori_point = Point.Point(X_train[index], y)
            ori_point.checked = 1
            p_list.append(ori_point)
            wait_list = []
            l_sort(point_list, X_train[index], idim, wait_list, p_list)
            while len(wait_list) > 0:
                i = confir_point(p_list, idim, wait_list)
                y_ch = wait_list[i].y
                x_ch = wait_list[i].x[idim]
                i_com, y_com = closed_y(y_ch, p_list)
                x_com = p_list[i_com].x[idim]
                if y_ch > y_com:
                    q_index = quan(y_ch, y_quantile)
                else:
                    q_index = quan(y_com, y_quantile)
                if q_index < len(y_quantile):
                    b = models[q_index].param[idim]
                else:
                    b = ols.params[idim + 1]
                if b > 0 and (y_com - y_ch) * (x_com - x_ch) > 0 or b < 0 and (y_com - y_ch) * (
                        x_com - x_ch) < 0:
                    wait_list[i].checked = 1
                    confir_sxl(wait_list[i],  sample_list, xlim_list, 1, idim)
                    vir_xpoint.append(wait_list[i].x)
                    vir_ypoint.append(wait_list[i].y)
                    p_list.append(wait_list[i])
                    del wait_list[i]
                else:
                    wait_list[i].checked = 1
                    confir_sxl(wait_list[i], sample_list, xlim_list, 0, idim)
                    del wait_list[i]
        n[idim] = 10000
        idim = np.argmax(n)
        ni = np.amax(n)

    n = copy.deepcopy(n_sample)
    idim = np.argmin(n)
    ni = np.amin(n)
    while ni != 10000:
        for x in x_value[idim]:
            if x in x_value_ori[idim]:
                break
            p_list = []
            wait_list = []
            l_sort(point_list, x, idim, wait_list, p_list)
            while len(wait_list) > 0:
                i = confir_point(p_list, idim, wait_list)
                y_ch = wait_list[i].y
                x_ch = wait_list[i].x[idim]
                i_com, y_com = closed_y(y_ch, p_list)
                x_com = p_list[i_com].x[idim]
                if y_ch > y_com:
                    q_index = quan(y_ch, y_quantile)
                else:
                    q_index = quan(y_com, y_quantile)
                if q_index < len(y_quantile):
                    b = models[q_index].param[idim]
                else:
                    b = ols.params[idim + 1]
                if b > 0 and (y_com - y_ch) * (x_com - x_ch) > 0 or b < 0 and (y_com - y_ch) * (
                        x_com - x_ch) < 0:
                    wait_list[i].checked = 1
                    confir_sxl(wait_list[i], sample_list, xlim_list, 1, idim)
                    vir_xpoint.append(wait_list[i].x)
                    vir_ypoint.append(wait_list[i].y)
                    p_list.append(wait_list[i])
                    del wait_list[i]
                else:
                    wait_list[i].checked = 1
                    confir_sxl(wait_list[i], sample_list, xlim_list, 0, idim)
                    del wait_list[i]
        n[idim] = 10000
        idim = np.argmin(n)
        ni = np.amin(n)
    return


def confir_point(p_list, dim, wait_list):
    i = -1
    d = 1000
    for index in range(len(wait_list)):
        point = wait_list[index]
        d1 = dis(point, p_list, dim)
        if d1 < d:
            d = d1
            i = index
    return i


def dis(point, p_list, dim):
    d = 1000
    for index in range(len(p_list)):
        t = abs(point.x[dim] - p_list[index].x[dim])
        if t < d:
            d = t
    return d


def l_sort(point_list, x_train, dim, wait_list, p_list):
    for point in point_list:
        r = 0
        for i in range(len(point.x)):
            if i != dim and point.x[i] != x_train[i]:
                break
            elif i != dim and point.x[i] == x_train[i]:
                r += 1
            elif i == dim and point.x[i] == x_train[i]:
                break
            else:
                r += 1
        if r == len(point.x) and point.checked == 0:
            wait_list.append(point)
        elif r == len(point.x) and point.checked == 1:
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


def confir_sxl(point, sample_list, xlim_list, flag, dim):
    for index in range(len(xlim_list[dim])):
        xl = xlim_list[dim][index]
        if xl.xl < point.x[dim] < xl.xu:
            if flag == 0:
                xl.del_num += 1
                xl.checked_num += 1
                xl.uncheck_num -= 1
            else:
                xl.checked_num += 1
                xl.uncheck_num -= 1
        break

    for index in range(len(sample_list)):
        r = 0
        for i in range(len(point.x)):
            if sample_list[index].lim[i][0] <= point.x[i] < sample_list[index].lim[i][1]:
                r += 1
        if r == len(point.x):
            if flag == 0:
                sample_list[index].del_num += 1
                sample_list[index].checked_num += 1
            else:
                sample_list[index].checked_num += 1


    return


def find_orix(dim, xv, X_train):
    for index in range(len(X_train)):
        if xv == X_train[index][dim]:
            return index
    return -1


def quan(y_pre, y_quantile):
    for i in range(len(y_quantile)):
        if y_pre < y_quantile[i]:
            return i
    return len(y_quantile)


def closed_y(y, p_list):
    index = 0
    y1 = 10000
    for i in range(len(p_list)):
        if abs(p_list[i].y - y) < abs(y1 - y) :
            y1 = p_list[i].y
            index = i
    return index, y1