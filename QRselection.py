import numpy as np


def qr_selection(con_list, models, vir_xpoint, vir_ypoint, y_quantile):
    dim, lim = chose_xlim(con_list)
    print("start from ")
    print("x"+str(dim)+" No."+str(lim)+"gap")
    checked_xlist = []
    checked_ylist = []
    checked_list(con_list[dim][lim], checked_xlist, checked_ylist)
    for sample in con_list[dim][lim].slist:
        if sample.checked == 0:
            qr(sample, vir_xpoint, vir_ypoint, models, checked_ylist, checked_xlist, dim, y_quantile)



def qr(sample, vir_xpoint, vir_ypoint, models, checked_ylist, checked_xlist, dim, y_quantile):
    larger_y_index, larger_y_n = larger_y(sample.y_pre, checked_ylist)
    smaller_y_index, smaller_y_n = smaller_y(sample.y_pre, checked_ylist)
    larger_x_n = checked_xlist[larger_y_index][dim]
    smaller_x_n = checked_xlist[smaller_y_index][dim]
    y_pre = sample.y_pre
    q_index = quan(y_pre, y_quantile)
    if q_index < len(y_quantile):
        b = models[q_index][2 + dim]
        if b > 0 and larger_x_n > smaller_x_n or b < 0 and larger_x_n < smaller_x_n:
            sample.checked = 1
            sample.checked_num = 1
            vir_xpoint.append(sample.center)
            vir_ypoint.append(sample.y_pre)
            checked_xlist.append(sample.center)
            checked_ylist.append(sample.y_pre)
    else:
        sample.checked = 1
    return


def quan(y_pre, y_quantile):
    for i in range(len(y_quantile)):
        if y_pre < y_quantile[i]:
            return i
    return len(y_quantile)


def larger_y(pre_y, checked_ylist):
    index = 0
    y = pre_y
    for i in range(len(checked_ylist)):
        if checked_ylist[i] > pre_y and checked_ylist[i] < y:
            y = checked_ylist[i]
            index = i
    return index, y


def smaller_y(pre_y, checked_ylist):
    index = 0
    y = pre_y
    for i in range(len(checked_ylist)):
        if checked_ylist[i] < pre_y and checked_ylist[i] > y:
            y = checked_ylist[i]
            index = i
    return index, y


def checked_list(xlim, checked_xlist, checked_ylist):
    for index in range(len(xlim.slist)):
        sample = xlim.slist[index]
        if sample.ori_num > 0:
            for x,y in sample.ori_xlist, sample.ori_ylist:
                checked_xlist.append(x)
                checked_ylist.append(y)
        if sample.checked_num == 1:
            checked_xlist.append(sample.center)
            checked_ylist.append(sample.y_pre)
    return


def chose_xlim(con_list):
    d = 0
    l = 0
    r = 0
    for dim in range(len(con_list)):
        for index in range(len(con_list[dim])):
            xlim = con_list[dim][index]
            r1 = (xlim.checked_num + xlim.ori_num) / (xlim.checked_num + xlim.uncheck_num + xlim.ori_num)
            if r1 > r:
                r = r1
                d = dim
                l = index
    return d, l