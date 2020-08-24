
def qr_selection(con_list, models, vir_xpoint, vir_ypoint, y_quantile, gen_x_point, ols):
    l = len(gen_x_point)
    while l > 0:
        dim, lim = chose_xlim(con_list)
        #print("start from ")
        print("x"+str(dim)+" No."+str(lim)+" gap")
        checked_xlist = []
        checked_ylist = []
        checked_list(con_list[dim][lim], checked_xlist, checked_ylist)
        for sample in con_list[dim][lim].slist:
            if sample.checked == 0:
                qr(sample, vir_xpoint, vir_ypoint, models, checked_ylist, checked_xlist, dim, y_quantile, ols)
                l -= 1
        con_list[dim][lim].checked = 1
    return


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
            if r1 > r and xlim.checked == 0:
                r = r1
                d = dim
                l = index
    return d, l