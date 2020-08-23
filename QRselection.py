import numpy as np


def qr_selection(con_list, sample_list, vir_sample, models):
    dim, lim = chose_xlim(con_list)
    print("start from ")
    print("x"+str(dim)+" No."+str(lim)+"gap")
    for sample in con_list[dim][lim].slist:
        a = 0


def qr(vir_point, models, ):

    return


def checked_list(xlim, checked_list):

    return


def chose_xlim(con_list):
    d = 0
    l = 0
    r = 0
    for dim in range(len(con_list)):
        for index in range(len(con_list[dim])):
            xlim = con_list[dim][index]
            r1 = (xlim.checked_num + xlim.ori_num) / (xlim.ori_num + xlim.checked_num + xlim.uncheck_num)
            if r1 > r:
                r = r1
                d = dim
                l = index
    return d, l