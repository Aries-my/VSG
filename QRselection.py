import numpy as np


def qr_selection(con_list, sample_list, vir_sample, models):
    d, l = chose_xlim(con_list)
    print("start from ")
    print("x"+str(d)+" No."+str(l)+"gap")
    for sample in con_list[d][l].slist:
        a = 0


def qr(vir_point, models, ):

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
                d = dim
                l = index
    return d, l