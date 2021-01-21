import numpy as np
from scipy.stats import multivariate_normal


def mae1(point_list):
    p = len(point_list)
    a = 0
    for point in point_list:
        a += abs(point.y - point.true)
    return a / p


def mse1(point_list):
    p = len(point_list)
    a = 0
    for point in point_list:
        a += pow(point.y - point.true, 2)
    return a / p


def mape1(point_list):
    p = len(point_list)
    a = 0
    for point in point_list:
        a += abs((point.y - point.true) / point.true)
    return a / p

def gaussian_NLPD(y_real, y_pred, cov, title=""):
    nll = -np.mean(
        [multivariate_normal.logpdf(x=y_real[i], mean=y_pred[i], cov=cov[i]) for i in range(len(y_pred))])
    print(f"{title} Gaussian NLPD: {nll}")
    return nll

def mae2(point_list):
    p = len(point_list)
    a1 = 0
    a2 = 0
    for point in point_list:
        a1 += abs(point.erro_old)
        a2 += abs(point.erro_new)
    return a1 / p, a2 / p


def mse2(point_list):
    p = len(point_list)
    a1 = 0
    a2 = 0
    for point in point_list:
        a1 += pow(point.erro_old, 2)
        a2 += pow(point.erro_new, 2)
    return a1 / p, a2 / p


def mape2(point_list):
    p = len(point_list)
    a1 = 0
    a2 = 0
    for point in point_list:
        a1 += abs((point.erro_old) / point.true)
        a2 += abs((point.erro_new) / point.true)
    return a1 / p, a2 / p