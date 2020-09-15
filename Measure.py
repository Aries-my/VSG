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