class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.checked = 0
        self.true = 0
        self.erro = 0


def con_point(gen_x_cross, gen_y, point_list):
    for index in range(len(gen_x_cross)):
        point = Point(gen_x_cross[index], gen_y[index])
        point_list.append(point)
    return


class TestPoint:
    def __init__(self, x, true, pre_old, pre_new):
        self.x = x
        self.true = true
        self.pre_old = pre_old
        self.pre_new = pre_new
        self.erro_old = pre_old - true
        self.erro_new = pre_new - true

