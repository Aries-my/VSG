class SampleCharacter:
    def __init__(self, length, cPoint):
        self.dim = len(length)
        self.center = cPoint
        self.lim = []


def con_s(gen_sample_point, length, n_sample):
    for point in gen_sample_point:
        sc = SampleCharacter(length, point)
        for i in range(len(n_sample)):
            xl = point[i] - length[i] / 2
            xu = point[i] + length[i] / 2
            xlim = []
            xlim.append(xl)
            xlim.append(xu)
            sc.lim.append(xlim)
    return