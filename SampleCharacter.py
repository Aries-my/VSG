class SampleCharacter:
    def __init__(self, dim, cPoint, y_pre):
        self.dim = dim
        self.center = cPoint
        self.lim = []
        self.ori_num = 0
        self.gen_num = 0
        self.del_num = 0
        self.checked_num = 0
        self.checked = 0
        self.y_pre = y_pre
        self.ori_xlist = []
        self.ori_ylist = []


