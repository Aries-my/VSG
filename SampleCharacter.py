class SampleCharacter:
    def __init__(self, length, cPoint, y_pre):
        self.dim = len(length)
        self.center = cPoint
        self.lim = []
        self.ori_num = 0
        self.gen_num = 0
        self.del_num = 0
        self.checked_num = 0
        self.uncheck_num = 0
        self.vir_point = []
        self.y_pre = y_pre


