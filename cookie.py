from math import pi


class Cookie:
    def __init__(self, center=(0, 0), width=0, length=0, centerHeight=0, rotation=0, maxHeight=0):
        self.center = center  # мм
        self.width = width  # мм
        self.length = length  # мм
        self.centerHeight = centerHeight  # мм
        self.maxHeight = maxHeight  # мм
        self.rotation = rotation  # угол
        self.drawing = None  # dxf или ещё один класс под рисунок или набор точек рисунка (sliced) соответствующий ей

    def __str__(self):
        return 'Центр:\n' + \
               f'    X: {self.center[0]: 4.2f} мм\n' + \
               f'    Y: {self.center[1]: 4.2f} мм\n' + \
               f'    Z: {self.centerHeight: 4.2f} мм\n' + \
               f'Поворот: {self.rotation * 180 / pi:4.2f} градусов'
