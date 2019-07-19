class Cookie:
    def __init__(self, center=(0, 0), width=0, length=0, centerHeight=0, rotation=0, maxHeight=0):
        self.center = center  # мм
        self.width = width  # мм
        self.length = length  # мм
        self.centerHeight = centerHeight  # мм
        self.maxHeight = maxHeight  # мм
        self.rotation = rotation  # угол
        self.drawing = None  # dxf или ещё один класс под рисунок или набор точек рисунка (sliced) соответствующий ей
