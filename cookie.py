class Cookie:
    def __init__(self, center=(0, 0), width=0, length=0, height=0, rotation=0):
        self.center = center  # мм
        self.width = width  # мм
        self.length = length  # мм
        self.height = height # мм
        self.rotation = rotation  # угол
        self.drawing = None  # dxf или ещё один класс под рисунок, предположительно не нужно
