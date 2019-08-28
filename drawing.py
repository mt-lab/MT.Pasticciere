import ezdxf as ez
# from globalValues import heightMap
from elements import *

class Drawing:
    def __init__(self, dxf, offset=(0,0), rotation=0):
        self.dxf = dxf
        self.modelspace = self.dxf.modelspace()
        self.elements = []
        self.readDxf(self.modelspace)
        self.offset = offset
        self.rotation = rotation
        self.path = ()
        self.organizePath()

    def readDxf(self, root):
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                self.readDxf(block)
            elif elementRedef(element):
                self.elements.append(elementRedef(element))

    def slice(self, step=1.0):
        for element in self.elements:
            element.slice(step)

    def setOffset(self, offset=(0,0)):
        self.offset = offset
        self.offset()

    def offset(self):
        for element in self.path:
            element.setOffset(self.offset)

    def setRotation(self, rotation=0):
        self.rotation = rotation
        self.rotate()

    def rotate(self):
        pass

    def addZ(self):
        pass

    # def adjustPath(self, offset=(0, 0), pathToPly=PCD_PATH):
    #     pcd, pcd_xy, pcd_z = readPointCloud(pathToPly)
    #     # add volume to dxf, also add offset
    #     for element in self.path:
    #         element.setOffset(offset)
    #         element.addZ(pcd_xy, pcd_z, pcd)

    def organizePath(self, start_point=(0, 0)):
        """
        Сортирует и ориентирует элементы друг за другом относительно данной точки

        :param elements: элементы, которые необходимо сориентировать и сортировать
        :param start_point: точка, относительно которой выбирается первый элемент
        :return path: отсортированный и ориентированный массив элементов
        """
        path = []
        elements = self.elements.copy()
        # сортировать элементы по их удалению от точки
        elements.sort(key=lambda x: x.bestDistance(start_point))
        while len(elements) != 0:
            # первый элемент в списке (ближайший к заданной точке) - текущий
            current = elements[0]
            # добавить его в сориентированный массив
            path.append(current)
            # убрать этот элемент из неотсортированного списка
            elements.pop(0)
            # отсортировать элементы по их удалению от последней точки предыдущего элемента
            elements.sort(key=lambda x: x.bestDistance(current.getPoints()[-1]))
        self.path = path

def elementRedef(element):
    """
    Функция для переопределения полученного элемента в соответствующий подкласс класса Element

    :param element: элемент из dxf
    :return: переопределение этого элемента
    """
    if element.dxftype() == 'POLYLINE':
        return Polyline(element)
    elif element.dxftype() == 'SPLINE':
        return Spline(element)
    elif element.dxftype() == 'LINE':
        return Line(element)
    elif element.dxftype() == 'CIRCLE':
        return Circle(element)
    elif element.dxftype() == 'ARC':
        return Arc(element)
    elif element.dxftype() == 'ELLIPSE':
        pass
    elif element.dxftype() == 'LWPOLYLINE':
        pass
    elif element.dxftype() == 'POINT':
        pass
    else:
        print('Unknown element')
        return None









