"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""
# TODO Переписать ВСЁ используя библиотеки для работы с геометрией (pyeuclid)
#   или написать класс vector3d с необходимыми операциями

import ezdxf as ez
import numpy as np
from utilities import X, Y, Z, pairwise, diap, findPointInCloud, distance
from numpy import sqrt, cos, sin, pi
from configValues import accuracy


class Element:
    """
    Общий класс с функциями общими для всех элементов, многие оверрайдятся в конкретных случаях
    """

    def __init__(self, entity, first=(0, 0), last=(0, 0)):
        """
        Конструктор объекта

        :param entity: элемент из dxf
        """
        self.entity = entity
        self.points = []
        self.sliced = []
        self.backwards = False
        self.offset = (0, 0)
        self.length = 0
        self.flatLength = 0

    def firstPoint(self):
        if len(self.sliced) != 0:
            return self.sliced[0] if not self.backwards else self.sliced[-1]
        elif len(self.points) != 0:
            return self.points[0] if not self.backwards else self.points[-1]
        else:
            return None

    def lastPoint(self):
        if len(self.sliced) != 0:
            return self.sliced[-1] if not self.backwards else self.sliced[0]
        elif len(self.points) != 0:
            return self.points[-1] if not self.backwards else self.points[0]
        else:
            return None

    def setOffset(self, offset=(0, 0)):
        """
        Задать смещение для рисунка (добавить к нарезанным координатам смещение)

        :param offset: величина смещение
        :return: None
        """
        if len(self.sliced) != 0:
            for point in self.sliced:
                point[X] -= self.offset[X]
                point[X] += offset[X]
                point[Y] -= self.offset[Y]
                point[Y] += offset[Y]
            self.offset = offset
        else:
            print('nothing to offset')

    def bestDistance(self, point):
        """
        Вычисляет с какой стороны точка находится ближе к элементу и ориентирует его соответственно

        :param point: точка от которой считается расстояние
        :return: минимальное расстояние до одного из концов объекта
        """
        dist2first = sqrt(abs(self.firstPoint()[X] - point[X]) ** 2 + abs(self.firstPoint()[Y] - point[Y]) ** 2)
        dist2last = sqrt(abs(self.lastPoint()[X] - point[X]) ** 2 + abs(self.lastPoint()[Y] - point[Y]) ** 2)
        self.backwards = dist2last < dist2first
        return min(dist2first, dist2last)

    def getPoints(self):
        """
        Возвращает точки как из dxf
        """
        return self.points if not self.backwards else self.points[::-1]

    def getSlicedPoints(self):
        """
        Возвращает нарезанные координаты
        """
        return self.sliced if not self.backwards else self.sliced[::-1]

    def calculateLength(self):
        """
        Рассчитать длину элемента
        """
        if len(self.sliced) != 0:
            for p1, p2 in pairwise(self.sliced):
                self.length += distance(p1, p2)
        elif len(self.points) != 0:
            for p1, p2 in pairwise(self.points):
                self.flatLength += distance(p1, p2)

    def slice(self, step=1):
        """
        Нарезать элемент на более менее линии с заданным шагом
        :param step: шаг нарезки
        :return:
        """
        for start, end in pairwise(self.points):
            for p in diap(start, end, step):
                p = list(p)
                if len(p) != 3:
                    p.append(0)
                elif len(p) > 3:
                    p = p[:3]
                self.sliced.append(p)

    def addZ(self, pcd_xy, pcd_z, pcd=None):
        """
        Добавить координату Z к элементу
        :param pcd_xy: часть облака точек с X и Y координатами
        :param pcd_z: часть облака точек с Z координатами
        :return: None
        """
        # TODO: вычисление высоты точки по 4 соседям (т.к. облако точек это равномерная сетка) используя веса
        #       весами сделать расстояние до соседей и проверить скорость вычислений
        if len(self.sliced) != 0:
            for p in self.sliced:
                p[Z] = findPointInCloud(p, pcd_xy, pcd_z, pcd)
        elif len(self.points) != 0:
            for p in self.points:
                p.append(findPointInCloud(p, pcd_xy, pcd_z, pcd))


class Point(Element):
    # TODO: написать обработку точек
    pass


class Polyline(Element):
    """
    Подкласс для элемента Полилиния из dxf
    """

    def __init__(self, polyline):
        super().__init__(polyline)
        self.points = [point for point in polyline.points()]
        self.first = self.points[0]
        self.last = self.points[-1]
        self.sliced = []
        self.length = 0


class LWPolyline(Polyline):
    # TODO: написать обработку LW полилиний
    pass


class Spline(Element):
    """
    Подкласс для объека Сплайн
    """

    # TODO: написать обработку сплайнов для нарезки
    #   прочитать книгу о NURBS, доработать алгоритм Антона
    def __init__(self, spline):
        super().__init__(spline)
        self.points = [point for point in spline.control_points]
        self.first = spline.control_points[0]
        self.last = spline.control_points[-1]
        self.sliced = []


class Line(Element):
    """
    Подкласс для объекта Линия
    """

    def __init__(self, line):
        super().__init__(line)
        self.points = [line.dxf.start, line.dxf.end]
        self.first = self.points[0]
        self.last = self.points[-1]
        self.sliced = []
        self.length = 0


class Circle(Element):
    """
    Подкласс для объекта Окружность
    """

    def __init__(self, circle):
        super().__init__(circle)
        self.center = circle.dxf.center
        self.radius = circle.dxf.radius
        self.startAngle = 0
        self.endAngle = 2 * pi
        self.first = (
            self.center[X] + self.radius * cos(self.startAngle), self.center[Y] + self.radius * sin(self.startAngle))
        self.last = (
            self.center[X] + self.radius * cos(self.endAngle), self.center[Y] + self.radius * sin(self.endAngle))
        self.points = [self.first, self.last]
        self.sliced = []
        self.length = 0

    def slice(self, step=1):
        angle_step = step / self.radius * (self.endAngle - self.startAngle) / abs(
            self.endAngle - self.startAngle)  # в радианах с учетом знака
        for angle in np.arange(self.startAngle, self.endAngle, angle_step):
            p = [self.radius * cos(angle) + self.center[X], self.radius * sin(angle) + self.center[Y], 0]
            self.sliced.append(p)
        last = list(self.last)
        if len(last) < 3:
            last.append(0)
        else:
            last = last[:2]
        self.sliced.append(last)  # дбавление конечной точки в массив нарезанных точек


class Arc(Circle):
    """
    Подклас для объекта Дуга
    """

    def __init__(self, arc):
        super().__init__(arc)
        self.startAngle = arc.dxf.start_angle * pi / 180  # в радианах
        self.endAngle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.startAngle > self.endAngle:
            self.endAngle += 2 * pi
        self.first = (
            self.center[X] + self.radius * cos(self.startAngle), self.center[Y] + self.radius * sin(self.startAngle))
        self.last = (
            self.center[X] + self.radius * cos(self.endAngle), self.center[Y] + self.radius * sin(self.endAngle))
        self.points = [self.first, self.last]
        self.sliced = []
        self.length = 0


class Ellipse(Element):
    # TODO: написать обработку эллипсов
    pass


class Contour:
    def __init__(self, elements=None):
        """
        :param list of Element elements: элементы составляющие контур
        """
        if elements is None:
            self.elements = []
            self.n_elements = 0
            self.first = (0, 0)
            self.last = (0, 0)
            self.flatLength = 0
            self.length = 0
            self.closed = None
        else:
            self.elements = elements
            self.n_elements = len(elements)
            self.first = elements[0].firstPoint()
            self.last = elements[-1].lastPoint()
            self.flatLength = 0
            self.length = 0
            self.calculateLength()
            if distance(self.first, self.last) < accuracy:
                self.closed = True
            else:
                self.closed = False

    def addElement(self, element):
        """
        Добавить элемент в конец контура
        :param Element element: элементр контура
        """
        self.elements.append(element)
        self.calculateLength()
        self.last = element.lastPoint()

    def calculateLength(self):
        if self.n_elements == 1:
            self.elements[0].calculateLength()
            self.flatLength = self.elements[0].flatLength
            self.length = self.elements[0].length
        elif self.n_elements > 1:
            for element in self.elements:
                element.calculateLength()
                self.flatLength = element.flatLength
                self.length += element.length
        else:
            self.flatLength = 0
            self.length = 0

    def getPoints(self):
        points = []
        for element in self.elements:
            points += element.getPoints()
        return points

    def getSlicedPoints(self):
        points = []
        for element in self.elements:
            points += element.getSlicedPoints()
        return points

    def firstPoint(self):
        return self.elements[0].firstPoint()

    def lastPoint(self):
        return self.elements[0].lastPoint()


class Drawing:
    # TODO: шаблон dxf по которому рисунок делится на слои:
    #   0 - общий контур печенья, по которому найти центр и поворот рисунка
    #   1 - самый внешний/важный контур
    #   ...
    #   last - элементы для печати в конце
    def __init__(self, dxf=None, offset=(0, 0), rotation=0):
        """
        :param dxf: открытый библиотекой рисунок
        :param offset: смещение центра рисунка
        :param rotation: угол поворота рисунка (его ориентация)
        """
        if dxf is None:
            self.dxf = None
            self.modelspace = None
            self.elements = []
            self.contours = []
            self.length = 0
            self.flatLength = 0
            self.center = (0, 0)
            self.offs = offset
            self.rotation = rotation
            self.path = ()
        else:
            self.dxf = dxf
            self.modelspace = self.dxf.modelspace()
            self.elements = []
            self.contours = []
            self.length = 0
            self.flatLength = 0
            self.center = (0, 0)
            self.offs = offset
            self.rotation = rotation
            self.path = ()
            self.readDxf(self.modelspace)
            self.organizePath()
            self.findContours()
            self.findCenter()
            self.findRotation()
            self.calculateLength()

    def __str__(self):
        return f'Геометрический центр рисунка: X: {self.center[X]:4.2f} Y: {self.center[Y]:4.2f} мм\n' + \
               f'Ориентация рисунка: {self.rotation * 180 / pi: 4.2f} градуса\n' + \
               f'Общая плоская длина рисунка: {self.flatLength: 4.2f} мм'

    def readDxf(self, root):
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                self.readDxf(block)
            elif elementRedef(element):
                self.elements.append(elementRedef(element))
        print('dxf прочтён.')

    def slice(self, step=1.0):
        for element in self.elements:
            element.slice(step)
        print(f'Объекты нарезаны с шагом {step:2.1f} мм')

    def findCenter(self):
        # TODO: расчёт центра рисунка
        """
        Расчитывает геометрический центр рисунка
        :return:
        """
        pass

    def setOffset(self, offset=(0, 0)):
        self.offs = offset
        self.offset()

    def offset(self):
        for element in self.path:
            element.setOffset(self.offs)

    def findRotation(self):
        # TODO: расчёт ориентации рисунка
        """
        Расчитывает поворот рисунка
        :return:
        """
        pass

    def setRotation(self, rotation=0):
        self.rotation = rotation
        self.rotate()

    def rotate(self):
        # TODO: функция поворота рисунка
        pass

    def addZ(self, pcd_xy, pcd_z):
        for element in self.elements:
            element.addZ(pcd_xy, pcd_z)
        self.calculateLength()

    # def adjustPath(self, offset=(0, 0), pathToPly=PCD_PATH):
    #     pcd, pcd_xy, pcd_z = readPointCloud(pathToPly)
    #     # add volume to dxf, also add offset
    #     for element in self.path:
    #         element.setOffset(offset)
    #         element.addZ(pcd_xy, pcd_z, pcd)

    def calculateLength(self):
        for contour in self.contours:
            contour.calculateLength()
            self.length += contour.length
            self.flatLength += contour.flatLength

    def organizePath(self, start_point=(0, 0)):
        """
        Сортирует и ориентирует элементы друг за другом относительно данной точки
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
        print('Сформирована очередность элементов.')

    def findContours(self):
        elements = []
        for e1, e2 in pairwise(self.path):
            elements.append(e1)
            d = distance(e1.last, e2.first)
            if d > accuracy:
                contour = Contour(elements)
                self.contours.append(contour)
                elements = [e2]
            else:
                elements.append(e2)
        contour = Contour(elements)
        self.contours.append(contour)
        print('Найдены контуры.')


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
