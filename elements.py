"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""
# TODO Переписать ВСЁ используя библиотеки для работы с геометрией (pyeuclid)
#   или написать класс vector3d с необходимыми операциями

from typing import List
import ezdxf as ez
import ezdxf.math as geom
from ezdxf.math.vector import Vector, NULLVEC
from ezdxf.math.bspline import BSpline
import numpy as np
# from tkinter import *
from utilities import X, Y, Z, pairwise, diap, findPointInCloud, distance
from numpy import sqrt, cos, sin, pi
from configValues import accuracy


class Element():
    """
    Общий класс с функциями общими для всех элементов, многие оверрайдятся в конкретных случаях
    """

    def __init__(self, entity, points: List['Vector'] = None):
        """
        Конструктор объекта

        :param entity: элемент из dxf
        """
        self.entity = entity
        self.points = points  # type: List[Vector]
        self.sliced = False
        self.withZ = False
        self.backwards = False

    @property
    def first(self):
        return self.points[0] if not self.backwards else self.points[-1]

    @property
    def last(self):
        return self.points[-1] if not self.backwards else self.points[0]

    @property
    def length(self):
        try:
            return self._length
        except AttributeError:
            length = 0
            for v1, v2 in pairwise(self.points):
                length += v1.distance(v2)
            self._length = length
            return length

    @property
    def flatLength(self):
        try:
            return self._length
        except AttributeError:
            flatLength = 0
            for v1, v2 in pairwise(self.points):
                flatLength += v1.vec2.distance(v2.vec2)
            self._flatLength = flatLength
            return flatLength

    def __str__(self):
        return f'Element: {self.entity.dxftype()}\n ' + \
               f'first point: {self.first}\n ' + \
               f'last point: {self.last}'

    def __repr__(self):
        return f'Element: {self.entity.dxftype()}\n ' + \
               f'first point: {self.first}\n ' + \
               f'last point: {self.last}'

    def translate(self, vector: 'Vector' = NULLVEC):
        """
        Задать смещение для рисунка (добавить к нарезанным координатам смещение)

        :param vector: величина смещение
        :return: None
        """
        self.points = [v + vector for v in self.points]

    def rotate(self, angle: float):
        self.points = [v.rotate(angle) for v in self.points]

    def bestDistance(self, point: 'Vector' = NULLVEC):
        """
        Вычисляет с какой стороны точка находится ближе к элементу и ориентирует его соответственно

        :param point: точка от которой считается расстояние
        :return: минимальное расстояние до одного из концов объекта
        """
        dist2first = self.points[0].distance(point)
        dist2last = self.points[-1].distance(point)
        self.backwards = dist2last < dist2first
        return min(dist2first, dist2last)

    def getPoints(self):
        """
        Возвращает точки
        """
        return self.points if not self.backwards else self.points[::-1]

    def getSlicedPoints(self):
        """
        Возвращает нарезанные координаты
        """
        if self.sliced:
            return self.points if not self.backwards else self.points[::-1]
        else:
            return None

    def slice(self, step=1):
        """
        Нарезать элемент на более менее линии с заданным шагом
        :param float step: шаг нарезки
        :return:
        """
        sliced = []
        for start, end in pairwise(self.points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            param_step = step / dist
            for i in range(n_steps + 1):
                v = start.lerp(end, param_step)
                sliced.append(v)
            sliced.append(end)
        self.points = sliced
        self.sliced = True
        try:
            del self._length
        except AttributeError:
            pass

    def addZ(self, pcd_xy=None, pcd_z=None, pcd=None, constantShift=None):
        """
        Добавить координату Z к элементу
        :param pcd_xy: часть облака точек с X и Y координатами
        :param pcd_z: часть облака точек с Z координатами
        :param float constantShift: для задания одной высоты всем точкам
        :return: None
        """
        # TODO: вычисление высоты точки по 4 соседям (т.к. облако точек это равномерная сетка) используя веса
        #       весами сделать расстояние до соседей и проверить скорость вычислений
        # TODO: переделать под новое облако точек
        if constantShift is not None:
            self.points = [v.replace(z=constantShift) for v in self.points]
            return None
        else:
            if pcd_z is None or pcd_xy is None:
                if pcd is None:
                    raise Exception('Point cloud is needed.')
                else:
                    pcd_xy, pcd_z = np.split(pcd, [Z], axis=1)
            self.points = [v.replace(z=findPointInCloud(v.xyz, pcd_xy, pcd_z)) for v in self.points]
            self.withZ = True
        try:
            del self._length
        except AttributeError:
            pass


class Point(Element):
    # TODO: написать обработку точек
    pass


class Polyline(Element):
    """
    Подкласс для элемента Полилиния из dxf
    """

    def __init__(self, polyline):
        points = [Vector(point) for point in polyline.points()]
        super().__init__(polyline, points)


class LWPolyline(Polyline):
    # TODO: написать обработку LW полилиний
    pass


class Spline(Element, BSpline):
    """
    Подкласс для объека Сплайн
    """

    # TODO: написать обработку сплайнов для нарезки
    #   прочитать книгу о NURBS, доработать алгоритм Антона
    def __init__(self, spline):
        control_points = [Vector(point) for point in spline.control_points]
        knots = [knot for knot in spline.knots]
        weights = [weight for weight in spline.weights]
        order = spline.dxf.degree + 1
        BSpline.__init__(self, control_points, order, knots, weights)
        points = [point for point in self.approximate()]
        Element.__init__(self, spline, points)

    def slice(self, step=1):
        # TODO: использовать функции бибилиотеки ezdxf для нарезки сплайна
        pass


class Line(Element):
    """
    Подкласс для объекта Линия
    """

    def __init__(self, line):
        points = [Vector(line.dxf.start), Vector(line.dxf.end)]
        super().__init__(line, points)


class Circle(Element):
    """
    Подкласс для объекта Окружность
    """

    def __init__(self, circle):
        super().__init__(circle)
        self.center = circle.dxf.center  # type: Vector
        self.radius = circle.dxf.radius  # type: float
        self.points = [self.first, self.last]

    @property
    def first(self):
        return self.center.replace(x=self.center.x + self.radius)

    @property
    def last(self):
        return self.first

    @property
    def flatLength(self):
        try:
            return self._flatLength
        except AttributeError:
            flatLength = 2 * pi * self.radius
            self._flatLength = flatLength
            return flatLength

    def slice(self, step=1):
        n_steps = int(self.flatLength / step)
        angle_step = pi / n_steps
        sliced = []
        for i in range(n_steps + 1):
            sliced.append(self.first.rotate(angle_step * i))
        sliced.append(self.last)
        self.points = sliced
        self.sliced = True
        try:
            del self._length
        except AttributeError:
            pass


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
        self.points = [self.first, self.last]

    @property
    def first(self):
        return Vector.from_angle(self.startAngle,
                                 self.radius) + self.center if not self.backwards else Vector.from_angle(self.endAngle,
                                                                                                         self.radius) + self.center

    @property
    def last(self):
        return Vector.from_angle(self.endAngle, self.radius) + self.center if not self.backwards else Vector.from_angle(
            self.startAngle, self.radius) + self.center

    @property
    def flatLength(self):
        try:
            return self._flatLength
        except AttributeError:
            flatLength = (self.endAngle - self.startAngle) * self.radius
            self._flatLength = flatLength
            return flatLength

    def slice(self, step=1):
        n_steps = int(self.flatLength / step)
        angle_step = (self.endAngle - self.startAngle) / n_steps
        sliced = []
        for i in range(n_steps + 1):
            sliced.append(self.first.rotate(i * angle_step))
        sliced.append(self.last)
        self.sliced = True
        self.points = sliced
        try:
            del self._length
        except AttributeError:
            pass

    def __str__(self):
        super().__str__()
        return 'Arc object: ' + super().__str__()


class Ellipse(Element):
    # TODO: написать обработку эллипсов
    pass


class Contour:
    def __init__(self, elements: List[Element] = None):
        """
        :param elements: элементы составляющие контур
        """
        if elements is None:
            self.elements = []
            self.n_elements = 0
            self.closed = False
            self.backwards = False
        else:
            self.elements = elements
            self.n_elements = len(elements)
            self.backwards = False
            if distance(self.first, self.last) < accuracy:
                self.closed = True
            else:
                self.closed = False

    @property
    def firstElement(self):
        return self.elements[0] if not self.backwards else self.elements[-1]

    @property
    def lastElement(self):
        return self.elements[-1] if not self.backwards else self.elements[0]

    @property
    def firstPoint(self):
        return self.firstElement.first

    @property
    def lastPoint(self):
        return self.lastElement.last

    def addElement(self, element):
        """
        Добавить элемент в конец контура
        :param Element element: элементр контура
        """
        self.elements.append(element)
        self.n_elements += 1

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
        self.first = self.elements[0].firstPoint()
        return self.first

    def lastPoint(self):
        self.last = self.elements[-1].lastPoint()
        return self.last


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
            self.elements = []  # type: List[Element]
            self.contours = []  # type: List[Contour]
            self.length = 0
            self.flatLength = 0
            self.center = (0, 0)
            self.offs = offset
            self.rotation = rotation
            self.path = []  # type: List[Element]
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

    def addZ(self, pcd_xy=None, pcd_z=None, constantShift=None):
        if constantShift is not None:
            for element in self.elements:
                element.addZ(constantShift=constantShift)
            self.calculateLength()
            return None
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
        :return list of Element path: отсортированный и ориентированный массив элементов
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
            elements.sort(key=lambda x: x.bestDistance(current.lastPoint()))
        self.path = path
        print('Сформирована очередность элементов.')

    def findContours(self):
        path = self.path.copy()
        contour = Contour([path[0]])
        for e1, e2 in pairwise(path):
            d = distance(e1.lastPoint(), e2.firstPoint(), True)
            if d < accuracy ** 2:
                contour.addElement(e2)
            else:
                self.contours.append(contour)
                contour = Contour([e2])
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
