"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""
# TODO Переписать ВСЁ используя библиотеки для работы с геометрией (pyeuclid)
#   или написать класс vector3d с необходимыми операциями

from typing import List, Union, Any, Optional
import math
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
                v = start.lerp(end, param_step * i)
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
        self.startAngle = arc.dxf.start_angle * pi / 180  # в радианах
        self.endAngle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.startAngle > self.endAngle:
            self.endAngle += 2 * pi
        super().__init__(arc)

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
            self.closed = False
        else:
            self.elements = elements
            if self.firstPoint == self.lastPoint:
                self.closed = True
            else:
                self.closed = False

    def __add__(self, other: Union['Contour', Element]) -> 'Contour':
        if isinstance(other, Contour):
            if not len(self):
                elements = other.elements
                try:
                    del self._length
                    del self._flatLength
                except AttributeError:
                    pass
                return Contour(elements)
            if self.closed or other.closed:
                raise Exception('Cannot add closed contours.')
            """
             1. end to start
              c1 + c2
            2. end to end
              c1 + c2.reversed
            3. start to end
              c2 + c1
            4. start to start
              c2.reversed + c1
            """
            if self.lastPoint == other.firstPoint:
                elements = self.elements + other.elements
                # return Contour(elements)
            elif self.lastPoint == other.lastPoint:
                elements = self.elements + other.elements[::-1]
                # return Contour(elements)
            elif self.firstPoint == other.lastPoint:
                elements = other.elements + self.elements
                # return Contour(elements)
            elif self.firstPoint == other.firstPoint:
                elements = other.elements[::-1] + self.elements
                # return Contour(elements)
            else:
                raise Exception('Contours not connected.')
            # return Contour(elements)
        elif isinstance(other, Element):
            if self.lastPoint == other.first:
                elements = self.elements + [other]
                # return Contour(elements)
            elif self.lastPoint == other.last:
                elements = self.elements + [other]
                other.backwards = not other.backwards
                # return Contour(elements)
            elif self.firstPoint == other.last:
                elements = [other] + self.elements
                # return Contour(elements)
            elif self.firstPoint == other.first:
                elements = [other] + self.elements
                other.backwards = not other.backwards
                # return Contour(elements)
            else:
                raise Exception('Shapes not connected.')
        else:
            raise TypeError('Can add only Contour or Element')
        try:
            del self._length
            del self._flatLength
        except AttributeError:
            pass
        return Contour(elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item: Union[int, slice]) -> Element:
        return self.elements[item]

    def __reversed__(self):
        for element in self.elements[::-1]:
            yield element

    @property
    def firstElement(self) -> Element:
        return self.elements[0]

    @property
    def lastElement(self) -> Element:
        return self.elements[-1]

    @property
    def firstPoint(self) -> Vector:
        return self.firstElement.first

    @property
    def flatLength(self) -> float:
        try:
            return self._flatLength
        except AttributeError:
            flatLength = 0
            for element in self.elements:
                flatLength += element.flatLength
            self._flatLength = flatLength
            return flatLength

    @property
    def length(self) -> float:
        try:
            return self._length
        except AttributeError:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
            return length

    @property
    def lastPoint(self) -> Vector:
        return self.lastElement.last

    def isclose(self, other: Union[Vector, Element, "Contour"], abs_tol: float = 1e-12) -> bool:
        if isinstance(other, Vector):
            close2first = self.firstPoint.isclose(other, abs_tol)
            close2last = self.lastPoint.isclose(other, abs_tol)
            return close2first or close2last
        elif isinstance(other, Element):
            close2first = self.firstPoint.isclose(other.first, abs_tol) or self.firstPoint.isclose(other.last, abs_tol)
            close2last = self.lastPoint.isclose(other.first, abs_tol) or self.lastPoint.isclose(other.last, abs_tol)
            return close2first or close2last
        elif isinstance(other, Contour):
            close2first = self.firstPoint.isclose(other.firstPoint, abs_tol) or self.firstPoint.isclose(other.lastPoint,
                                                                                                        abs_tol)
            close2last = self.lastPoint.isclose(other.firstPoint, abs_tol) or self.lastPoint.isclose(other.lastPoint,
                                                                                                     abs_tol)
            return close2first or close2last
        else:
            raise TypeError('Should be Vector or Element or Contour.')

    def bestDistance(self, point: Vector = NULLVEC) -> float:
        dist2first = 0 if self.firstPoint == point else self.firstPoint.distance(point)
        dist2last = 0 if self.lastPoint == point else self.lastPoint.distance(point)
        return min(dist2first, dist2last)

    def getPoints(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.getPoints()
        return points

    def getSlicedPoints(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.getSlicedPoints()
        return points


class Drawing:
    # TODO: шаблон dxf по которому рисунок делится на слои:
    #   0 - общий контур печенья, по которому найти центр и поворот рисунка
    #   1 - самый внешний/важный контур
    #   ...
    #   last - элементы для печати в конце
    def __init__(self, dxf=None, center: Vector = None, rotation: float = None):
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
            self._center = NULLVEC  # type: Vector
            self._rotation = 0  # type: float
            self.organized = False
        else:
            self.dxf = dxf
            self.modelspace = self.dxf.modelspace()
            self.elements = []  # type: List[Element]
            self.contours = []  # type: List[Contour]
            self.readDxf(self.modelspace)
            self._center = NULLVEC  # type: Vector
            if center is not None:
                self.center = center
            self._rotation = 0  # type: float
            if rotation is not None:
                self.rotation = rotation
            self.organized = False
            self.organizeElements()
            self.findContours()

    def __str__(self):
        return f'Геометрический центр рисунка: X: {self.center[X]:4.2f} Y: {self.center[Y]:4.2f} мм\n' + \
               f'Ориентация рисунка: {self.rotation * 180 / pi: 4.2f} градуса\n' + \
               f'Общая плоская длина рисунка: {self.flatLength: 4.2f} мм'

    @property
    def center(self) -> Vector:
        return self._center

    def findCenter(self) -> Vector:
        # TODO: расчёт центра рисунка
        """
        Расчитывает геометрический центр рисунка
        :return:
        """
        self._center = NULLVEC
        return NULLVEC

    @center.setter
    def center(self, center: Vector):
        self.translate(center - self._center)
        self._center = center

    def translate(self, vector: Vector):
        for element in self.elements:
            element.translate(vector)

    @property
    def rotation(self) -> float:
        return self._rotation

    def findRotation(self) -> float:
        # TODO: расчёт ориентации рисунка
        """
        Расчитывает поворот рисунка
        :return:
        """
        self._rotation = 0
        return 0

    @rotation.setter
    def rotation(self, angle: float):
        self.rotate(angle)
        self._rotation += angle

    def rotate(self, angle: float):
        for element in self.elements:
            element.rotate(angle)

    @property
    def length(self) -> float:
        try:
            return self._length
        except AttributeError:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
            return length

    @property
    def flatLength(self) -> float:
        try:
            return self._flatLength
        except AttributeError:
            flatLength = 0
            for element in self.elements:
                flatLength += element.flatLength
            self._flatLength = flatLength
            return flatLength

    def readDxf(self, root):
        # TODO: read by layer
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                self.readDxf(block)
            elif elementRedef(element):
                self.elements.append(elementRedef(element))
        self.organized = False
        print('dxf прочтён.')

    def readByLayer(self):
        pass

    def slice(self, step: float = 1.0):
        for element in self.elements:
            element.slice(step)
        try:
            del self._length
        except AttributeError:
            pass
        print(f'Объекты нарезаны с шагом {step:2.1f} мм')

    def addZ(self, pcd_xy=None, pcd_z=None, constantShift=None):
        if constantShift is not None:
            for element in self.elements:
                element.addZ(constantShift=constantShift)
        elif pcd_xy is not None and pcd_z is not None:
            for element in self.elements:
                element.addZ(pcd_xy, pcd_z)
        else:
            raise Exception('No height data.')
        try:
            del self._length
        except AttributeError:
            pass

    def organizeElements(self, start_point=(0, 0)):
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
            elements.sort(key=lambda x: x.bestDistance(current.last))
        self.elements = path
        self.organized = True
        print('Сформирована очередность элементов.')

    def findContours(self):
        contour = Contour([self.elements[0]])
        contours = []
        for element in self.elements:
            if contour.isclose(element):
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        new_contours = []
        for contour in contours:
            new_contour = Contour() + contour
            for contour2 in contours:
                if contour.isclose(contour2) and not (contour is contour2):
                    new_contour += contour2
            new_contours.append(new_contour)
        self.contours = new_contours
        print('Найдены контуры.')


def elementRedef(element) -> Optional[Element]:
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
        print('Unknown element.')
        return None
