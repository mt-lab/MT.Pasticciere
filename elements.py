"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""

from typing import List, Union, Any, Optional, Tuple, Dict
from cv2 import moments
import math
import ezdxf as ez
import ezdxf.math as geom
from ezdxf.math.vector import Vector, NULLVEC
from ezdxf.math.bspline import BSpline
from re import findall
import numpy as np
# from tkinter import *
from utilities import X, Y, Z, pairwise, diap, findPointInCloud, distance, generate_ordered_numbers
from numpy import sqrt, cos, sin, pi, arctan
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
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            centroid = NULLVEC
            for p1, p2 in pairwise(self.points):
                centroid += p1.lerp(p2)
            self._centroid = centroid
            return centroid

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

    def rotate(self, angle: float, center: Vector = None):
        if center is not None:
            self.translate(-center)
        self.points = [v.rotate(angle) for v in self.points]
        if center is not None:
            self.translate(center)

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
        sliced = [self.points[0]]
        for start, end in pairwise(self.points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
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

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            points = [Vector(point) for point in self.entity.points()]
            centroid = NULLVEC
            for p1, p2 in pairwise(points):
                centroid += p1.lerp(p2)
            self._centroid = centroid
            return centroid

    def slice(self, step=1):
        points = [Vector(point) for point in self.entity.points()]
        sliced = [points[0]]
        for start, end in pairwise(points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
                sliced.append(end)
        self.points = sliced
        self.sliced = True
        try:
            del self._length
        except AttributeError:
            pass


class LWPolyline(Polyline):
    # TODO: написать обработку LW полилиний
    pass


class Spline(Element, BSpline):
    """
    Подкласс для объека Сплайн
    """

    def __init__(self, spline):
        control_points = [Vector(point) for point in spline.control_points]
        knots = [knot for knot in spline.knots]
        weights = [weight for weight in spline.weights] if spline.weights else None
        order = spline.dxf.degree + 1
        BSpline.__init__(self, control_points, order, knots, weights)
        points = [point for point in self.approximate()]
        Element.__init__(self, spline, points)

    @property
    def first(self):
        return self.point(0) if not self.backwards else self.point(self.max_t)

    @property
    def last(self):
        return self.point(self.max_t) if not self.backwards else self.point(0)

    def slice(self, step=1):
        t = 0
        dt = 1
        prev = self.point(0)
        sliced = [prev]
        while t <= self.max_t:
            p = self.point(t)
            if p.distance(prev) > step:
                t -= dt
                dt = dt / 2
                t += dt
                continue
            elif p.distance(prev) < 0.8 * step:
                dt = dt * 2
                t += dt
                continue
            sliced.append(p)
            prev = p
            t += dt
        if not self.point(self.max_t).isclose(prev):
            sliced.append(self.point(self.max_t))
        self.sliced = True
        self.points = sliced
        try:
            del self._length
        except AttributeError:
            pass


class Line(Element):
    """
    Подкласс для объекта Линия
    """

    def __init__(self, line):
        points = [Vector(line.dxf.start), Vector(line.dxf.end)]
        super().__init__(line, points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            self._centroid = Vector(self.entity.dxf.start).lerp(self.entity.dxf.end)
            return self._centroid

    def slice(self, step=1):
        points = [Vector(self.entity.dxf.start), Vector(self.entity.dxf.end)]
        sliced = [points[0]]
        for start, end in pairwise(points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
                sliced.append(end)
        self.points = sliced
        self.sliced = True
        try:
            del self._length
        except AttributeError:
            pass


class Circle(Element):
    """
    Подкласс для объекта Окружность
    """

    def __init__(self, circle):
        self.center = circle.dxf.center  # type: Vector
        self.radius = circle.dxf.radius  # type: float
        points = [self.center.replace(x=self.center.x + self.radius),
                  self.center.replace(x=self.center.x + self.radius)]
        super().__init__(circle, points=points)

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
        angle_step = 2 * pi / n_steps
        sliced = []
        v = Vector()
        for i in range(n_steps + 1):
            v = self.first - self.center
            v = v.rotate(i * angle_step)
            v += self.center
            sliced.append(v)
        if not v.isclose(self.last):
            sliced.append(self.last)
        self.points = sliced
        self.sliced = True
        try:
            del self._length
        except AttributeError:
            pass

    @property
    def centroid(self):
        return self.center


class Arc(Element):
    """
    Подклас для объекта Дуга
    """

    def __init__(self, arc):
        self.center = arc.dxf.center  # type: Vector
        self.radius = arc.dxf.radius  # type: float
        self.startAngle = arc.dxf.start_angle * pi / 180  # в радианах
        self.endAngle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.startAngle > self.endAngle:
            self.endAngle += 2 * pi
        points = [Vector.from_angle(self.startAngle, self.radius) + self.center,
                  Vector.from_angle(self.endAngle, self.radius) + self.center]
        super().__init__(arc, points=points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            centroid_x = self.radius / self.flatLength * (sin(self.endAngle) - sin(self.startAngle)) + self.center.x
            centroid_y = self.radius / self.flatLength * (cos(self.startAngle) - cos(self.endAngle)) + self.center.y
            self._centroid = Vector(centroid_x, centroid_y, 0)
            return self._centroid

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
        v = Vector()
        for i in range(n_steps + 1):
            v = self.first - self.center
            v = v.rotate(i * angle_step)
            v += self.center
            sliced.append(v)
        if not v.isclose(self.last):
            sliced.append(self.last)
        self.sliced = True
        self.points = sliced
        try:
            del self._length
        except AttributeError:
            pass

    def __str__(self):
        return 'Arc object: ' + super().__str__()


class Ellipse(Element):
    # TODO: написать обработку эллипсов
    pass


class Contour:
    def __init__(self, elements: Union[List[Element], Element] = None):
        """
        :param elements: элементы составляющие контур
        """
        if elements is None:
            self.elements = []
            self.closed = False
        else:
            if isinstance(elements, List):
                self.elements = elements
            elif isinstance(elements, Element):
                self.elements = [elements]
            else:
                raise TypeError('Contour should be either List[Element] or Element.')
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

    def addElement(self, element: Element):
        if element in self.elements:
            raise Exception('Element is in contour already.')
        if not isinstance(element, Element):
            raise TypeError('Adding object should be Element.')
        if self.firstPoint == element.last:
            self.elements = [element] + self.elements
        elif self.lastPoint == element.first:
            self.elements += [element]
        elif self.firstPoint == element.first:
            element.backwards = not element.backwards
            self.elements = [element] + self.elements
        elif self.lastPoint == element.last:
            element.backwards = not element.backwards
            self.elements += [element]
        else:
            raise Exception('Element does not connected to contour.')

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
    def lastPoint(self) -> Vector:
        return self.lastElement.last

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


class Layer:
    number_generator = generate_ordered_numbers()

    def __init__(self, name=None, contours: Union[List[Contour], Contour] = None, priority=None):
        if isinstance(contours, List):
            self.contours = contours
        elif isinstance(contours, Contour):
            self.contours = [contours]
        self.number = next(Layer.number_generator)
        self.name = name if name is not None else f'Layer {self.number}'
        self.cookieContour = True if name == 'Contour' else False
        self.priority = priority

    def addContour(self, contours: Union[List[Contour], Contour]):
        if isinstance(contours, List):
            self.contours += contours
        elif isinstance(contours, Contour):
            self.contours += [contours]

    def getElements(self):
        elements = []
        for contour in self.contours:
            elements += contour.elements
        return elements


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
            self.layers = {}
            self._center = NULLVEC  # type: Vector
            self._rotation = 0  # type: float
            self.organized = False
        else:
            self.dxf = dxf
            self.modelspace = self.dxf.modelspace()
            self.elements = []  # type: List[Element]
            self.contours = []  # type: List[Contour]
            self.layers = {}  # type: Dict[str, Layer]
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

    def findCenterAndRotation(self) -> Tuple[Vector, float]:
        # TODO: расчёт центра рисунка
        """
        Расчитывает геометрический центр рисунка
        :return:
        """
        cookie_contour_layer = self.layers.get('Contour')
        if cookie_contour_layer is None:
            # TODO: place warning here
            self._center = NULLVEC
            self._rotation = 0
            return NULLVEC, 0
        else:
            points = []
            for element in cookie_contour_layer.getElements():
                element.slice(0.01)
                points += element.getPoints()
            points = [list(v.vec2) for v in points]
            M = moments(points)
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            a = M['m20'] / M['m00'] - cx ** 2
            b = 2 * (M['m11'] / M['m00'] - cx * cy)
            c = M['m02'] / M['m00'] - cy ** 2
            theta = 1 / 2 * arctan(b / (a - c)) + (a < c) * pi / 2
            self._center = Vector(cx, cy)
            self._rotation = theta
            return self._center, self._rotation

    @center.setter
    def center(self, center: Union[Vector, List[float], Tuple[float]]):
        center = Vector(center)
        self.translate(center - self._center)
        self._center = center

    def translate(self, vector: Vector):
        for element in self.elements:
            element.translate(vector)

    @property
    def rotation(self) -> float:
        return self._rotation

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

    def readEntities(self, root, entities=None):
        if entities is None:
            entities = []
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                entities += self.readEntities(block, entities)
            elif elementRedef(element):
                entities.append(element)
        return entities

    def readByLayer(self):
        layers = {}
        for layer in self.dxf.layers:
            name = layer.dxf.name
            if name == 'Defpoints':
                continue
            priority = findall('\d+', name)
            priority = priority[0] if priority else None
            entities_in_layer = self.modelspace().query(f'*[layer=="{name}"]')
            entities_in_layer = self.readEntities(entities_in_layer)
            self.elements += entities_in_layer
            contours_in_layer = self.makeContours(entities_in_layer)
            self.contours += contours_in_layer
            layers[name] = Layer(name, contours_in_layer, priority)
        self.layers = layers

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

    def organizeEntities(self, entities: List[Element], start_point: Vector = NULLVEC):
        path = []
        elements = entities
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
        return path

    def makeContours(self, entities: List[Element]):
        # TODO: исправить неверный реверс элементов
        contour = Contour([entities[0]])
        contours = []
        for element in entities[1:]:
            if contour.isclose(element):
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        i = -1
        while i < len(contours) - 1:
            if contours[i].isclose(contours[i + 1]):
                if i == -1:
                    contours[i + 1] = contours[i] + contours[i + 1]
                    del contours[i]
                else:
                    contours[i:i + 2] = [contours[i] + contours[i + 1]]
            else:
                i += 1
        return contours

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
        # TODO: исправить неверный реверс элементов
        contour = Contour([self.elements[0]])
        contours = []
        for element in self.elements[1:]:
            if contour.isclose(element):
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        i = -1
        while i < len(contours) - 1:
            if contours[i].isclose(contours[i + 1]):
                if i == -1:
                    contours[i + 1] = contours[i] + contours[i + 1]
                    del contours[i]
                else:
                    contours[i:i + 2] = [contours[i] + contours[i + 1]]
            else:
                i += 1
        self.contours = contours
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
