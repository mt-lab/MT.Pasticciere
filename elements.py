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
from numpy import sign
# from tkinter import *
from utilities import X, Y, Z, pairwise, diap, find_point_in_cloud, distance, generate_ordered_numbers, \
    apprx_point_height
from numpy import sqrt, cos, sin, pi, arctan


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
        self.with_z = False
        self.backwards = False
        self._length = None
        self._flat_length = None

    @property
    def first(self) -> Vector:
        return self.points[0] if not self.backwards else self.points[-1]

    @property
    def last(self) -> Vector:
        return self.points[-1] if not self.backwards else self.points[0]

    @property
    def centroid(self) -> Vector:
        try:
            return self._centroid
        except AttributeError:
            centroid = NULLVEC
            for p1, p2 in pairwise(self.points):
                centroid += p1.lerp(p2)
            self._centroid = centroid
            return centroid

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for v1, v2 in pairwise(self.points):
                length += v1.distance(v2)
            self._length = length
        return self._length

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for v1, v2 in pairwise(self.points):
                flat_length += v1.vec2.distance(v2.vec2)
            self._flat_length = flat_length
        return self._flat_length

    def __str__(self) -> str:
        return f'Element: {self.entity.dxftype()}\n ' + \
               f'first point: {self.first}\n ' + \
               f'last point: {self.last}'

    def __repr__(self) -> str:
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

    def best_distance(self, point: 'Vector' = NULLVEC) -> float:
        """
        Вычисляет с какой стороны точка находится ближе к элементу и ориентирует его соответственно

        :param point: точка от которой считается расстояние
        :return: минимальное расстояние до одного из концов объекта
        """
        dist2first = self.points[0].distance(point)
        dist2last = self.points[-1].distance(point)
        self.backwards = dist2last < dist2first
        return min(dist2first, dist2last)

    def get_points(self) -> List[Vector]:
        """
        Возвращает точки
        """
        return self.points if not self.backwards else self.points[::-1]

    def get_sliced_points(self) -> List[Vector]:
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
        self._length = None

    # def addZ(self, pcd_xy=None, pcd_z=None, pcd=None, constantShift=None):
    #     """
    #     Добавить координату Z к элементу
    #     :param pcd_xy: часть облака точек с X и Y координатами
    #     :param pcd_z: часть облака точек с Z координатами
    #     :param float constant_shift: для задания одной высоты всем точкам
    #     :return: None
    #     """
    #     if constant_shift is not None:
    #         self.points = [v.replace(z=constant_shift) for v in self.points]
    #         return None
    #     else:
    #         if pcd_z is None or pcd_xy is None:
    #             if pcd is None:
    #                 raise Exception('Point cloud is needed.')
    #             else:
    #                 pcd_xy, pcd_z = np.split(pcd, [Z], axis=1)
    #         self.points = [v.replace(z=findPointInCloud(v.xyz, pcd_xy, pcd_z)) for v in self.points]
    #         self.withZ = True
    #     try:
    #         del self._length
    #     except AttributeError:
    #         pass

    def add_z(self, height_map: Optional[np.ndarray] = None, constant_shift=0):
        if height_map is None:
            pass
        self.points = [v.replace(z=apprx_point_height(v, height_map)) for v in self.points]
        self.with_z = True
        self._length = None


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
        self._length = None


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
        return self.points[0] if not self.backwards else self.points[-1]

    @property
    def last(self):
        return self.points[-1] if not self.backwards else self.points[0]

    def slice(self, step=1):
        # n1, n2 = 0, 0
        # t = 0.001
        # dt = 0.0001
        # prev_point = self.point(0)
        # prev_e = 0
        # sliced = [prev_point]
        # while t < self.max_t:
        #     n1 += 1
        #     p = self.point(t + dt)
        #     d = p.distance(prev_point)
        #     e = step - d
        #     while abs(e) > step * .2:
        #         n2 += 1
        #         if sign(prev_e * e) > 0:
        #             dt += abs(dt) * sign(e)
        #             p = self.point(t + dt)
        #             d = p.distance(prev_point)
        #             prev_e = e
        #             e = step - d
        #         elif sign(prev_e * e) < 0:
        #             dt = 3 / 4 * abs(dt) * sign(e)
        #             p = self.point(t + dt)
        #             d = p.distance(prev_point)
        #             e = step - d
        #         elif prev_e == 0:
        #             prev_e = e
        #             continue
        #         else:
        #             raise Exception('wtf')
        #         print(n1, n2, e, t, dt, p)
        #     n2 = 0
        #     sliced.append(p)
        #     prev_point = p
        #     t += dt
        # if not self.point(self.max_t).isclose(prev_point):
        #     sliced.append(self.point(self.max_t))
        self.sliced = True
        points = [Vector(point) for point in self.approximate(int(self.max_t / step))]
        self.points = points
        self._length = None


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
        self._length = None


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
    def flat_length(self):
        if self._flat_length is None:
            flat_length = 2 * pi * self.radius
            self._flat_length = flat_length
        return self._flat_length

    def slice(self, step=1):
        n_steps = int(self.flat_length / step)
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
        self._length = None

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
        self.start_angle = arc.dxf.start_angle * pi / 180  # в радианах
        self.end_angle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.start_angle > self.end_angle:
            self.end_angle += 2 * pi
        points = [Vector.from_angle(self.start_angle, self.radius) + self.center,
                  Vector.from_angle(self.end_angle, self.radius) + self.center]
        super().__init__(arc, points=points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            centroid_x = self.radius / self.flat_length * (sin(self.end_angle) - sin(self.start_angle)) + self.center.x
            centroid_y = self.radius / self.flat_length * (cos(self.start_angle) - cos(self.end_angle)) + self.center.y
            self._centroid = Vector(centroid_x, centroid_y, 0)
            return self._centroid

    @property
    def flat_length(self):
        if self._flat_length is None:
            flat_length = (self.end_angle - self.start_angle) * self.radius
            self._flat_length = flat_length
        return self._flat_length

    def slice(self, step=1):
        n_steps = int(self.flat_length / step)
        angle_step = (self.end_angle - self.start_angle) / n_steps
        sliced = []
        v = Vector()
        for i in range(n_steps + 1):
            v = self.points[0] - self.center
            v = v.rotate(i * angle_step)
            v += self.center
            sliced.append(v)
        if not v.isclose(self.points[-1]):
            sliced.append(self.points[-1])
        self.sliced = True
        self.points = sliced
        self._length = None

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
        self._length = None
        self._flat_length = None
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
            if self.first_point == self.last_point:
                self.closed = True
            else:
                self.closed = False

    def __add__(self, other: Union['Contour', Element]) -> 'Contour':
        if isinstance(other, Contour):
            if not len(self):
                elements = other.elements
                self._length = None
                self._flat_length = None
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
            if self.last_point == other.first_point:
                elements = self.elements + other.elements
                # return Contour(elements)
            elif self.last_point == other.last_point:
                elements = self.elements + other.elements[::-1]
                # return Contour(elements)
            elif self.first_point == other.last_point:
                elements = other.elements + self.elements
                # return Contour(elements)
            elif self.first_point == other.first_point:
                elements = other.elements[::-1] + self.elements
                # return Contour(elements)
            else:
                raise Exception('Contours not connected.')
            # return Contour(elements)
        elif isinstance(other, Element):
            if self.last_point == other.first:
                elements = self.elements + [other]
                # return Contour(elements)
            elif self.last_point == other.last:
                elements = self.elements + [other]
                other.backwards = not other.backwards
                # return Contour(elements)
            elif self.first_point == other.last:
                elements = [other] + self.elements
                # return Contour(elements)
            elif self.first_point == other.first:
                elements = [other] + self.elements
                other.backwards = not other.backwards
                # return Contour(elements)
            else:
                raise Exception('Shapes not connected.')
        else:
            raise TypeError('Can add only Contour or Element')
        self._length = None
        self._flat_length = None
        return Contour(elements)

    def add_element(self, element: Element):
        if element in self.elements:
            raise Exception('Element is in contour already.')
        if not isinstance(element, Element):
            raise TypeError('Adding object should be Element.')
        if self.first_point == element.last:
            self.elements = [element] + self.elements
        elif self.last_point == element.first:
            self.elements += [element]
        elif self.first_point == element.first:
            element.backwards = not element.backwards
            self.elements = [element] + self.elements
        elif self.last_point == element.last:
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
    def first_element(self) -> Element:
        return self.elements[0]

    @property
    def last_element(self) -> Element:
        return self.elements[-1]

    @property
    def first_point(self) -> Vector:
        return self.first_element.first

    @property
    def last_point(self) -> Vector:
        return self.last_element.last

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for element in self.elements:
                flat_length += element.flat_length
            self._flat_length = flat_length
        return self._flat_length

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
        return self._length

    def isclose(self, other: Union[Vector, Element, "Contour"], abs_tol: float = 1e-12) -> bool:
        if isinstance(other, Vector):
            close2first = self.first_point.isclose(other, abs_tol)
            close2last = self.last_point.isclose(other, abs_tol)
            return close2first or close2last
        elif isinstance(other, Element):
            close2first = self.first_point.isclose(other.first, abs_tol) or self.first_point.isclose(other.last,
                                                                                                     abs_tol)
            close2last = self.last_point.isclose(other.first, abs_tol) or self.last_point.isclose(other.last, abs_tol)
            return close2first or close2last
        elif isinstance(other, Contour):
            close2first = self.first_point.isclose(other.first_point, abs_tol) or self.first_point.isclose(
                other.last_point,
                abs_tol)
            close2last = self.last_point.isclose(other.first_point, abs_tol) or self.last_point.isclose(
                other.last_point,
                abs_tol)
            return close2first or close2last
        else:
            raise TypeError('Should be Vector or Element or Contour.')

    def best_distance(self, point: Vector = NULLVEC) -> float:
        dist2first = 0 if self.first_point == point else self.first_point.distance(point)
        dist2last = 0 if self.last_point == point else self.last_point.distance(point)
        return min(dist2first, dist2last)

    def get_points(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.get_points()
        return points

    def get_sliced_points(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.get_sliced_points()
        return points


class Layer:
    number_generator = generate_ordered_numbers()

    def __init__(self, name=None, contours: Union[List[Contour], Contour] = None, priority=None):
        if isinstance(contours, List):
            self.contours = contours
        elif isinstance(contours, Contour):
            self.contours = [contours]
        elif contours is None:
            self.contours = []
        self.number = next(Layer.number_generator)
        self.name = name if name is not None else f'Layer {self.number}'
        self.cookieContour = True if name == 'Contour' else False
        self.priority = priority if priority is not None else 0

    def add_contour(self, contours: Union[List[Contour], Contour]):
        if isinstance(contours, List):
            self.contours += contours
        elif isinstance(contours, Contour):
            self.contours += [contours]

    def get_elements(self):
        elements = []
        for contour in self.contours:
            elements += contour.elements
        return elements


class Drawing:
    """

    Attributes:
        dxf: An ezdxf Drawing which basically contains all the necessary data
        modelspace: A dxf.modelspace(), only for a convenience
        layers Dict[str, Layer] : A dict of [layer.name, layer]
        elements List[Element]: Contains all graphic entities from dxf.
        contours List[Contour]: Contains all contours found in layers.
        center Vector: Drawing geometrical center.
        rotation float: Drawing angle or orientation.
        organized bool: True if elements are ordered and contours are constructed
    """

    # TODO: шаблон dxf по которому рисунок делится на слои:
    #   0 - общий контур печенья, по которому найти центр и поворот рисунка
    #   1 - самый внешний/важный контур
    #   ...
    #   last - элементы для печати в конце
    def __init__(self, dxf=None, center: Vector = None, rotation: float = None):
        """
        :param dxf: открытый библиотекой рисунок
        :param center: смещение центра рисунка
        :param rotation: угол поворота рисунка (его ориентация)
        lookup Drawing for more
        """
        self.layers = {}  # type: Dict[str, Layer]
        self.elements = []  # type: List[Element]
        self.contours = []  # type: List[Contour]
        self.organized = False  # type: bool
        self._length = None
        self._flat_length = None
        if dxf is None:
            self.dxf = None
            self.modelspace = None
        else:
            self.dxf = dxf
            self.modelspace = self.dxf.modelspace()
            self.read_by_layer()
        self._center, self._rotation = self.find_center_and_rotation()
        if center is not None:
            self.center = center
        if rotation is not None:
            self.rotation = rotation

    def __str__(self):
        return f'Геометрический центр рисунка: X: {self.center[X]:4.2f} Y: {self.center[Y]:4.2f} мм\n' + \
               f'Ориентация рисунка: {self.rotation * 180 / pi: 4.2f} градуса\n' + \
               f'Общая плоская длина рисунка: {self.flat_length: 4.2f} мм'

    @property
    def center(self) -> Vector:
        return self._center

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
        self.rotate(angle - self._rotation)
        self._rotation = angle

    def rotate(self, angle: float):
        for element in self.elements:
            element.rotate(angle, self.center)

    def find_center_and_rotation(self) -> Tuple[Vector, float]:
        """
        Расчитывает геометрический центр рисунка
        :return:
        """
        cookie_contour_layer = self.layers.get('Contour')
        if cookie_contour_layer is None:
            # TODO: place warning here
            return NULLVEC, 0
        else:
            # points = []
            # for element in cookie_contour_layer.getElements():
            #     element.slice(0.1)
            #     points += element.getPoints()
            # points = np.asarray([list(v.vec2) for v in points])
            # # TODO: Переделать, почему то cv2.moments не подходит
            # M = moments(points)
            # cx = M['m10'] / M['m00']
            # cy = M['m01'] / M['m00']
            # a = M['m20'] / M['m00'] - cx ** 2
            # b = 2 * (M['m11'] / M['m00'] - cx * cy)
            # c = M['m02'] / M['m00'] - cy ** 2
            # theta = 1 / 2 * arctan(b / (a - c)) + (a < c) * pi / 2
            # return Vector(cx, cy), theta
            return NULLVEC, 0

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
        return self._length

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for element in self.elements:
                flat_length += element.flat_length
            self._flat_length = flat_length
        return self._flat_length

    def read_dxf(self, root):
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                self.read_dxf(block)
            elif element_redef(element):
                self.elements.append(element_redef(element))
        self.organized = False
        print('dxf прочтён.')

    def read_entities(self, root, entities=None):
        if entities is None:
            entities = []
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                entities += self.read_entities(block)
            elif element_redef(element):
                entities.append(element_redef(element))
        print('элементы получены')
        return entities

    def read_by_layer(self):
        layers = {}
        elements_in_dwg = []
        contours_in_dwg = []
        for layer in self.dxf.layers:
            name = layer.dxf.name
            print(f'чтение слоя {name}')
            if name == 'Defpoints':
                print('    пропуск')
                continue
            priority = findall('\d+', name)
            priority = int(priority[0]) if priority else None
            entities_in_layer = self.modelspace.query(f'*[layer=="{name}"]')
            entities_in_layer = self.read_entities(entities_in_layer)
            if not entities_in_layer:
                continue
            entities_in_layer = self.organize_entities(entities_in_layer)
            elements_in_dwg += entities_in_layer
            contours_in_layer = self.make_contours(entities_in_layer)
            contours_in_dwg += contours_in_layer
            layers[name] = Layer(name, contours_in_layer, priority)
        self.layers = layers
        self.elements = elements_in_dwg
        self.contours = contours_in_dwg
        self.organized = True
        print('файл прочтён')

    def slice(self, step: float = 1.0):
        for element in self.elements:
            element.slice(step)
        self._length = None
        print(f'Объекты нарезаны с шагом {step:2.1f} мм')

    def add_z(self, height_map: np.ndarray, constant_shift=0):
        # def add_z(self, pcd_xy=None, pcd_z=None, constant_shift=None):
        # if constant_shift is not None:
        #     for element in self.elements:
        #         element.add_z(constant_shift=constant_shift)
        # elif pcd_xy is not None and pcd_z is not None:
        #     for element in self.elements:
        #         element.add_z(pcd_xy, pcd_z)
        # else:
        #     raise Exception('No height data.')
        if height_map is None:
            pass
        for element in self.elements:
            element.add_z(height_map, constant_shift)
        self._length = None

    def organize_entities(self, entities: List[Element], start_point: Vector = NULLVEC):
        path = []
        elements = entities
        # сортировать элементы по их удалению от точки
        elements.sort(key=lambda x: x.best_distance(start_point))
        while len(elements) != 0:
            # первый элемент в списке (ближайший к заданной точке) - текущий
            current = elements[0]
            # добавить его в сориентированный массив
            path.append(current)
            # убрать этот элемент из неотсортированного списка
            elements.pop(0)
            # отсортировать элементы по их удалению от последней точки предыдущего элемента
            elements.sort(key=lambda x: x.best_distance(current.last))
        print('элементы отсортированы')
        return path

    def make_contours(self, entities: List[Element]):
        contour = Contour([entities[0]])
        contours = []
        for element in entities[1:]:
            if contour.isclose(element) and not contour.closed:
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        i = -1
        while i < len(contours) - 1:
            if contours[i].isclose(contours[i + 1]) and not contours[i].closed and not contours[i + 1].closed:
                if i == -1:
                    contours[i + 1] = contours[i] + contours[i + 1]
                    del contours[i]
                else:
                    contours[i:i + 2] = [contours[i] + contours[i + 1]]
            else:
                i += 1
        print('контуры составлены')
        return contours

    def organize_elements(self, start_point=(0, 0)):
        """
        Сортирует и ориентирует элементы друг за другом относительно данной точки
        :param start_point: точка, относительно которой выбирается первый элемент
        :return list of Element path: отсортированный и ориентированный массив элементов
        """
        path = []
        elements = self.elements.copy()
        # сортировать элементы по их удалению от точки
        elements.sort(key=lambda x: x.best_distance(start_point))
        while len(elements) != 0:
            # первый элемент в списке (ближайший к заданной точке) - текущий
            current = elements[0]
            # добавить его в сориентированный массив
            path.append(current)
            # убрать этот элемент из неотсортированного списка
            elements.pop(0)
            # отсортировать элементы по их удалению от последней точки предыдущего элемента
            elements.sort(key=lambda x: x.best_distance(current.last))
        self.elements = path
        self.organized = True
        print('Сформирована очередность элементов.')

    def find_contours(self):
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
