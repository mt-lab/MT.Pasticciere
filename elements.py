"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""

import ezdxf as ez
import numpy as np
from utilities import X, Y, Z, pairwise, diap, findPointInCloud, distance
from math import sqrt, cos, sin, pi


class Element:
    """
    Общий класс с функциями общими для всех элементов, многие оверрайдятся в конкретных случаях
    """

    def __init__(self, entity, first=(0, 0), last=(0, 0)):
        """
        Конструктор объекта

        :param entity: элемент из dxf
        :param first: первая точка элемента
        :param last: последняя точка элемента
        """
        self.entity = entity
        self.points = []
        self.first = first
        self.last = last
        self.sliced = []
        self.backwards = False
        self.offset = (0, 0)
        self.length = 0

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
        dist2first = sqrt(abs(self.first[X] - point[X]) ** 2 + abs(self.first[Y] - point[Y]) ** 2)
        dist2last = sqrt(abs(self.last[X] - point[X]) ** 2 + abs(self.last[Y] - point[Y]) ** 2)
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

    def getLength(self):
        """
        Рассчитать длину элемента
        """
        pass

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

    def addZ(self, pcd_xy, pcd_z):
        """
        Добавить координату Z к элементу
        :param pcd_xy: часть облака точек с X и Y координатами
        :param pcd_z: часть облака точек с Z координатами
        :return: None
        """
        if len(self.sliced) != 0:
            for p in self.sliced:
                p[Z] = findPointInCloud(p, pcd_xy, pcd_z, self.offset)
        elif len(self.points) != 0:
            for p in self.points:
                p.append(findPointInCloud(p, pcd_xy, pcd_z, self.offset))


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
        self.length = self.getLength()

    def getLength(self):
        length = 0
        for p1, p2 in pairwise(self.points):
            length += distance(p1, p2)
        return length


class LWPolyline(Polyline):
    # TODO: написать обработку LW полилиний
    pass


class Spline(Element):
    """
    Подкласс для объека Сплайн
    """

    # TODO: написать обработку сплайнов для нарезки (использовать чужой dxf2gcode)
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
        self.length = self.getLength()

    def getLength(self):
        return distance(self.first, self.last)


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
        self.length = self.getLength()

    def getLength(self):
        return (self.endAngle - self.startAngle) * self.radius

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
        self.length = self.getLength()


class Ellipse(Element):
    # TODO: написать обработку эллипсов
    pass
