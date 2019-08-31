"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""
# TODO Переписать ВСЁ используя библиотеки для работы с геометрией (pyeuclid)
#   или написать класс vector3d с необходимыми операциями

from typing import List, Optional
import ezdxf as ez
import numpy as np
# from tkinter import *
from utilities import X, Y, Z, pairwise, diap, findPointInCloud, distance
from numpy import sqrt, cos, sin, pi
from configValues import accuracy


class Element:
    """
    Общий класс с функциями общими для всех элементов, многие оверрайдятся в конкретных случаях
    """

    def __init__(self, entity, first=(.0, .0), last=(.0, .0)):
        """
        Конструктор объекта

        :param entity: элемент из dxf
        """
        self.entity = entity
        self.points = []
        self.sliced = [] # type: List[List[float]]
        self.backwards = False
        self.first = (.0, .0, .0)
        self.last = (.0, .0, .0)
        self.offset = (0, 0)
        self.length = 0
        self.flatLength = 0

    def __str__(self):
        return f'first: {self.firstPoint()}\n' + \
               f'last: {self.lastPoint()}'

    def firstPoint(self):
        if len(self.sliced) != 0:
            self.first = self.sliced[0] if not self.backwards else self.sliced[-1]
            return self.first
        elif len(self.points) != 0:
            self.first = self.points[0] if not self.backwards else self.points[-1]
            return self.first
        else:
            self.first = 0
            return None

    def lastPoint(self):
        if len(self.sliced) != 0:
            self.last = self.sliced[-1] if not self.backwards else self.sliced[0]
            return self.last
        elif len(self.points) != 0:
            self.last = self.points[-1] if not self.backwards else self.points[0]
            return self.last
        else:
            self.last = 0
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
        dist2first = distance(self.points[0], point)
        dist2last = distance(self.points[-1], point)
        # dist2first = sqrt(abs(self.firstPoint()[X] - point[X]) ** 2 + abs(self.firstPoint()[Y] - point[Y]) ** 2)
        # dist2last = sqrt(abs(self.lastPoint()[X] - point[X]) ** 2 + abs(self.lastPoint()[Y] - point[Y]) ** 2)
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
        :param float step: шаг нарезки
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


# root = Tk()#Создаем окно
# canv = Canvas(root, width=1400,height=750)#Создаем полотно для рисования
# canv.pack()
# canv.create_line(0, 0, 1400, 0, fill='blue', arrow=LAST)#Рисуем оси со стрелочками направления
# canv.create_line(0, 0, 0, 1400, fill='blue', arrow=LAST)

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
    # def slice(self, st=1):
    # u = 10
    # x1, y1 = 0, 0
    # x2, y2 = 0, 0
    # x, y = 0, 0
    # count = len(self.points)
    # ngr = 4
    # step = count//ngr
    # first = count%ngr
    # for j in range(first, count+1, step if step > 0 else 1):
    #     if j <= count:
    #         if j == first:
    #             c = first
    #         else:
    #             c = step + 1
    #         for t in range(0, 1000, 1):
    #             t = t/1000
    #             if (c-1 >= 0):
    #                 a = t**(c-1)
    #             if t == 0:
    #                 (x, y) = 0, 0
    #             else:
    #                 k = c - 1
    #                 while k > - 1:
    #                     x = x + self.points[j-(c-k)][X] * a
    #                     y = y + self.points[j-(c-k)][Y] * a
    #                     a = a * k * (1 - t) / ((c - 1 - k + 1) * t)
    #                     k -= 1
    #             #canv.create_oval(u * x - u * 100 + 200, u * y + u * 100 + 400, u * x - u * 100 + 200+1, u * y + u * 100 + 400+1)
    #
    #             #Разбиваем сплайн на отрезки не больше заданной длины
    #             if (sqrt((x-x1)**2 + (y-y1)**2) <= st) :
    #                 (x2,y2) = (x, y)
    #             else:
    #                 if (x1,y1) != (0, 0):
    #                     g = 500
    #                     b = 600
    #                     canv.create_line(u*x1-u*100+g, u*y1+u*100+b, u*x2-u*100+g, u*y2+u*100+b)
    #                     self.sliced.append([x2, y2, 0])
    #                 print(x1, y1)
    #                 x1, y1 = x, y
    #             x, y = 0, 0


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

    def __str__(self):
        super().__str__()
        return 'Arc object: ' + super().__str__()


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
        self.n_elements += 1
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
            if d < accuracy**2:
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
