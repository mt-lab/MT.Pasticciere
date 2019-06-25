"""
dxf2gcode.py
Author: bedlamzd of MT.lab

Чтение .dxf файла, слайсер этого dxf, перевод в трёхмерное пространство на рельеф объектов,
генерация gcode в соответствующий файл
"""
import ezdxf as ez
from global_variables import *
from elements import *
from utilities import *
from scanner import *
import pygcode as pg

Z_up = Z_max + 3  # later should be cloud Z max + few mm сейчас это глобальный максимум печати принтера по Z
extrusionCoefficient = 0.41


# when path is a set of elements
def gcode_generator(path, preGcode=[], postGcode=[]):
    # TODO: префиксный и постфиксный Gcode
    # TODO: генерация кода по печенькам
    # TODO: генерация кода по слоям в рисунке (i.e. отдельным контурам)
    gcode = []
    last_point = (0, 0, 0)
    E = 0
    gcode.append('G28')
    for count, element in enumerate(path, 1):
        way = element.getSlicedPoints()
        gcode.append(f'; {count:3d} element')
        if distance(last_point, way[0]) > accuracy:
            gcode.append(str(pg.GCodeRapidMove(Z=Z_up)))
            gcode.append(str(pg.GCodeRapidMove(X=way[0][X], Y=way[0][Y])))
            gcode.append(str(pg.GCodeRapidMove(Z=way[0][Z])))
            last_point = way[0]
        for point in way[1:]:
            E += round(extrusionCoefficient * distance(last_point, point), 3)
            gcode.append(str(pg.GCodeLinearMove(X=point[X], Y=point[Y], Z=point[Z])) + f' E{E:3.3f}')
            last_point = point
        last_point = way[-1]
    return gcode


def dxfReader(dxf, modelspace, elementsHeap=[]):  # at first input is modelspace
    """
    Рекуррентная функция
    Собирает данные об элементах в dxf с заходом во все внутренние блоки. При этом описывая их
    собственным классом Element для удобства

    :param dxf: исходное векторное изображение
    :param modelspace: пространство элементов изображения, либо пространство блока с элементами
    :param elementsHeap: массив в который собираются все элементы
    :return elements_heap: массив со всеми элементами из dxf
    """
    # TODO: чтение по отдельным слоям (i.e. отдельным контурам)
    for element in modelspace:
        # если элемент это блок, то пройтись через вложенные в него элементы
        if element.dxftype() == 'INSERT':
            block = dxf.blocks[element.dxf.name]
            dxfReader(dxf, block, elementsHeap)
        # если элемент не блок и при этом переопределяем, то записать его переопределение в массив
        elif elementRedef(element):
            elementsHeap.append(elementRedef(element))
        # если переопределение вернуло пустой элемент (ещё не описанный), то написать об этом
        else:
            print('empty element')
    return elementsHeap


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
    # else:
    #     return Element(element)


def organizePath(elements, start_point=(0, 0)):
    """
    Сортирует и ориентирует элементы друг за другом относительно данной точки

    :param elements: элементы, которые необходимо сориентировать и сортировать
    :param start_point: точка, относительно которой выбирается первый элемент
    :return path: отсортированный и ориентированный массив элементов
    """
    path = []
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
    return path


def processPath(path, offset=(0, 0), pathToPly=PCD_PATH):
    """
    Обработка элементов рисунка, их нарезка (slicing), смещение и добавление координаты по Z

    :param path: массив элементов
    :param offset: смещение
    :param pathToPly: путь до облака точек
    :return: None
    """
    pcd, pcd_xy, pcd_z = readPointCloud(pathToPly)
    # slice dxf and add volume to it, also add offset
    for element in path:
        element.slice(step)
        element.setOffset(offset)
        element.addZ(pcd_xy, pcd_z)


def writeGcode(gcodeInstructions, filename='cookie.gcode'):
    """
    Генерирует текстовый файл с инструкциями для принтера

    :param gcodeInstructions: массив строк с командами для принтера
    :param filename: имя файла для записи команд
    :return: None
    """
    with open(filename, 'w+') as gcode:
        for line in gcodeInstructions:
            gcode.write(line + '\n')


def dxf2gcode(pathToDxf=DXF_PATH, pathToPly=PCD_PATH):
    """
    Функция обработки dxf в Gcode

    :param pathToDxf: путь до рисунка
    :param pathToPly: путь до облака точек
    :param offset: смещение рисунка
    :return: None
    """
    # TODO: переписать под работу с классом печенек

    # прочесть dxf
    dxf = ez.readfile(pathToDxf)
    # пространство элементов модели
    msp = dxf.modelspace()
    # получить все элементы из рисунка
    elementsHeap = dxfReader(dxf, msp)

    # сформировать порядок элементов для печати
    path = organizePath(elementsHeap)

    # нарезать рисунок, сместить, добавить координату Z
    cookies = findCookies()[0]
    offset = cookies[0][0][::-1]
    processPath(path, offset, pathToPly)

    # сгенерировать инструкции для принтера
    gcodeInstructions = gcode_generator(path)

    # записать инструкции в текстовый файл
    writeGcode(gcodeInstructions)
