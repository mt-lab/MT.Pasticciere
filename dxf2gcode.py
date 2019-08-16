"""
dxf2gcode.py
Author: bedlamzd of MT.lab

Чтение .dxf файла, слайсер этого dxf, перевод в трёхмерное пространство на рельеф объектов,
генерация gcode в соответствующий файл
"""
import ezdxf as ez
from configValues import accuracy, sliceStep, DXF_PATH, PCD_PATH, zOffset, extrusionCoefficient
from elements import *
from utilities import readPointCloud
import globalValues
from scanner import findCookies

# TODO: написать логи

Z_max = 30
Z_up = Z_max + zOffset  # later should be cloud Z max + few mm сейчас это глобальный максимум печати принтера по Z
# TODO: написать динамический коэффициент с учетом специфики насоса
# extrusionCoefficient = 0.41  # коэффицент экструзии, поворот/мм(?)


# when path is a set of elements
def gcode_generator(listOfElements, listOfCookies, pathToPly=PCD_PATH, preGcode=None, postGcode=None):
    """
    Генерирует список с командами для принтера
    :param listOfElements: список элементов из которых состоит рисунок
    :param preGcode: код для вставки в начало
    :param postGcode: код для вставки в конец
    :return gcode: список команд в Gcode
    """
    # TODO: генерация кода по слоям в рисунке (i.e. отдельным контурам)
    # проверка наличия пре-кода и пост-кода
    if preGcode is None:
        preGcode = []
    if postGcode is None:
        postGcode = []
    gcode = []
    last_point = (0, 0, 0)  # начало в нуле
    E = 0  # начальное значение выдавливания глазури (положение мешалки)
    gcode.append('G28')  # домой
    gcode += preGcode
    # для каждого элемента в рисунке
    for count, cookie in enumerate(listOfCookies, 1):
        adjustPath(listOfElements, cookie.center, pathToPly)
        gcode.append(f'; {count:3d} cookie')
        for idx, element in enumerate(listOfElements, 1):
            way = element.getSlicedPoints()
            gcode.append(f'; {idx:3d} element')  # коммент с номером элемента
            if distance(last_point, way[0]) > accuracy:
                # если от предыдущей точки до текущей расстояние больше точности, поднять сопло и довести до нужной точки
                gcode.append(f'G0 Z{Z_up:3.3f}')
                gcode.append(f'G0 X{way[0][X]:3.3f} Y{way[0][Y]:3.3f}')
                gcode.append(f'G0 Z{way[0][Z] + zOffset:3.3f}')
                last_point = way[0]  # обновить предыдущую точку
            for point in way[1:]:
                E += round(extrusionCoefficient * distance(last_point, point), 3)
                gcode.append(f'G1 X{point[X]:3.3f} Y{point[Y]:3.3f} Z{point[Z] + zOffset:3.3f} E{E:3.3f}')
                last_point = point
            E -= extrusionCoefficient*5
            last_point = way[-1]
        gcode.append(f'G0 Z{Z_up:3.3f}')
    gcode += postGcode
    gcode.append(f'G0 Z{Z_up:3.3f}')  # в конце поднять
    gcode.append('G28')  # и увести домой
    return gcode


def dxfReader(dxf, modelspace, elementsHeap=None):  # at first input is modelspace
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
    if elementsHeap is None:
        elementsHeap = []

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
            print('undefined element')
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


def slicePath(path, step=1.0):
    """
    Обработка элементов рисунка, их нарезка (slicing), смещение и добавление координаты по Z

    :param path: массив элементов
    :param offset: смещение
    :param pathToPly: путь до облака точек
    :return: None
    """
    # slice dxf
    for element in path:
        element.slice(step)


def adjustPath(path, offset=(0, 0), pathToPly=PCD_PATH):
    pcd, pcd_xy, pcd_z = readPointCloud(pathToPly)
    # add volume to dxf, also add offset
    for element in path:
        element.setOffset(offset)
        element.addZ(pcd_xy, pcd_z, pcd)


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

    # прочесть dxf
    dxf = ez.readfile(pathToDxf)
    # пространство элементов модели
    msp = dxf.modelspace()
    # получить все элементы из рисунка
    elementsHeap = dxfReader(dxf, msp)
    print('dxf прочтён')

    # сформировать порядок элементов для печати
    path = organizePath(elementsHeap)
    print('Сформирован порядок печати')

    # нарезать рисунок
    slicePath(path, sliceStep)
    print(f'Объекты нарезаны с шагом {sliceStep:2.1f} мм')

    # сгенерировать инструкции для принтера
    cookies, _ = findCookies('height_map.png', globalValues.heightMap, globalValues.distanceToLaser)  # найти положения объектов на столе
    print('Положения печенек найдены')
    gcodeInstructions = gcode_generator(path, cookies, pathToPly)
    print('Инструкции сгенерированы')

    # записать инструкции в текстовый файл
    writeGcode(gcodeInstructions)
    print('Gcode сгенерирован')
