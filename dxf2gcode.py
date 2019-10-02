"""
dxf2gcode.py
Author: bedlamzd of MT.lab

Чтение .dxf файла, слайсер этого dxf, перевод в трёхмерное пространство на рельеф объектов,
генерация gcode в соответствующий файл
"""
import ezdxf as ez
from typing import Union, Optional
from gcodeGen import *
from elements import *
from utilities import read_point_cloud
import globalValues
from scanner import find_cookies
from cookie import Cookie

# TODO: написать логи

Z_max = 30
Z_up = Z_max + 0.2  # later should be cloud Z max + few mm сейчас это глобальный максимум печати принтера по Z


def gcode_generator(dwg: Drawing, cookies: Optional[List[Cookie]] = None,
                    height_map: np.ndarray = globalValues.height_map,
                    extrusion_coefficient=0.041, extrusion_multiplex=1, p0=0.05, p1=0.05, p2=0.05, z_offset=0.2,
                    **kwargs):
    E = 0
    gcode = Gcode()
    gcode += home()
    gcode += move_Z(Z_max, kwargs.get('F0'))
    for count, cookie in enumerate(cookies, 1):
        Z_up = cookie.maxHeight + 5 if cookie.maxHeight + 5 <= Z_max else Z_max
        gcode += gcode_comment(f'{count:3d} cookie')
        dwg.center = cookie.center
        dwg.rotation = cookie.rotation
        dwg.addZ(height_map)  # TODO: ПЕРЕПИСАТЬ ЧАСТЬ С PLY
        for layer_index, layer in enumerate(sorted(dwg.layers.values(), key=lambda x: x.priority)):
            gcode += gcode_comment(f'{layer_index:3d} layer: {layer.name} in drawing')
            if layer.name == 'Contour':  # or layer.priority == 0:
                gcode += gcode_comment(f'    layer skiped')
                print(f'Layer skipped. Name: {layer.name}; Priority: {layer.priority}')
                continue
            for contour_index, contour in enumerate(layer.contours):
                printed_length = 0
                dE = extrusion_coefficient * extrusion_multiplex
                gcode += gcode_comment(f'    {contour_index:3d} contour in layer')
                gcode += linear_move(X=contour.firstPoint.x, Y=contour.firstPoint.y, Z=Z_up)
                gcode += move_Z(contour.firstPoint.z + z_offset)
                gcode += linear_move('G1', F=kwargs.get('F1'))
                last_point = contour.firstPoint
                for element_index, element in enumerate(contour.elements, 1):
                    gcode += gcode_comment(f'        {element_index:3d} element in contour')
                    for point in element.getPoints()[1:]:
                        dL = point.distance(last_point)
                        E += round(dE * dL, 3)
                        gcode += linear_move('G1', X=point.x, Y=point.y, Z=point.z + z_offset, E=E)
                        printed_length += dL
                        last_point = point
                        printed_percent = printed_length / contour.length
                        if printed_percent < p0:
                            dE = extrusion_coefficient * extrusion_multiplex
                        elif printed_percent < p1:
                            dE = 0
                        elif printed_percent < p2:
                            dE = extrusion_coefficient
                        else:
                            dE = 0
                gcode += linear_move(F=kwargs.get('F0'))
                gcode += move_Z(Z_up)
    gcode += move_Z(Z_max)
    gcode += home()
    print('Команды сгенерированы')
    gcode.save()


def testGcode(pathToDxf, dE=0.041, *args, **kwargs):
    if 'dynamic_extrusion' in args:
        p_0 = kwargs.get('p0', 0.05)
        p_1 = kwargs.get('p1', 0.15)
        p_2 = kwargs.get('p2', 0.9)
        k = kwargs.get('extrusion_multiplex', p_1 / p_0)
    else:
        k = 1
        p_0 = 0
        p_1 = 0
        p_2 = 0
    dxf = ez.readfile(pathToDxf)
    dwg = Drawing(dxf)
    dwg.slice()
    preGcode = ['G0 E1 F300', 'G92 E0', 'G0 F3000']
    gcode_generator(dwg, None, None, dE, k, p_0, p_1, p_2, preGcode=preGcode, **kwargs)


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
    """
    # slice dxf
    for element in path:
        element.slice(step)


def adjustPath(path, offset=(0, 0), pathToPly=globalValues.PCD_PATH):
    pcd, pcd_xy, pcd_z = read_point_cloud(pathToPly)
    # add volume to dxf, also add offset
    for element in path:
        element.setOffset(offset)
        element.addZ(pcd_xy, pcd_z, pcd)


def writeGcode(gcodeInstructions, filename='cookie.gcode'):
    """
    Генерирует текстовый файл с инструкциями для принтера

    :param gcodeInstructions: массив строк с командами для принтера
    :param filename: имя файла для записи команд
    """
    with open(filename, 'w+') as gcode:
        for line in gcodeInstructions:
            gcode.write(line + '\n')
    print('Команды для принтера сохранены.')


def dxf2gcode(pathToDxf=globalValues.DXF_PATH, pathToPly=globalValues.PCD_PATH):
    """
    Функция обработки dxf в Gcode

    :param pathToDxf: путь до рисунка
    :param pathToPly: путь до облака точек
    :return: None
    """

    settings_values = globalValues.get_settings_values(**globalValues.settings_sections)

    # прочесть dxf
    dxf = ez.readfile(pathToDxf)
    dwg = Drawing(dxf)
    dwg.slice(settings_values.get('slice_step'))
    print(dwg)
    if globalValues.cookies is None:
        cookies, _ = find_cookies('height_map.png', globalValues.height_map)  # найти положения объектов на столе
        if len(cookies) != 0:
            globalValues.cookies = cookies
            print(f'Объектов найдено: {len(cookies):{3}}')
            print('#############################################')
            for i, cookie in enumerate(cookies, 1):
                print(f'Объект №{i:3d}')
                print('#############################################')
                print(cookie)
                print('#############################################')
            print()
    else:
        cookies = globalValues.cookies
    gcode_generator(dwg, cookies, globalValues.height_map)
