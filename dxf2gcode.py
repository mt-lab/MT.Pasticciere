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
from globalValues import cookies as global_cookies
from globalValues import *
from scanner import find_cookies
from cookie import Cookie

# TODO: написать логи

Z_max = 30
Z_up = Z_max + z_offset  # later should be cloud Z max + few mm сейчас это глобальный максимум печати принтера по Z


def ggen(dwg: Drawing, cookies: Optional[List[Cookie]] = None, height_map: np.ndarray = height_map,
         preGcode: Optional[List[str]] = None,
         postGcode: Optional[List[str]] = None, *args, **kwargs):
    p = {'ke': extrusion_coefficient, 'em': 1, 'p_0': p0, 'p_1': p1, 'p_2': p2, 'F0': None, 'F1': None}
    E = 0
    for key in p:
        value = kwargs.get(key)
        p[key] = value if value is not None else p[key]
    gcode = Gcode()
    gcode += home()
    gcode += move_Z(Z_max, p['F0'])
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
                dE = p['ke'] * p['em']
                gcode += gcode_comment(f'    {contour_index:3d} contour in layer')
                gcode += linear_move(X=contour.firstPoint.x, Y=contour.firstPoint.y, Z=Z_up)
                gcode += move_Z(contour.firstPoint.z + z_offset)
                gcode += linear_move('G1', F=p['F1'])
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
                        if printed_percent < p['p_0']:
                            dE = p['em'] * p['ke']
                        elif printed_percent < p['p_1']:
                            dE = 0
                        elif printed_percent < p['p_2']:
                            dE = p['ke']
                        else:
                            dE = 0
                gcode += linear_move(F=p['F0'])
                gcode += move_Z(Z_up)
    gcode += move_Z(Z_max)
    gcode += home()
    print('Команды сгенерированы')
    gcode.save()


def gcodeGenerator(dwg, cookies: Optional[List[Cookie]] = None, preGcode: Optional[List[str]] = None,
                   postGcode: Optional[List[str]] = None, ke=extrusion_coefficient, k=1, p_0=p0(), p_1=p1(), p_2=p2(),
                   *args) -> List[str]:
    """
    Генерирует gcode для печати рисунка dwg на печеньках cookies и возвращает список команд.
    :param Drawing dwg: рисунок для печати
    :param list of Cookie cookies: список печенек на которые наносится рисунок
    :param list preGcode: gcode для вставки перед сгенерированными командами
    :param list postGcode: gcode для вставки после сгенерированных команд
    :return list gcode: список команд для принтера
    """
    if cookies is None:
        cookies = [Cookie(center=args[0], maxHeight=args[1])]
        pcd, pcd_xy, pcd_z = None, None, None
        F0 = args[2]
        F1 = args[3]
    else:
        pcd, pcd_xy, pcd_z = read_point_cloud(None)
        args = [None, None, None, None]
        F0 = args[2]
        F1 = args[3]
    # проверка наличия пре-кода и пост-кода
    if preGcode is None:
        preGcode = []
    if postGcode is None:
        postGcode = []
    gcode = []
    E = 0  # начальное значение выдавливания глазури (положение мешалки)
    gcode.append('G28')  # домой
    gcode.append(f'G0 Z{Z_max} F3000')
    if F0:
        gcode.append(f'G0 F{F0}')
    gcode += preGcode
    # для каждой печеньки в списке
    for count, cookie in enumerate(cookies, 1):
        Z_up = cookie.maxHeight + 5 if cookie.maxHeight + 5 <= Z_max else Z_max
        gcode.append(f'; {count:3d} cookie')
        # подгонка рисунка для печенья
        dwg.center = cookie.center
        dwg.rotation = cookie.rotation
        dwg.addZ(pcd_xy, pcd_z, constantShift=args[1])
        for index, contour in enumerate(dwg.contours, 1):
            printed_length = 0
            dE = ke * k
            gcode.append(f';    {index:3d} contour in drawing')
            gcode.append(f'G0 X{contour.firstPoint[X]:3.3f} Y{contour.firstPoint[Y]:3.3f} Z{Z_up:3.3f}')
            gcode.append(f'G0 Z{contour.firstPoint[Z] + z_offset:3.3f}')
            gcode.append(f'G1 F{F1}')
            last_point = contour.firstPoint
            for idx, element in enumerate(contour.elements, 1):
                gcode.append(f';        {idx:3d} element in contour')
                for point in element.getSlicedPoints()[1:]:
                    dL = distance(last_point, point)
                    E += round(dE * dL, 3)
                    gcode.append(f'G1 X{point[X]:3.3f} Y{point[Y]:3.3f} Z{point[Z] + z_offset:3.3f} E{E:3.3f}')
                    printed_length += dL
                    last_point = point
                    if printed_length / contour.length < p_0:
                        dE = k * ke
                    elif printed_length / contour.length < p_1:
                        dE = 0
                    elif printed_length / contour.length < p_2:
                        dE = ke
                    else:
                        dE = 0
            gcode.append(f'G0 F{F0}')
            gcode.append(f'G0 Z{Z_up:3.3f}')
    gcode += postGcode
    gcode.append(f'G0 Z{Z_max:3.3f}')
    gcode.append('G28')
    print('Команды для принтера сгенерированы.')
    return gcode


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
                gcode.append(f'G0 Z{way[0][Z] + z_offset:3.3f}')
                last_point = way[0]  # обновить предыдущую точку
            for point in way[1:]:
                E += round(extrusion_coefficient * distance(last_point, point), 3)
                gcode.append(f'G1 X{point[X]:3.3f} Y{point[Y]:3.3f} Z{point[Z] + z_offset:3.3f} E{E:3.3f}')
                last_point = point
            E -= retract_amount
            last_point = way[-1]
        gcode.append(f'G0 Z{Z_up:3.3f}')
    gcode += postGcode
    gcode.append(f'G0 Z{Z_up:3.3f}')  # в конце поднять
    gcode.append('G28')  # и увести домой
    return gcode


def testGcode(pathToDxf, dE=extrusion_coefficient, F=300, height=0, center=(0, 0), retract=0, dynamicE=False, *args,
              **kwargs):
    if dynamicE:
        k = kwargs.get('k')
        p_0 = kwargs.get('p0')
        p_1 = kwargs.get('p1')
        p_2 = kwargs.get('p2')
        if p_0 is None:
            p_0 = p0
        if p_1 is None:
            p_1 = p1
        if p_2 is None:
            p_2 = p2
        if k is None:
            k = p_1 / p_0
    else:
        k = 1
        p_0 = 0
        p_1 = 0
        p_2 = 0
    dxf = ez.readfile(pathToDxf)
    dwg = Drawing(dxf)
    dwg.slice()
    preGcode = ['G0 E1 F300', 'G92 E0', 'G0 F3000']
    gcode = gcodeGenerator(dwg, None, None, preGcode, None, dE, k, p_0, p_1, p_2, center, height, 3000, F)
    writeGcode(gcode)


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


def adjustPath(path, offset=(0, 0), pathToPly=PCD_PATH):
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


def dxf2gcode(pathToDxf=DXF_PATH, pathToPly=PCD_PATH):
    """
    Функция обработки dxf в Gcode

    :param pathToDxf: путь до рисунка
    :param pathToPly: путь до облака точек
    :return: None
    """

    update_config()

    # прочесть dxf
    dxf = ez.readfile(pathToDxf)
    dwg = Drawing(dxf)
    dwg.slice(slice_step)
    print(dwg)
    if global_cookies is None:
        cookies, _ = find_cookies('height_map.png', height_map)  # найти положения объектов на столе
        if len(cookies) != 0:
            cookies = cookies
            print(f'Объектов найдено: {len(cookies):{3}}')
            print('#############################################')
            for i, cookie in enumerate(cookies, 1):
                print(f'Объект №{i:3d}')
                print('#############################################')
                print(cookie)
                print('#############################################')
            print()
    else:
        cookies = global_cookies
    # gcodeInstructions = gcodeGenerator(dwg, cookies, pathToPly)
    # writeGcode(gcodeInstructions)
    ggen(dwg, cookies, height_map)
