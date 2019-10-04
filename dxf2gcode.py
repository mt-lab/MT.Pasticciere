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
        dwg.add_z(height_map)
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
                gcode += linear_move(X=contour.first_point.x, Y=contour.first_point.y, Z=Z_up)
                gcode += move_Z(contour.first_point.z + z_offset)
                gcode += linear_move('G1', F=kwargs.get('F1'))
                last_point = contour.first_point
                for element_index, element in enumerate(contour.elements, 1):
                    gcode += gcode_comment(f'        {element_index:3d} element in contour')
                    for point in element.get_points()[1:]:
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


def test_gcode(path_to_dxf, d_e=0.041, *args, **kwargs):
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
    dxf = ez.readfile(path_to_dxf)
    dwg = Drawing(dxf)
    dwg.slice()
    preGcode = ['G0 E1 F300', 'G92 E0', 'G0 F3000']
    gcode_generator(dwg, None, None, d_e, k, p_0, p_1, p_2, preGcode=preGcode, **kwargs)


def dxf2gcode(path_to_dxf=globalValues.DXF_PATH, *args, **kwargs):
    """
    Функция обработки dxf в Gcode

    :param path_to_dxf: путь до рисунка
    :return: None
    """

    settings_values = globalValues.get_settings_values(**globalValues.settings_sections)

    # прочесть dxf
    dxf = ez.readfile(path_to_dxf)
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
