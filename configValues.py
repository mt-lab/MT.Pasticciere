"""
scanner.py
Author: bedlamzd of MT.lab

Файл для парса конфига
"""

import configparser
from math import radians


def get_config(path):
    """
    Выбираем файл настроек
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


def update_setting(path, section, setting, value):
    """
    Обновляем параметр в настройках
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)


def get_setting(path, section, setting):
    """
    Выводим значение из настроек
    """
    config = get_config(path)
    value = config.get(section, setting)
    msg = "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value
    )
    print(msg)
    return value


config_path = 'settings.ini'


# параметры рабочей области
def table_width():
    return float(get_setting(config_path, 'Table', 'width'))


def table_length():
    return float(get_setting(config_path, 'Table', 'length'))


def table_height():
    return float(get_setting(config_path, 'Table', 'height'))


def X0():
    return float(get_setting(config_path, 'Table', 'X0'))


def Y0():
    return float(get_setting(config_path, 'Table', 'Y0'))


def Z0():
    return float(get_setting(config_path, 'Table', 'Z0'))


# характеристики камеры
focal = float(get_setting(config_path, 'Camera', 'focalLength'))
pxlSize = float(get_setting(config_path, 'Camera', 'pixelSize'))
cameraAngle = radians(float(get_setting(config_path, 'Camera', 'angle')))
cameraHeight = float(get_setting(config_path, 'Camera', 'cameraHeight'))
cameraShift = float(get_setting(config_path, 'Camera', 'cameraShift'))

# параметры фильтра для сканера (предположительно свои для каждого принтера)
hsvLowerBoundString = get_setting(config_path, 'Scanner', 'hsv_min')  # строка
hsvUpperBoundString = get_setting(config_path, 'Scanner', 'hsv_max')  # строка
hsvLowerBound = [int(hsvLowerBoundString.split(', ')[i]) for i in range(3)]  # список
hsvUpperBound = [int(hsvUpperBoundString.split(', ')[i]) for i in range(3)]  # список
markPicture = get_setting(config_path, 'Scanner', 'markpic')  # path to mark picture
markCenterString = get_setting(config_path, 'Scanner', 'markcenter')
markCenter = [int(markCenterString.split(', ')[i]) for i in range(2)]  # where mark shoul be in the frame

# окрестность в пределах которой точки считаются совпадающими
accuracy = float(get_setting(config_path, 'GCoder', 'accuracy'))

# шаг нарезки рисунка
sliceStep = float(get_setting(config_path, 'GCoder', 'slice_step'))
zOffset = float(get_setting(config_path, 'GCoder', 'zoffset'))
extrusionCoefficient = float(get_setting(config_path, 'GCoder', 'extrusionCoefficient'))
retractAmount = float(get_setting(config_path, 'GCoder', 'retractamount'))
p0 = float(get_setting(config_path, 'GCoder', 'p0'))
p1 = float(get_setting(config_path, 'GCoder', 'p1'))
p2 = float(get_setting(config_path, 'GCoder', 'p2'))

# пути по умолчанию для dxf, облака точек и видео соответственно
DXF_PATH = get_setting(config_path, 'GCoder', 'dxfpath')
PCD_PATH = get_setting(config_path, 'GCoder', 'pointcloudpath')
VID_PATH = get_setting(config_path, 'GCoder', 'videoforpointcloud')
