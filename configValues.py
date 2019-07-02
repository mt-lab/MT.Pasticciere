"""
scanner.py
Author: bedlamzd of MT.lab

Файл для парса конфига
"""

import configparser


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
tableWidth = float(get_setting(config_path, 'Table','width'))
tableLength = float(get_setting(config_path, 'Table','length'))
tableHeight = float(get_setting(config_path, 'Table','height'))
X0 = float(get_setting(config_path, 'Table','X0'))
Y0 = float(get_setting(config_path, 'Table','Y0'))
Z0 = float(get_setting(config_path, 'Table','Z0'))

# характеристики камеры
focal = float(get_setting(config_path, 'Camera','pixelSize'))
pxlSize = float(get_setting(config_path, 'Camera','focalLength'))
cameraAngl = float(get_setting(config_path, 'Camera','angle'))
distanceToLaser = float(get_setting(config_path, 'Camera','distanceToLaser'))

# параметры фильтра для сканера (предположительно свои для каждого принтера)
hsvLowerBoundString = get_setting(config_path, 'Scanner', 'hsv_min')[1:-1]  # строка
hsvUpperBoundString = get_setting(config_path, 'Scanner', 'hsv_max')[1:-1]  # строка
hsvLowerBound = [int(hsvLowerBoundString.split(', ')[i]) for i in range(len(hsvLowerBoundString.split(', ')))]  # список
hsvUpperBound = [int(hsvUpperBoundString.split(', ')[i]) for i in range(len(hsvUpperBoundString.split(', ')))]  # список

# окрестность в пределах которой точки считаются совпадающими
accuracy = float(get_setting(config_path, 'GCoder', 'accuracy'))

# шаг нарезки рисунка
sliceStep = float(get_setting(config_path, 'GCoder', 'slice_step'))

# пути по умолчанию для dxf, облака точек и видео соответственно
DXF_PATH = get_setting(config_path, 'GCoder', 'dxfpath')
PCD_PATH = get_setting(config_path, 'GCoder', 'pointcloudpath')
VID_PATH = get_setting(config_path, 'GCoder', 'videoforpointcloud')
