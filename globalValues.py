from math import radians
from numpy import loadtxt, cos, tan, float16
from os.path import isfile
import configparser
from typing import List

config_path = 'settings.ini'


def string2list(delim=' ', func=None):
    def _string2list(string: str, delim=' ', func=None):
        return string.split(delim) if func is None else [func(x) for x in string.split(delim)]

    def wrapper(string):
        return _string2list(string, delim, func)

    return wrapper


def get_settings_values(path=config_path, **kwargs):
    settings_values = {}
    for setting, (section, formatting) in kwargs.items():
        value = get_setting(path, section, setting)
        settings_values[setting] = formatting(value) if formatting is not None else value
    return settings_values


def read_height_map(filename='height_map.txt'):
    if isfile(filename):
        with open(filename, 'r') as infile:
            shape = infile.readline()
            shape = shape[1:-2]
            shape = [int(shape.split(', ')[i]) for i in range(3)]
            height_map = loadtxt(filename, skiprows=1, dtype=float16)
            height_map = height_map.reshape(shape)
            return height_map
    return None


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


def get_setting(path, section, setting) -> str:
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


height_map = read_height_map()
cookies = None

settings_sections = {
    'table_width': ('Table', float),
    'table_length': ('Table', float),
    'table_height': ('Table', float),
    'x0': ('Table', float),
    'y0': ('Table', float),
    'z0': ('Table', float),
    'pixel_size': ('Camera', float),
    'focal_length': ('Camera', float),
    'camera_angle': ('Camera', lambda x: radians(float(x))),
    'camera_height': ('Camera', float),
    'camera_shift': ('Camera', float),
    'distance_camera2laser': ('Camera', float),
    'hsv_lower_bound': ('Scanner', string2list(', ', float)),
    'hsv_upper_bound': ('Scanner', string2list(', ', float)),
    'mark_pic_path': ('Scanner', None),
    'mark_center': ('Scanner', string2list(', ', float)),
    'accuracy': ('GCoder', float),
    'z_offset': ('GCoder', float),
    'extrusion_coefficient': ('GCoder', float),
    'retract_amount': ('GCoder', float),
    'p0': ('GCoder', float),
    'p1': ('GCoder', float),
    'p2': ('GCoder', float),
    'slice_step': ('GCoder', float),
}

VID_PATH = get_setting(config_path, 'GCoder', 'videoforpointcloud')
PCD_PATH = get_setting(config_path, 'GCoder', 'pointcloudpath')
DXF_PATH = get_setting(config_path, 'GCoder', 'dxfpath')
