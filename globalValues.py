from math import radians
from numpy import loadtxt, float32
from os.path import isfile
import configparser

config_path = 'settings.ini'


def string2list(sep=' ', f=None):
    def _string2list(string: str, s=' ', func=None):
        return string.split(s) if func is None else [func(x) for x in string.split(s)]

    def wrapper(string):
        return _string2list(string, sep, f)

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
            print('Читаю карту высот')
            shape = infile.readline()
            shape = shape[1:-2]
            shape = tuple(i for i in map(int, shape.split(', ')))
            _height_map = loadtxt(filename, skiprows=1, dtype=float32)
            _height_map = _height_map.reshape(shape)
            return _height_map
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


height_map = None
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
    'reverse': ('Scanner', lambda str: True if str == 'True' else False),
    'mirrored': ('Scanner', lambda str: True if str == 'True' else False),
    'extraction_mode': ('Scanner', None),
    'avg_time': ('Scanner', float),
    'laser_angle_tol': ('Scanner', float),
    'laser_pos_tol': ('Scanner', float),
    'accuracy': ('GCoder', float),
    'z_offset': ('GCoder', float),
    'extrusion_coefficient': ('GCoder', float),
    'retract_amount': ('GCoder', float),
    'p0': ('GCoder', float),
    'p1': ('GCoder', float),
    'p2': ('GCoder', float),
    'slice_step': ('GCoder', float),
}
