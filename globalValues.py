from math import radians
from numpy import loadtxt, cos, tan, float16
from os.path import isfile
import configparser
from typing import List


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


config_path = 'settings.ini'
config = get_config(config_path)


def update_config(path=config_path):
    global config
    global cos_alpha
    global tan_alpha
    config = get_config(path)
    cos_alpha = get_cos_alpha()
    tan_alpha = get_tan_alpha()


height_map = read_height_map()
cookies = None


def get_accuracy(): return config.getfloat('GCoder', 'accuracy')


def get_z_offset(): return config.getfloat('GCoder', 'zoffset')


def get_extrusion_coefficient(): return config.getfloat('GCoder', 'extrusioncoefficient')


def get_retract_amount(): return config.getfloat('GCoder', 'retractamount')


def get_p0(): return config.getfloat('GCoder', 'p0')


def get_p1(): return config.getfloat('GCoder', 'p1')


def get_p2(): return config.getfloat('GCoder', 'p2')


def get_slice_step(): return config.getfloat('GCoder', 'slice_step')


def get_VID_PATH(): return config.get('GCoder', 'videoforpointcloud')


def get_PCD_PATH(): return config.get('GCoder', 'pointcloudpath')


def get_DXF_PATH(): return config.get('GCoder', 'dxfpath')


def get_table_width(): return config.getfloat('Table', 'width')


def get_table_height(): return config.getfloat('Table', 'height')


def get_table_length(): return config.getfloat('Table', 'length')


def get_X0(): return config.getfloat('Table', 'x0')


def get_Y0(): return config.getfloat('Table', 'y0')


def get_Z0(): return config.getfloat('Table', 'z0')


def get_pxl_size(): return config.getfloat('Camera', 'pixelsize')


def get_focal(): return config.getfloat('Camera', 'focallength')


def get_camera_angle(): return config.getfloat('Camera', 'angle')


def get_cos_alpha(): return cos(get_camera_angle())


def get_tan_alpha(): return tan(get_camera_angle())


def get_camera_height(): return config.getfloat('Camera', 'cameraheight')


def get_camera_shift(): return config.getfloat('Camera', 'camerashift')


def get_hsv_upper_bound(): return [int(config.get('Scanner', 'hsv_max').split(', ')[i]) for i in range(3)]


def get_hsv_lower_bound(): return [int(config.get('Scanner', 'hsv_min').split(', ')[i]) for i in range(3)]


def get_mark_pic_path(): return config.getfloat('Scanner', 'markpic')


def get_mark_center(): return [int(config.get('Scanner', 'markcenter').split(', ')[i]) for i in range(2)]

cos_alpha = get_cos_alpha()
tan_alpha = get_tan_alpha()
