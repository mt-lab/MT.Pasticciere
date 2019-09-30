from math import radians
from numpy import loadtxt, cos, tan, float16
from os.path import isfile
import configparser
from typing import List


def update_config_values():
    global accuracy
    global slice_step
    global VID_PATH
    global DXF_PATH
    global PCD_PATH
    global z_offset
    global extrusion_coefficient
    global retract_amount
    global p0
    global p1
    global p2
    global table_width  # type: float
    global table_height  # type: float
    global table_length  # type: float
    global X0
    global Y0
    global Z0
    global pxl_size  # type: float
    global focal  # type: float
    global camera_angle  # type: float
    global cos_alpha  # type: float
    global tan_alpha  # type: float
    global camera_height  # type: float
    global camera_shift  # type: float
    global hsv_upper_bound  # type: List[int]
    global hsv_lower_bound  # type: List[int]
    global mark_center  # type: List[int]
    accuracy = get_accuracy()
    slice_step = get_slice_step()
    VID_PATH = get_vid_path()
    DXF_PATH = get_dxf_path()
    PCD_PATH = get_pcd_path()
    z_offset = get_z_offset()
    extrusion_coefficient = get_extrusion_coefficient()
    retract_amount = get_retract_amount()
    p0 = get_p0()
    p1 = get_p1()
    p2 = get_p2()
    table_width = get_table_width()
    table_height = get_table_height()
    table_length = get_table_length()
    X0 = get_X0()
    Y0 = get_Y0()
    Z0 = get_Z0()
    pxl_size = get_pxl_size()
    focal = get_focal()
    camera_angle = get_camera_angle()
    cos_alpha = cos(camera_angle)
    tan_alpha = tan(camera_angle)
    camera_height = get_camera_height()
    camera_shift = get_camera_shift()
    hsv_upper_bound = get_hsv_upper_bound()
    hsv_lower_bound = get_hsv_lower_bound()
    mark_center = get_mark_center()


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


# параметры рабочей области
def get_table_width() -> float:
    return float(get_setting(config_path, 'Table', 'width'))


def get_table_length() -> float:
    return float(get_setting(config_path, 'Table', 'length'))


def get_table_height() -> float:
    return float(get_setting(config_path, 'Table', 'height'))


def get_X0() -> float:
    return float(get_setting(config_path, 'Table', 'X0'))


def get_Y0() -> float:
    return float(get_setting(config_path, 'Table', 'Y0'))


def get_Z0() -> float:
    return float(get_setting(config_path, 'Table', 'Z0'))


# характеристики камеры
def get_focal() -> float:
    return float(get_setting(config_path, 'Camera', 'focalLength'))


def get_pxl_size() -> float:
    return float(get_setting(config_path, 'Camera', 'pixelSize'))


def get_camera_angle() -> float:
    return radians(float(get_setting(config_path, 'Camera', 'angle')))


def get_camera_height() -> float:
    return float(get_setting(config_path, 'Camera', 'cameraHeight'))


def get_camera_shift() -> float:
    return float(get_setting(config_path, 'Camera', 'cameraShift'))


# параметры фильтра для сканера (предположительно свои для каждого принтера)
def get_hsv_lower_bound() -> List[int]:
    hsv_lower_bound_string = get_setting(config_path, 'Scanner', 'hsv_min')
    return [int(hsv_lower_bound_string.split(', ')[i]) for i in range(3)]


def get_hsv_upper_bound() -> List[int]:
    hsv_upper_bound_string = get_setting(config_path, 'Scanner', 'hsv_max')
    return [int(hsv_upper_bound_string.split(', ')[i]) for i in range(3)]


def get_mark_picture() -> str:
    return get_setting(config_path, 'Scanner', 'markpic')  # path to mark picture


def get_mark_center() -> List[int]:
    mark_center_string = get_setting(config_path, 'Scanner', 'markcenter')
    return [int(mark_center_string.split(', ')[i]) for i in range(2)]  # where mark should be in the frame


# окрестность в пределах которой точки считаются совпадающими
def get_accuracy() -> float:
    return float(get_setting(config_path, 'GCoder', 'accuracy'))


# шаг нарезки рисунка
def get_slice_step() -> float:
    return float(get_setting(config_path, 'GCoder', 'slice_step'))


def get_z_offset() -> float:
    return float(get_setting(config_path, 'GCoder', 'zoffset'))


def get_extrusion_coefficient() -> float:
    return float(get_setting(config_path, 'GCoder', 'extrusionCoefficient'))


def get_retract_amount() -> float:
    return float(get_setting(config_path, 'GCoder', 'retractamount'))


def get_p0() -> float:
    return float(get_setting(config_path, 'GCoder', 'p0'))


def get_p1() -> float:
    return float(get_setting(config_path, 'GCoder', 'p1'))


def get_p2() -> float:
    return float(get_setting(config_path, 'GCoder', 'p2'))


# пути по умолчанию для dxf, облака точек и видео соответственно
def get_dxf_path() -> str:
    return get_setting(config_path, 'GCoder', 'dxfpath')


def get_pcd_path() -> str:
    return get_setting(config_path, 'GCoder', 'pointcloudpath')


def get_vid_path() -> str:
    return get_setting(config_path, 'GCoder', 'videoforpointcloud')


height_map = read_height_map()
cookies = None

accuracy = None
slice_step = None
VID_PATH = None
DXF_PATH = None
PCD_PATH = None
z_offset = None
extrusion_coefficient = None
retract_amount = None
p0 = None
p1 = None
p2 = None
table_width = None
table_height = None
table_length = None
X0 = None
Y0 = None
Z0 = None
pxl_size = None
focal = None
camera_angle = None
cos_alpha = None
tan_alpha = None
camera_height = None
camera_shift = None
hsv_upper_bound = None
hsv_lower_bound = None
mark_center = None

update_config_values()
