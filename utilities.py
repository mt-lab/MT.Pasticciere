from typing import List, Union
from itertools import tee
from functools import reduce

import numpy as np
from numpy import cos, sin, arctan, sqrt, floor, tan, arccos
from ezdxf.math.vector import Vector
from globalValues import focal as global_focal
from globalValues import pxl_size as global_pxl_size
from globalValues import camera_height as global_camera_height

""" Some tools for convenience """

# TODO: рефактор и комментарии

X, Y, Z = 0, 1, 2


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def closed(iterable):
    """ ABCD -> A, B, C, D, A """
    return [item for item in iterable] + [iterable[0]]


def diap(start, end, step=1):
    """ (s, e) -> (s, m1), (m1, m2) .. (m_n, e) """
    x, y = round(start[X], 3), round(start[Y], 3)
    yield (x, y)
    step_x, step_y = 0, 0
    if end[X] - start[X] == 0 and end[Y] - start[Y] == 0:  # если на вход пришла точка
        yield (x, y)  # вернуть начальную точку (она же конечная)
    elif end[X] - start[X] == 0:  # если линия вертикальная
        step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y])
    elif end[Y] - start[Y] == 0:  # если линия горизонтальная
        step_x = step * (end[X] - start[X]) / abs(end[X] - start[X])
    else:  # любая другая линия
        # step_[x, y] = step * (sign determiner) * abs(share of step)
        step_x = step * (end[X] - start[X]) / abs(end[X] - start[X]) * abs(cos(
            arctan((end[Y] - start[Y]) / (end[X] - start[X]))))
        step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y]) * abs(
            sin(arctan((end[Y] - start[Y]) / (end[X] - start[X]))))
        step_x = round(step_x, 3)
        step_y = round(step_y, 3)
    d = distance(start, end)
    number_of_slices = int(floor(d / step))
    for i in range(number_of_slices):
        x += step_x
        y += step_y
        yield (x, y)
    if round(d, 3) > number_of_slices * step:
        # вернуть конечную точку, если мы в неё не попали при нарезке
        x, y = round(end[X], 3), round(end[Y], 3)
        yield (x, y)


def avg(*arg):
    return reduce(lambda a, b: a + b, arg) / len(arg)


def generate_ordered_numbers(number: int = 0) -> int:
    """ n -> n, n+1, n+2 ... """
    while True:
        yield number
        number += 1


def distance(p1, p2=(.0, .0, .0), simple=False) -> float:
    # TODO: вынести в elements
    """
        Calculate distance between 2 points either 2D or 3D
        :param list of float p1: точка от которой считается расстояние
        :param list of float p2: точка до которой считается расстояние
        :param bool simple: if True то оценочный расчёт расстояния без использования sqrt
    """
    p1 = list(p1)
    p2 = list(p2)
    if len(p1) < 3:
        p1.append(0)
    if len(p2) < 3:
        p2.append(0)
    if simple:
        return abs(p1[X] - p2[X]) + abs(p1[Y] - p2[Y]) + abs(p1[Z] - p2[Z])
    return sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)


def line_side(m: Vector, p1: Vector = (0, 0, 0), p2: Vector = (1, 1, 0)) -> Union['-1', '0', '1']:
    """
    check to which side of the line (p1,p2) the point m is
    :param m: point to check
    :param p1: first point of the line
    :param p2: second point of the line
    :return: -1 if on the left side, 0 on line, 1 on the right side
    """
    p1, p2 = (p1, p2) if p1[Y] > p2[Y] else (p2, p1)
    pos = np.sign((p2[X] - p1[X]) * (m[Y] - p1[Y]) - (p2[Y] - p1[Y]) * (m[X] - p1[X]))
    return pos


def triangle_area(a, b, c):
    # TODO: вынести в elements
    area = (a[X] * (b[Y] - c[Y]) + b[X] * (c[Y] - a[Y]) + c[X] * (a[Y] - b[Y])) / 2
    return abs(area)


def polygon_area(*args: List[List[float]]):
    area = 0
    for v1, v2 in pairwise(closed(args)):
        area += (v1[X] * v2[Y] - v1[Y] * v2[X]) / 2
    return abs(area)


def inside_polygon(p, *args: List[List[float]]):
    p = [round(coord, 2) for coord in p]
    boundary_area = round(polygon_area(*args))
    partial_area = 0
    for v1, v2 in pairwise(closed(args)):
        partial_area += round(triangle_area(p, v1, v2))
    if boundary_area - partial_area > boundary_area * 0.01:
        return False
    return True


def inside_triangle(p, a, b, c):
    # TODO: вынести в elements
    total_area = round(triangle_area(a, b, c), 3)
    area_pab = triangle_area(p, a, b)
    area_pbc = triangle_area(p, b, c)
    area_pca = triangle_area(p, c, a)
    partial_area = round(area_pab + area_pbc + area_pca, 3)
    if partial_area == total_area:
        return True
    return False


def height_by_trigon(p=(0, 0), a=(0, 0, 0), b=(0, 0, 0), c=(0, 0, 0)):
    # TODO: вынести в elements
    axy = np.asarray(a)
    bxy = np.asarray(b)
    cxy = np.asarray(c)
    pxy = np.r_[p, 1]
    axy[Z] = 1
    bxy[Z] = 1
    cxy[Z] = 1
    area = triangle_area(axy, bxy, cxy)
    area1 = triangle_area(pxy, axy, bxy)
    area2 = triangle_area(pxy, bxy, cxy)
    area3 = triangle_area(pxy, cxy, axy)
    height = c[Z] * area1 / area + a[Z] * area2 / area + b[Z] * area3 / area
    return height


def apprx_point_height(point: Vector, height_map: np.ndarray) -> float:
    # find closest point in height_map(ndarray)
    # determine if given point above or below closest, take corresponding upper/lower point in map
    # determine side on which given point is relative to 2 points from map
    # take corresponding left or right points
    # check if point inside the triangle formed by founded 3 points
    # calculate height for a point
    if not inside_polygon(point, height_map[0, 0, :2], height_map[0, -1, :2], height_map[-1, -1, :2],
                          height_map[-1, 0, :2]):
        print(f'point {point} not in the area')
        return 0
    idx_first = np.unravel_index(np.sum(np.abs(height_map[:, :, :2] - point[:2]), axis=2).argmin(),
                                 height_map.shape[:2])
    first = Vector(height_map[idx_first])
    above = point[Y] > first[Y]
    idx_second = (idx_first[X], idx_first[Y] + 1) if above else (idx_first[X], idx_first[Y] - 1)
    try:
        second = Vector(height_map[idx_second])
    except IndexError:
        return first.z
    first2second = first.distance(second)
    side = line_side(point, first, second)
    if side == 0:
        height = first.lerp(second, point.distance(first) / first2second).z
        # height = point.distance(first)/first2second * first.z + point.distance(second)/first2second * second.z
        return height
    else:
        idx_third = (idx_first[X] + int(side), idx_first[Y])
    try:
        third = height_map[idx_third]
    except IndexError:
        return first.z
    if inside_triangle(point, first, second, third):
        height = height_by_trigon(point, first, second, third)
        return height
    return first.z


def read_point_cloud(path):
    """ Read PLY point cloud into numpy array, also split it for xy and z coordinates """
    pcd = []
    with open(path) as cld:
        ply = cld.readlines()
    ply = ply[8:]
    for row in ply:
        pcd.append([float(x) for x in row.split()])
    pcd = np.asarray(pcd)
    pcd_xy, pcd_z = np.split(pcd, [Z], axis=1)
    return pcd, pcd_xy, pcd_z


def find_point_in_cloud(point, pcd_xy, pcd_z):
    """ Find corresponding Z coordinate for a given point in given point cloud """
    point = list(point)[:2]
    # closest_points = sorted(pcd, key=lambda p:distance(p, point[:2], simple=True))[:3]
    # point[X] += offset[X] #-50
    # point[Y] += offset[Y] #120
    # z = apprx_point_height(point, closest_points)
    z = pcd_z[np.sum(np.abs(pcd_xy - point), axis=1).argmin()][0]
    # point.append(z if z else 0)
    return z if z else 0


def find_angle_of_view(view_range: int = 640,
                       focal: float = global_focal,
                       pxl_size: float = global_pxl_size) -> float:
    """

    :param view_range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxl_size: размер пикселя на матрице
    :return: угол в радианах
    """
    return 2 * arctan(view_range * pxl_size / 2 / focal)


def find_camera_angle(view_width: float,
                      frame_width: int = 640,
                      camera_height: float = global_camera_height,
                      focal: float = global_focal,
                      pxl_size: float = global_pxl_size) -> float:
    """

    :param view_width: ширина обзора по центру кадра в мм
    :param frame_width: ширина кадра в пикселях
    :param camera_height: высота камеры над поверхностью в мм
    :param focal: фокусное расстояние линзы
    :param pxl_size: размер пикселя на матрице в мм
    :return: угол наклона камеры в радианах
    """
    view_angle = find_angle_of_view(frame_width, focal, pxl_size)
    cos_camera_angle = 2 * camera_height / view_width * tan(view_angle / 2)
    camera_angle = arccos(cos_camera_angle)
    return camera_angle


def save_height_map(height_map: np.ndarray, filename='heightMap.txt'):
    with open(filename, 'w') as outfile:
        outfile.write('{0}\n'.format(height_map.shape))
        outfile.write('# Data starts here\n')
        for dimension in height_map:
            np.savetxt(outfile, dimension, fmt='%-7.3f')
            outfile.write('# New dimension\n')


def generate_ply(points_array, filename='cloud.ply'):
    """
    Генерирует файл облака точек

    :param points_array - массив точек с координатами
    :param filename - имя файла для записи, по умолчанию cloud.ply
    :return: None
    """
    print('Generating point cloud...')
    ply = []
    for count, point in enumerate(points_array, 1):
        ply.append(f'{point[X]:.3f} {point[Y]:.3f} {point[Z]:.3f}\n')
    with open(filename, 'w+') as cloud:
        cloud.write("ply\n"
                    "format ascii 1.0\n"
                    f"element vertex {len(ply)}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "end_header\n")
        for count, point in enumerate(ply, 1):
            cloud.write(point)
    print(f'{len(ply):{6}} points recorded')
