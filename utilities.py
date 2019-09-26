from typing import List, Optional, Union
from itertools import tee
from functools import reduce
from os.path import isfile
from configValues import PCD_PATH, focal, pxlSize, cameraHeight
import numpy as np
from numpy import cos, sin, arctan, sqrt, floor, tan, arccos
from ezdxf.math.vector import Vector, NULLVEC

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


def triangleArea(A, B, C):
    # TODO: вынести в elements
    area = (A[X] * (B[Y] - C[Y]) + B[X] * (C[Y] - A[Y]) + C[X] * (A[Y] - B[Y])) / 2
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
        partial_area += round(triangleArea(p, v1, v2))
    if boundary_area - partial_area > boundary_area * 0.01:
        return False
    return True

def insideTriangle(p, a, b, c):
    # TODO: вынести в elements
    S = round(triangleArea(a, b, c), 3)
    Spab = triangleArea(p, a, b)
    Spbc = triangleArea(p, b, c)
    Spca = triangleArea(p, c, a)
    partial_area = round(Spab + Spbc + Spca, 3)
    if partial_area == S:
        return True
    return False


def heightByTrigon(p=(0, 0), a=(0, 0, 0), b=(0, 0, 0), c=(0, 0, 0)):
    # TODO: вынести в elements
    axy = np.asarray(a)
    bxy = np.asarray(b)
    cxy = np.asarray(c)
    pxy = np.r_[p, 1]
    axy[Z] = 1
    bxy[Z] = 1
    cxy[Z] = 1
    S = triangleArea(axy, bxy, cxy)
    S1 = triangleArea(pxy, axy, bxy)
    S2 = triangleArea(pxy, bxy, cxy)
    S3 = triangleArea(pxy, cxy, axy)
    # TODO: add check on insideTriangle to escape unnecessary calcs
    height = c[Z] * S1 / S + a[Z] * S2 / S + b[Z] * S3 / S
    return height


def apprxPointHeight(point: Vector, height_map: np.ndarray, *arg):
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
    sub = height_map[:, :, :2] - point[:2]
    abs_sub = np.abs(sub)
    sum_abs = np.sum(abs_sub, axis=2)
    idx_first = sum_abs.argmin()
    idx_first = np.unravel_index(idx_first, height_map.shape[:2])
    # idx_first = np.unravel_index(np.sum(np.abs(height_map[:, :, :2] - point[:2]), axis=2).argmin(), height_map.shape)[:2]
    first = Vector(height_map[idx_first])
    above = point[Y] > first[Y]
    # TODO: fix out of bound
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
    if insideTriangle(point, first, second, third):
        height = heightByTrigon(point, first, second, third)
        return height
    return first.z


def lineFrom2points(p1=(0, 0), p2=(0, 0)):
    # TODO: вынести в elements
    k = (p2[X] - p1[X]) / (p2[Y] - p1[Y])
    b = p1[Y] - k * p1[X]
    return k, b


def perpendicular2line(p=(0, 0), k=1, b=0):
    # TODO: вынести в elements
    pk = 1 / k
    pb = p[X] / k + p[Y]
    return pk, pb


def crossectionOfLines(k1=1, b1=0, k2=1, b2=0):
    # TODO: вынести в elements
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return (x, y)


def readPointCloud(path=PCD_PATH):
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


def findPointInCloud(point, pcd_xy, pcd_z, pcd=None):
    """ Find corresponding Z coordinate for a given point in given point cloud """
    point = list(point)[:2]
    # closest_points = sorted(pcd, key=lambda p:distance(p, point[:2], simple=True))[:3]
    # point[X] += offset[X] #-50
    # point[Y] += offset[Y] #120
    # z = apprxPointHeight(point, closest_points)
    z = pcd_z[np.sum(np.abs(pcd_xy - point), axis=1).argmin()][0]
    # point.append(z if z else 0)
    return z if z else 0


def findAngleOfView(range: 'in pxls' = 640, focal: 'in mm' = focal, pxlSize: 'in mm' = pxlSize):
    """

    :param range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxlSize: размер пикселя на матрице
    :return: угол в радианах
    """
    return 2 * arctan(range * pxlSize / 2 / focal)


def findCameraAngle(viewWidth: 'in mm', frameWidth: 'in pxls' = 640, cameraHeight: 'in mm' = cameraHeight,
                    focal: 'in mm' = focal, pxlSize: 'in mm' = pxlSize):
    """

    :param viewWidth: ширина обзора по центру кадра в мм
    :param frameWidth: ширина кадра в пикселях
    :param cameraHeight: высота камеры над поверхностью в мм
    :param focal: фокусное расстояние линзы
    :param pxlSize: размер пикселя на матрице в мм
    :return: угол наклона камеры в радианах
    """
    viewAngle = findAngleOfView(frameWidth, focal, pxlSize)
    cos_cameraAngle = 2 * cameraHeight / viewWidth * tan(viewAngle / 2)
    cameraAngle = arccos(cos_cameraAngle)
    return cameraAngle


def saveHeightMap(heightMap: np.ndarray, filename='heightMap.txt'):
    with open(filename, 'w') as outfile:
        outfile.write('{0}\n'.format(heightMap.shape))
        outfile.write('# Data starts here\n')
        for dimension in heightMap:
            np.savetxt(outfile, dimension, fmt='%-7.3f')
            outfile.write('# New dimension\n')


def readHeightMap(filename='heightMap.txt'):
    if isfile(filename):
        with open(filename, 'r') as infile:
            shape = infile.readline()
            shape = shape[1:-2]
            shape = [int(shape.split(', ')[i]) for i in range(3)]
            heightMap = np.loadtxt(filename, skiprows=1, dtype=np.float16)
            heightMap = heightMap.reshape(shape)
            return heightMap
    return None
