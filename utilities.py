from itertools import tee
from functools import reduce
from global_variables import *
from math import cos, sin, atan, sqrt, floor
import numpy as np

""" Some tools for convenience """
# TODO: рефактор и комментарии

def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
            atan((end[Y] - start[Y]) / (end[X] - start[X]))))
        step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y]) * abs(
            sin(atan((end[Y] - start[Y]) / (end[X] - start[X]))))
        step_x = round(step_x, 3)
        step_y = round(step_y, 3)
    d = distance(start, end)
    number_of_slices = floor(d / step)
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


def distance(p1, p2):
    """ Calculate distance between 2 points either 2D or 3D """
    p1 = list(p1)
    p2 = list(p2)
    if len(p1) < 3:
        p1.append(0)
    if len(p2) < 3:
        p2.append(0)
    return sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)


def read_pcd(path=PCD_PATH):
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


def find_point_in_Cloud(point, pcd_xy, pcd_z, offset=(0, 0)):
    """ Find corresponding Z coordinate for a given point in given point cloud """
    point = list(point)[:2]
    # point[X] += offset[X] #-50
    # point[Y] += offset[Y] #120
    z = pcd_z[np.sum(np.abs(pcd_xy - point), axis=1).argmin()][0]
    # point.append(z if z else 0)
    return z if z else 0
