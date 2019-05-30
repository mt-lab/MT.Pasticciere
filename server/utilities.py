from itertools import tee
import math as m
import numpy as np

""" Some tools for convenience """

X, Y, Z = 0, 1, 2


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def diap(start, end, step=1):
    """ (s, e) -> (s, m1), (m1, m2) .. (m_n, e) """
    # PROBABLY TO BE REWRITTEN
    x, y = start[X], start[Y]
    # if line is horizontal
    if end[Y] - start[Y] == 0:
        try:
            step_x = step * (end[X] - start[X]) / abs(end[X] - start[X])
        except ZeroDivisionError:  # ZeroDivision only occurs when line is vertical or start point matches end point
            yield start  # return start point if start matches end
    else:
        try:
            # step_[x, y] = step * (sign determiner) * abs(share of step)
            step_x = step * (end[X] - start[X]) / abs(end[X] - start[X]) * abs(m.cos(
                m.atan((end[Y] - start[Y]) / (end[X] - start[X]))))
            step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y]) * abs(
                m.sin(m.atan((end[Y] - start[Y]) / (end[X] - start[X]))))
        except ZeroDivisionError:
            step_x = 0
            step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y])
        # while distance between current point and end bigger then step
        while m.sqrt((end[X] - x) ** 2 + (end[Y] - y) ** 2) > step:
            yield (x, y)
            x += step_x
            y += step_y
        yield (x, y)
    # always return end point in the end of sequence
    yield end


def distance(p1, p2):
    """ Calculate distance between 2 points either 2D or 3D """
    p1 = list(p1)
    p2 = list(p2)
    if len(p1) < 3:
        p1.append(0)
    if len(p2) < 3:
        p2.append(0)
    return m.sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)


def read_pcd(path=''):
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


def find_point_in_Cloud(point, pcd_xy, pcd_z, offset = (0, 0)):
    """ Find corresponding Z coordinate for a given point in given point cloud """
    point = list(point)[:2]
    # point[X] += offset[X] #-50
    # point[Y] += offset[Y] #120
    z = pcd_z[np.sum(np.abs(pcd_xy - point), axis=1).argmin()][0]
    # point.append(z if z else 0)
    return z if z else 0
