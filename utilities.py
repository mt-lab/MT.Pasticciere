from itertools import tee
from functools import reduce
from configValues import PCD_PATH, focal, pxlSize, cameraHeight
import numpy as np
from numpy import cos, sin, arctan, sqrt, floor, tan, arccos

""" Some tools for convenience """

# TODO: рефактор и комментарии

X, Y, Z = 0, 1, 2


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
            arctan((end[Y] - start[Y]) / (end[X] - start[X]))))
        step_y = step * (end[Y] - start[Y]) / abs(end[Y] - start[Y]) * abs(
            sin(arctan((end[Y] - start[Y]) / (end[X] - start[X]))))
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


def distance(p1, p2=(0, 0), simple=False):
    """ Calculate distance between 2 points either 2D or 3D """
    p1 = list(p1)
    p2 = list(p2)
    if len(p1) < 3:
        p1.append(0)
    if len(p2) < 3:
        p2.append(0)
    if simple:
        return abs(p1[X] - p2[X]) + abs(p1[Y] - p2[Y]) + abs(p1[Z] - p2[Z])
    return sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)


def triangleArea(p1, p2, p3):
    A = np.asarray([p1, p2, p3])
    detA = np.linalg.det(A)
    return abs(detA)

def insideTriangle(p, a, b, c):
    S = triangleArea(a, b, c)
    Spab = triangleArea(p, a, b)
    Spbc = triangleArea(p, b, c)
    Spca = triangleArea(p, c, a)
    if Spab + Spbc + Spca == S:
        return True
    return False

def heightByTrigon(p=(0,0), a=(0,0,0), b=(0,0,0), c=(0,0,0)):
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
    height = c[Z]*S1/S+a[Z]*S2/S+b[Z]*S3/S
    return height

def apprxPointHeight(point, *arg):
    if len(point)<2:
        return None
    if len(arg) != 3:
        if len(arg[0]) == 3:
            arg = arg[0]
        else:
            return None
    arg = sorted(arg, key=lambda p: distance(p[:2]))
    det_values = np.zeros(len(arg))
    points = np.asarray(arg)
    matrix = np.asarray([points[0][:2],points[1][:2],points[2][:2]])
    matrix = np.c_[matrix, np.ones(3)]
    area = np.linalg.det(matrix)
    for i, p1 in enumerate(points):
        j = i +1 if i < len(points)-1 else 0
        k = i +2 if i < len(points)-2 else i + 1 -(len(points)-1)
        p2 = points[j]
        matrix = np.asarray([point[:2],p1[:2], p2[:2]])
        matrix = np.c_[matrix, np.ones(3)]
        det = np.linalg.det(matrix)
        det_values[k] = det
    if area == det_values.sum():
        det_values = abs(det_values)
        a_values = det_values/det_values.sum()
        height = np.multiply(a_values, points[:,2]).sum()
        return height
    return None


def lineFrom2points(p1=(0,0), p2=(0,0)):
    k = (p2[X]-p1[X])/(p2[Y]-p1[Y])
    b = p1[Y]-k*p1[X]
    return k, b


def perpendicular2line(p=(0,0), k=1,b=0):
    pk = 1/k
    pb = p[X]/k + p[Y]
    return pk, pb


def crossectionOfLines(k1 = 1, b1=0, k2=1, b2=0):
    x = (b2-b1)/(k1-k2)
    y = k1*x + b1
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

def findAngleOfView(range:'in pxls'=640, focal: 'in mm'=focal, pxlSize: 'in mm'=pxlSize):
    """

    :param range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxlSize: размер пикселя на матрице
    :return: угол в радианах
    """
    return 2 * arctan(range*pxlSize/2/focal)

def findCameraAngle(viewWidth:'in mm', frameWidth:'in pxls'=640, cameraHeight:'in mm'=cameraHeight, focal:'in mm'=focal, pxlSize:'in mm'=pxlSize):
    """

    :param viewWidth: ширина обзора по центру кадра в мм
    :param frameWidth: ширина кадра в пикселях
    :param cameraHeight: высота камеры над поверхностью в мм
    :param focal: фокусное расстояние линзы
    :param pxlSize: размер пикселя на матрице в мм
    :return: угол наклона камеры в радианах
    """
    viewAngle = findAngleOfView(frameWidth, focal,pxlSize)
    cos_cameraAngle = 2*cameraHeight/viewWidth*tan(viewAngle/2)
    cameraAngle = arccos(cos_cameraAngle)
    return cameraAngle
