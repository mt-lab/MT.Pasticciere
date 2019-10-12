from typing import List, Union, Tuple, Iterable, Sequence, Optional, Any, TypeVar
from itertools import tee
from functools import reduce
import numpy as np
from numpy import arctan, sqrt, tan, arccos
from ezdxf.math.vector import Vector

""" Some tools for convenience """

# TODO: рефактор и комментарии

X, Y, Z = 0, 1, 2

T = TypeVar('T')


def mid_idx(arr: Sequence[T], shift: Optional[int] = 0) -> Union[T, int]:
    mid = int(len(arr) / 2) + shift
    if len(arr) % 2 == 0:
        return arr[slice(mid - 1, mid + 1)], mid - 1
    else:
        return arr[mid], mid


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def print_objects(objects: Any, pre_msg: Optional[str] = None, object_msg: Optional[str] = '',
                  sep: Optional[str] = '#'):
    if pre_msg is not None:
        print(pre_msg)
    print(sep * 30)
    for count, obj in enumerate(objects):
        print(f'{object_msg}', f'№{count:3d}')
        print(sep * 30)
        print(obj)
        print(sep * 30)


def closed(iterable):
    """ ABCD -> A, B, C, D, A """
    return [item for item in iterable] + [iterable[0]]


def diap(start, end, step=1) -> List[float]:
    """ Принимает две точки пространства и возвращает точки распределенные на заданном расстоянии
     между данными двумя.
    :param Iterable[float] start: начальная точка в пространстве
    :param Iterable[float] end: конечная точка в пространстве
    :param float step: шаг между точками
    :return: точка между start и end
    """
    start = Vector(start)
    end = Vector(end)
    d = start.distance(end)
    number_of_steps = int(d / step)
    ratio = step / d
    for i in range(number_of_steps):
        yield list(start.lerp(end, i * ratio))
    yield list(end)


def avg(*arg) -> float:
    """
     Вычисляет среднее арифметическое
    :param arg: набор чисел
    :return: среднее
    """
    return reduce(lambda a, b: a + b, arg) / len(arg)


def distance(p1, p2: Union[List[float], Tuple[float]] = (.0, .0, .0), simple=False) -> float:
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


def triangle_area(a, b, c, signed=False):
    # TODO: вынести в elements
    area = (a[X] * (b[Y] - c[Y]) + b[X] * (c[Y] - a[Y]) + c[X] * (a[Y] - b[Y])) / 2.0
    return area if signed else abs(area)


def polygon_area(poly: List):
    total_area = 0
    for i in range(len(poly) - 2):
        a, b, c = poly[0], poly[i + 1], poly[i + 2]
        area = triangle_area(a, b, c, True)
        total_area += area
    return abs(total_area)


def inside_polygon(p, *args: List[List[float]]):
    # TODO: use cv.polygonTest
    p = [round(coord, 2) for coord in p]
    boundary_area = round(polygon_area(args))
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
                       focal: float = 2.9,
                       pxl_size: float = 0.005) -> float:
    """

    :param view_range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxl_size: размер пикселя на матрице
    :return: угол в радианах
    """
    return 2 * arctan(view_range * pxl_size / 2 / focal)


def find_camera_angle(view_width: float,
                      frame_width: int = 640,
                      camera_height: float = 150,
                      focal: float = 2.9,
                      pxl_size: float = 0.005) -> float:
    """
    Вычисление угла наклона камеры от вертикали по ширине обзора камеры и
    её высоте над поверхностью.
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


def find_camera_angle2(img: np.ndarray,
                       distance_camera2laser: float = 90,
                       camera_height: float = 150,
                       focal: float = 2.9,
                       pixel_size: float = 0.005,
                       **kwargs) -> Tuple[float, np.ndarray]:
    """
    Вычисление угла наклона камеры от вертикали по расстоянию до лазера,
    высоте над поверхностью стола и изображению положения лазера с камеры.
    :param np.ndarray img: цветное или чб изображение (лучше чб)
    :param float distance_camera2laser: расстояние от камеры до лазера в мм
    :param float camera_height: расстояние от камеры до поверхности в мм
    :param float focal: фокусное расстояние камеры в мм
    :param float pixel_size: размер пикселя на матрице в мм
    :param kwargs: дополнительные параметры для обработки изображения
    :keyword ksize: int размер окна для ф-ии гаусса, нечетное число
    :keyword sigma: float сигма для ф-ии гаусса
    :raises Exception: 'Dimensions are not correct' if img not 3D or 2D
    :return: угол камеры в радианах
    """
    import cv2
    from scanner import predict_laser, laplace_of_gauss
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise Exception('Dimensions are not correct')
    roi = kwargs.get('roi', ((0, gray.shape[0]), (0, gray.shape[1])))
    gray = gray[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    deriv = laplace_of_gauss(gray, kwargs.get('ksize', 29), kwargs.get('sigma', 4.45))
    laser = predict_laser(deriv, 0, img.shape[0])
    middle, idx = mid_idx(laser)
    middle = avg(laser[idx])
    img[laser.astype(int) + roi[0][0], [i + roi[1][0] for i in range(laser.size)]] = (0, 255, 0)
    cv2.circle(img, (idx + roi[1][0], int(middle) + roi[0][0]), 2, (0, 0, 255), -1)
    dp0 = (middle - img.shape[0] / 2 - 1) * pixel_size
    tan_gamma = dp0 / focal
    d2h_ratio = distance_camera2laser / camera_height
    tan_alpha = (d2h_ratio - tan_gamma) / (1 + d2h_ratio * tan_gamma)
    camera_angle = arctan(tan_alpha)
    return camera_angle, img


def angle_from_video(video, **kwargs):
    import cv2
    cap = cv2.VideoCapture(video)
    mean_angle = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('max_row of video')
            break
        angle, frame = find_camera_angle2(frame, **kwargs)
        count += 1
        mean_angle = (mean_angle * (count - 1) + angle) / count
        print(np.rad2deg(mean_angle), np.rad2deg(angle))
        cv2.imshow('frame', frame)
        ch = cv2.waitKey(15)
        if ch == 27:
            print('closed by user')
            break
    cv2.destroyAllWindows()


def save_height_map(height_map: np.ndarray, filename='height_map.txt'):
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


def nothing(*args, **kwargs):
    pass


class OutOfScanArea(Exception):
    def __init__(self, message='Out of scaning area', **kwargs):
        pos = kwargs.get('pos')
        bounds = kwargs.get('bounds')
        msg = message
        if pos is not None:
            msg += f'. Real position {pos}'
        if bounds is not None:
            msg += f'. Allowed Range {bounds}'
        self.message = msg


def height_from_stream(video):
    import cv2
    from scanner import laplace_of_gauss, predict_laser, predict_zero_level, find_coords
    params = {'distance_camera2laser': (900, 1500),
              'camera_angle': (300, 900),
              'Ynull': (0, 480),
              'Yend': (480, 480),
              'Xnull': (0, 640),
              'Xend': (640, 640),
              }
    setwin = 'settings'

    cv2.namedWindow(setwin)
    for key, (initial, max_val) in params.items():
        cv2.createTrackbar(key, setwin, initial, max_val, nothing)

    def get_params(par: dict, win=setwin):
        par_vals = {}
        for key in par.keys():
            par_vals[key] = cv2.getTrackbarPos(key, win)
        return par_vals

    K = 29
    SIGMA = 4.45
    THRESH = 5

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        values = get_params(params)
        camera_angle = values.get('camera_angle', 300) / 10 * np.pi / 180
        values['camera_angle'] = camera_angle
        d = values.get('distance_camera2laser', 900) / 10
        values['distance_camera2laser'] = d
        row_start = values.pop('Ynull')
        row_stop = values.pop('Yend')
        col_start = values.pop('Xnull')
        col_stop = values.pop('Xend')
        if row_start >= row_stop and col_start >= col_stop:
            roi = frame
        else:
            roi = frame[row_start:row_stop, col_start:col_stop]
        if ret is False:
            cap.release()
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (K, K), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, mask = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)
        der = laplace_of_gauss(gray, K, SIGMA)
        der = cv2.bitwise_and(der, der, mask=mask)
        laser = predict_laser(der, row_start, row_stop)
        laser = np.pad(laser, (col_start, frame.shape[1] - col_stop), mode='constant')
        zero_lvl, _ = predict_zero_level(laser, frame.shape[0] // 2)
        # zero_lvl[(zero_lvl < row_start) | (zero_lvl > row_stop)] = row_stop
        laser[laser < zero_lvl] = zero_lvl[laser < zero_lvl]
        coords = find_coords(0, laser, zero_lvl, frame.shape,
                             focal_length=2.9,
                             pixel_size=0.005,
                             table_height=100,
                             camera_shift=113,
                             **values)
        height = coords[:, Z]
        for column in range(col_start, col_stop):
            row = laser[column]
            frame[int(row), column] = (0, 255, 0)
            frame[int(zero_lvl[column]), column] = (255, 0, 0)
        frame[[row_start, row_stop - 1], col_start:col_stop] = (127, 127, 0)
        frame[row_start:row_stop, [col_start, col_stop - 1]] = (127, 127, 0)
        max_column = laser[col_start:col_stop].argmax() + col_start
        max_row = int(laser[max_column])
        cv2.circle(frame, (max_column, max_row), 3, (0, 0, 255), -1)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        ch = cv2.waitKey(15)
        print(coords[col_start:col_stop, Z].max())
        if ch == 27:
            cap.release()
            continue
    cap.release()
    cv2.destroyAllWindows()
