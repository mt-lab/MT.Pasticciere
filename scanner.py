"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
from numpy import cos, tan, sqrt, arctan, pi
import cv2
from typing import Tuple, List, Sequence, Optional, Union, Any
from utilities import X, Y, Z, distance
import globalValues
from utilities import save_height_map, OutOfScanArea, mid_idx, print_objects
from cookie import *
import time
import imutils

# TODO: написать логи

# TODO: комментарии и рефактор; удалить ненужные функции

# TODO: для постобработки облака взять значения внутри найденных контуров (используя маску), найти среднее и отклонения
#       и обрезать всё что выше mean + std (или 2*std)


# масштабные коэффициенты для построения облака точек
kx = 1 / 3  # мм/кадр

# ширина изображения для обработки, пиксели
col_start = 0
col_stop = 640
row_start = 100
row_stop = 400
startMask = cv2.imread('startMask.png', 0)


def find_coords(frame_idx: int, laser_points: np.ndarray, zero_level: np.ndarray,
                frame_shape: Tuple = (480, 640),
                mirrored: bool = False, reverse: bool = False,
                distance_camera2laser: float = 94,
                camera_shift: float = 113,
                camera_angle: float = pi / 6,
                focal_length: float = 2.9,
                pixel_size: float = 0.005,
                table_length: float = 200, table_width: float = 200, table_height: float = 50,
                x0: float = 0, y0: float = 0, z0: float = 0,
                **kwargs) -> np.ndarray:
    """
    Расчёт физических координат точек по их положению в кадре и положению нулевой линии

    :param int frame_idx: номер кадра
    :param np.ndarray laser_points: массив длины frame_shape[1], позиции лазера
    :param np.ndarray zero_level: массив длины frame_shape[1], позиции нулевой линии
    :param Tuple frame_shape: размеры кадра
    :param bool mirrored: ориентация. 0 слева - False, 0 справа - True
    :param bool reverse: направление сканирования. от нуля - False, к нулю - True
    :param float distance_camera2laser: расстояние между камерой и лазером
    :param float camera_shift: смещение камеры по Y
    :param float camera_angle: угол камеры от вертикали
    :param float focal_length: фокусное расстояние камеры
    :param float pixel_size: размер пикселя камеры
    :param float table_length: длина сканируемой зоны
    :param float table_width: ширина сканируемой зоны
    :param float table_height: высота сканируемой зоны
    :param float x0: начальная координата сканирования (начало стола) по X
    :param float y0: начало стола по Y
    :param float z0: начало стола по Z
    :param kwargs: для лишних случайно переданных параметров
    :raises OutOfScanArea: если точка вне зоны сканирования по X
    :return: массив физических координат точек в глобальной системе координат
    """
    positions = np.zeros((frame_shape[1], 3))  # массив для координат точек
    row = mid_idx(zero_level)[0]  # найти центр лазера
    if isinstance(row, np.ndarray):
        row = row.mean()
    # найти высоту камеры по центральной точке
    tan_alpha = tan(camera_angle)
    cos_alpha = cos(camera_angle)
    dpy0 = (row - (frame_shape[0] / 2 - 1)) * pixel_size
    tan_gamma = dpy0 / focal_length
    tan_alphaPgamma = ((tan_alpha + tan_gamma) / (1 - tan_alpha * tan_gamma))
    camera_height = distance_camera2laser / tan_alphaPgamma  # высота камеры
    for column in range(laser_points.size):
        dpy0 = (zero_level[column] - (frame_shape[0] / 2 - 1)) * pixel_size
        dpy = (laser_points[column] - (frame_shape[0] / 2 - 1)) * pixel_size
        dpx = (column - (frame_shape[1] / 2) - 1) * pixel_size
        tan_gamma = dpy0 / focal_length
        tan_theta = dpy / focal_length
        tan_rho = dpx / (focal_length * cos_alpha)
        tan_alphaPgamma = ((tan_alpha + tan_gamma) / (1 - tan_alpha * tan_gamma))
        tan_thetaMgamma = (tan_theta - tan_gamma) / (1 + tan_theta * tan_gamma)
        tan_alphaPtheta = (tan_alpha + tan_theta) / (1 - tan_alpha * tan_theta)
        try:
            sin_thetaMgamma = 1 / (sqrt(1 + (1 / tan_thetaMgamma) ** 2))
            sin_alphaPtheta = 1 / (sqrt(1 + (1 / tan_alphaPtheta) ** 2))
            cos_alphaPgamma = 1 / (sqrt(1 + tan_alphaPgamma ** 2))
            z = camera_height * sin_thetaMgamma / (sin_alphaPtheta * cos_alphaPgamma)
        except ZeroDivisionError:
            z = 0
        y = (camera_height - z) * tan_rho
        y = camera_shift + y if not mirrored else camera_shift - y
        x = frame_idx * kx
        x = x if not reverse else - x
        if x0 + x < 0 or abs(x) > table_length:  # если x вне зоны сканирования райзнуть ошибку
            raise OutOfScanArea(pos=x + x0, bounds=table_length)
        if 0 <= z <= table_height and 0 <= y <= table_width:
            positions[column] = np.array([x + x0, y + y0, z + z0])
        else:
            positions[column] = np.array([x + x0, y0, z0])
    return positions


def find_laser_center(p=(0, 0), m=(0, 0), n=(0, 0)) -> Tuple[float, float]:
    """
    Аппроксимирует по трём точкам параболу и находит её вершину
    Таким образом более точно находит позицию лазера в изображении

    :param Tuple[int, float] p: предыдущая точка от m
    :param Tuple[int, float] m: точка с максимальной интенсивностью, (ряд, интенсивность)
    :param Tuple[int, float] n: следующая точка от m
    :return: уточнённая позиция лазера с субпиксельной точностью и её аппроксимированная интенсивность

    a, b, c - параметры квадратичной функции
    y = ax^2 + bx + c
    """
    # TODO: обработка багов связанных с вычислениями
    if p[X] == m[X] or m[X] == n[X]:  # если точки совпадают, аппроксимация не получится, вернуть среднюю
        return m
    a = ((m[Y] - p[Y]) * (p[X] - n[X]) + (n[Y] - p[Y]) * (m[X] - p[X])) / (
            (p[X] - n[X]) * (m[X] ** 2 - p[X] ** 2) + (m[X] - p[X]) * (n[X] ** 2 - p[X] ** 2))
    if a == 0:  # если а = 0, то получилась линия, вершины нет, вернуть среднюю точку
        return m
    b = ((m[Y] - p[Y]) - a * (m[X] ** 2 - p[X] ** 2))
    c = p[Y] - a * p[X] ** 2 - b * p[X]
    xc = -b / (2 * a)
    yc = a * xc ** 2 + b * xc + c
    return xc, yc


def laplace_of_gauss(img: np.ndarray, ksize: int, sigma: float = .0, delta: float = .0):
    """
    По сути находит яркие линии паттерн которых соответствует гауссовому распределению
    Последовательно по X и Y применяет к изображению фильтр с гауссовым распределением и результат инвертирует
    затем вычисляет лаплассиан изображения

    :param img: 8-битное чб изображение
    :param ksize: размер окна
    :param sigma: дисперсия гаусса
    :param delta: хз
    :return: изображение с применённой обратной двойной гауссовой производной
    """
    kernelX = cv2.getGaussianKernel(ksize, sigma)
    kernelY = kernelX.T
    gauss = -cv2.sepFilter2D(img, cv2.CV_64F, kernelX, kernelY, delta=delta)
    laplace = cv2.Laplacian(gauss, cv2.CV_64F)
    return laplace


def predict_laser(deriv: np.ndarray, min_row=0, max_row=None) -> np.ndarray:
    """

    :param deriv: laplace_of_gauss transformed img
    :param min_row: минимально возможный ряд
    :param max_row: максимально возможный ряд
    :return fine_laser_center: list of predicted laser subpixel positions
    """
    approx_laser_center = np.argmax(deriv, axis=0)
    approx_laser_center[approx_laser_center > (max_row - min_row - 1)] = 0
    fine_laser_center = np.zeros(approx_laser_center.shape)
    for column, row in enumerate(approx_laser_center):
        if row == 0:
            continue
        prevRow = row - 1
        nextRow = row + 1 if row < deriv.shape[0] - 1 else deriv.shape[0] - 1
        p1 = (1.0 * prevRow, deriv[prevRow, column])
        p2 = (1.0 * row, deriv[row, column])
        p3 = (1.0 * nextRow, deriv[nextRow, column])
        fine_laser_center[column] = find_laser_center(p1, p2, p3)[0] + min_row
    if max_row is not None:
        fine_laser_center[fine_laser_center > max_row - 1] = max_row
    return fine_laser_center


def predict_zero_level(array: np.ndarray, mid_row: Union[int, float] = 239) -> Tuple[np.ndarray, float]:
    """
    Расчитывает положение нулевой линии и её угол по крайним точкам из массива

    :param np.ndarray array: массив точек описывающих положение лазера
    :param mid_row: средний ряд кадра, значение по умолчанию если расчёт не получится
    :return: массив точек нулевой линии и тангенс наклона линии от горизонтали
    """
    zero_level = np.full_like(array, mid_row)
    tangent = .0
    nonzero_indices = array.nonzero()[0]
    if nonzero_indices.size:
        first_nonzero = nonzero_indices[0]
        last_nonzero = nonzero_indices[-1]
        tangent = (array[last_nonzero] - array[first_nonzero]) / (last_nonzero - first_nonzero)
        tangent = .0 if tangent == np.inf else tangent
        for column in range(zero_level.size):
            row = (column - first_nonzero) * tangent + array[first_nonzero]
            zero_level[column] = row
    return zero_level, tangent


def calibrate_kx(video_fps: float, printer_velocity: float = 300):
    """
    Функция калибровки коэффициента Kx
    :param video_fps: frames per second
    :param printer_velocity: mm/minute
    :return:
    """
    global kx
    kx = printer_velocity / 60 / video_fps
    print(f'Kx is {kx}')


def get_hsv_mask(img, hsv_lower_bound, hsv_upper_bound, **kwargs):
    """
    Делает битовую маску лазера с цветного изображения

    :param img: исходное изображение BGR
    :param hsv_upper_bound: верхняя граница hsv фильтра
    :param hsv_lower_bound: нижняя граница hsv фильтра
    :return: изображение после обработки
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower_bound), np.array(hsv_upper_bound))
    return mask


def get_max_height(contour, height_map: 'np.ndarray' = globalValues.height_map):
    hMZ = np.dsplit(height_map, 3)[Z].reshape(height_map.shape[0], height_map.shape[1])
    mask = np.zeros(height_map.shape[:2], dtype='uint8')
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked = hMZ[mask == 255]
    maxHeight = masked.max()
    return maxHeight


def find_cookies(img_or_path, height_map: 'np.ndarray' = globalValues.height_map):
    """
    Функция нахождения расположения и габаритов объектов на столе из полученной карты высот
    :param img (np arr, str): карта высот
    :return cookies, result, rectangles, contours: параметры печенек, картинка с визуализацией, параметры боксов
            ограничивающих печеньки, контура границ печенья
    """
    # TODO: подробнее посмотреть происходящее в функции, где то тут баги

    # проверка параметр строка или нет
    original = None
    gray = None
    if isinstance(img_or_path, str):
        original = cv2.imread(img_or_path)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif isinstance(img_or_path, np.ndarray):
        if img_or_path.ndim == 3:
            original = img_or_path.copy()
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif img_or_path.ndim == 2:
            original = cv2.merge((img_or_path.copy(), img_or_path.copy(), img_or_path.copy()))
            gray = img_or_path.copy()
    else:
        raise TypeError(f'передан {type(img_or_path)}, ожидалось str или numpy.ndarray')

    if height_map is None:
        height_map = gray.copy() / 10

    # gray[gray < gray.mean()] = 0

    # избавление от минимальных шумов с помощью гауссова фильтра и отсу трешхолда
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, gausThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # нахождение замкнутых объектов на картинке с помощью морфологических алгоритмов
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(gausThresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
    # найти однозначный задний фон
    sureBg = cv2.dilate(opening, kernel, iterations=3)
    distTrans = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    # однозначно объекты
    ret, sureFg = cv2.threshold(distTrans, 0.1 * distTrans.max(), 255, 0)
    sureFg = np.uint8(sureFg)
    # область в которой находятся контура
    unknown = cv2.subtract(sureBg, sureFg)
    # назначение маркеров
    ret, markers = cv2.connectedComponents(sureFg)
    # отмечаем всё так, чтобы у заднего фона было точно 1
    markers += 1
    # помечаем граничную область нулём
    markers[unknown == 255] = 0
    markers = cv2.watershed(original, markers)
    # выделяем контуры на изображении
    original[markers == -1] = [255, 0, 0]
    # количество печенек на столе (уникальные маркеры минус фон и контур всего изображения)
    numOfCookies = len(np.unique(markers)) - 2
    # вырезаем ненужный контур всей картинки
    blankSpace = np.zeros(gray.shape, dtype='uint8')
    blankSpace[markers == 1] = 255
    blankSpace = cv2.bitwise_not(blankSpace)
    blankSpaceCropped = blankSpace[1:blankSpace.shape[0] - 1, 1:blankSpace.shape[1] - 1]
    # находим контуры на изображении
    contours = cv2.findContours(blankSpaceCropped.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numOfCookies]  # сортируем их по площади
    # применяем на изначальную картинку маску с задним фоном
    result = cv2.bitwise_and(original, original, mask=blankSpace)
    cookies = []
    for contour in contours:
        mask = np.zeros(height_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # height_map_mask = np.dstack((mask, mask.copy(), mask.copy()))
        col, row, w, h = cv2.boundingRect(contour)
        height_map_fragment = height_map.copy()
        height_map_fragment[:, :, Z][mask == 0] = 0
        height_map_fragment = height_map[row:row + h, col:col + w]
        cookie = Cookie(height_map=height_map_fragment, contour=contour, bounding_box=(col, row, w, h))
        cookies.append(cookie)
        cv2.circle(result, cookie.pixel_center[::-1], 3, (0, 255, 0), -1)
    print('Положения печений найдены.')
    return cookies, result


def compare(img, mask, threshold=0.5):
    """
    Побитовое сравнение по маске по количеству белых пикселей
    :param img: изображение для сравнения
    :param mask: применяемая маска
    :param threshold: порог схожести
    :return: True/False в зависимости от схожести
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv, np.array([88, 161, 55]), np.array([144, 255, 92]))
    compare = cv2.bitwise_and(hsv, hsv, mask=mask)
    # cv2.imshow('compare', compare)
    # cv2.imshow('raw', img)
    # cv2.waitKey(30)
    similarity = np.sum(compare == 255) / np.sum(mask == 255) if np.sum(mask == 255) != 0 else 0
    print(f'{similarity:4.2f}')
    if similarity >= threshold:
        return True
    else:
        return False


def avgK(frame, ksize):
    pad = int((ksize - 1) / 2)
    img = np.pad(frame, (pad, pad), 'constant', constant_values=(0, 0))
    result = np.zeros(frame.shape)
    for x in range(pad, frame.shape[1]):
        for y in range(pad, frame.shape[0]):
            crop = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            avg = 0
            if crop[pad, pad] != 0:
                nonZeros = np.sum(crop != 0)
                avg = np.sum(crop) / nonZeros if nonZeros != 0 else 0
            pxlvalue = avg
            result[y, x] = pxlvalue
    return result


def normalize(img, value=1):
    array = img.copy().astype(np.float64)
    array = (array - array.min()) / (array.max() - array.min()) * value
    return array


def detect_start(cap, mask, threshold=0.5):
    """
    Поиск кадра для начала сканирования
    :param cap: видеопоток из файла
    :param mask: маска для поиска кадра
    :param threshold: порог схожести
    :return: если видеопоток кончился -1;
             до тех пор пока для потока не найден нужный кадр False;
             когда кадр найден и все последующие вызовы генератора для данного потока True;
    """
    if threshold == -1:
        print('Сканирование без привязки к глобальной СК')
        yield True
    start = False
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        frameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret != True:
            yield -1
        if compare(frame, mask, threshold):
            start = True
        while start:
            yield True
        print(f'{frameIdx + 1:{3}}/{frameCount:{3}} frames skipped waiting for starting point')
        yield False


def detectStart2(cap, contourPath='', mark_center=(0, 0), threshold=0.5):
    # копия find_cookies заточеная под поиск конкретного контура и его положение с целью привязки к глобальной СК
    # работает сразу с видео потоком по принципу detect_start()
    # TODO: разделить на функции, подумать как обобщить вместе с find_cookies()

    if threshold < 0:
        print('Сканирование без привязки к глобальной СК')
        yield True

    # прочитать контур метки
    markPic = cv2.imread(contourPath, cv2.IMREAD_GRAYSCALE)
    markContours = cv2.findContours(markPic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    markContours = imutils.grab_contours(markContours)
    markContour = sorted(markContours, key=cv2.contourArea, reverse=True)[0]

    start = False
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        frameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()

        if ret != True or not cap.isOpened():
            # если видео закончилось или не открылось вернуть ошибку
            yield -1

        original = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, gausThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(gausThresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
        # найти однозначный задний фон
        sureBg = cv2.dilate(opening, kernel, iterations=3)
        distTrans = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        # однозначно объекты
        ret, sureFg = cv2.threshold(distTrans, 0.1 * distTrans.max(), 255, 0)
        sureFg = np.uint8(sureFg)
        # область в которой находятся контура
        unknown = cv2.subtract(sureBg, sureFg)
        # назначение маркеров
        ret, markers = cv2.connectedComponents(sureFg)
        # отмечаем всё так, чтобы у заднего фона было точно 1
        markers += 1
        # помечаем граничную область нулём
        markers[unknown == 255] = 0
        markers = cv2.watershed(original, markers)
        # выделяем контуры на изображении
        original[markers == -1] = [255, 0, 0]
        # количество контуров на столе (уникальные маркеры минус фон и контур всего изображения)
        numOfContours = len(np.unique(markers)) - 2
        # вырезаем ненужный контур всей картинки
        blankSpace = np.zeros(gray.shape, dtype='uint8')
        blankSpace[markers == 1] = 255
        blankSpace = cv2.bitwise_not(blankSpace)
        blankSpaceCropped = blankSpace[1:blankSpace.shape[0] - 1, 1:blankSpace.shape[1] - 1]
        # находим контуры на изображении
        contours = cv2.findContours(blankSpaceCropped.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        if len(contours) != 0 and contours is not None:
            contours = sorted(contours, key=lambda x: cv2.matchShapes(markContour, x, cv2.CONTOURS_MATCH_I2, 0))[
                       :numOfContours]  # сортируем их по схожести с мастер контуром
            if cv2.matchShapes(markContour, contours[0], cv2.CONTOURS_MATCH_I2, 0) < threshold:
                moments = cv2.moments(contours[0])
                candidateCenter = (int(moments['m01'] / moments['m00']), int(moments['m10'] / moments['m00']))
                if distance(mark_center, candidateCenter) <= 2:
                    start = True
        while start:
            yield True
        print(f'{frameIdx + 1:{3}}/{frameCount:{3}} frames skipped waiting for starting point')
        yield False
        # применяем на изначальную картинку маску с задним фоном
        # result = cv2.bitwise_and(original, original, mask=blankSpace)
        # result = cv2.bitwise_and(original, original, mask=sureBg)


def detect_start3(cap, threshhold=50):
    if threshhold < 0:
        yield True
    start = False
    mirror = False
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    firstLine = False
    while True:
        frameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret is not True or not cap.isOpened():
            yield -1
        if row_start <= row_stop and col_start <= col_stop:
            roi = frame[row_start:row_stop, col_start:col_stop]
        else:
            raise Exception('Incorrect bounds of image')
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        derivative = laplace_of_gauss(gray, 29, 4.45)
        blur = cv2.GaussianBlur(gray, (33, 33), 0)
        _, mask = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)
        derivative = cv2.bitwise_and(derivative, derivative, mask=mask)
        derivative[derivative < 0] = 0
        laser = predict_laser(derivative, row_start, row_stop)
        thresh = np.zeros(frame.shape[:2], dtype='uint8')
        thresh[laser.astype(int), np.array([i for i in range(laser.size)])] = 255
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        # TODO: проверку стабильности пропажи и появления лазера
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshhold, None, roi.shape[1] * 0.8, 10)
        if lines is not None:
            for count, line in enumerate(lines):
                for x1, y1, x2, y2 in line:
                    if (y2 - y1) / (x2 - x1) > tan(1 / 180 * pi):
                        del lines[count]
                        continue
                    #####################################################
                    # """ for debug purposes """
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #################################################################
        # """ for debug purposes """
        # cv2.imshow('thresh', thresh)  #
        # cv2.imshow('frame', frame)  #
        # cv2.waitKey(150)  #
        #####################################
        if not firstLine:
            if lines is not None:
                firstLine = True
        elif not mirror:
            if lines is None:
                mirror = True
        else:
            if lines is not None:
                start = True
        while start:
            print(f'{frameIdx + 1:{3}}/{frameCount:{3}} кадр. Начало сканирования')
            yield True
        print(f'{frameIdx + 1:{3}}/{frameCount:{3}} кадров пропущенно в ожидании точки старта')
        yield False


def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def scanning(cap: cv2.VideoCapture, initial_frame_idx: int = 0, colored: bool = False, **kwargs) -> np.ndarray:
    """

    :param cv2.VideoCapture cap: видеопоток для обработки
    :param int initial_frame_idx: начальный кадр
    :param bool colored: если видео цветное
    :param kwargs: дополнительные параметры для расчётов
    Параметры для сканирования:
        :keyword mirrored: ориентация сканирования. 0 слева - False, 0 справа - True
        :keyword reverse: направление сканирования. от нуля - False, к нулю - True
    Параметры из конфига:
        :keyword hsv_upper_bound: верхняя граница hsv фильтра для цветного скана
        :keyword hsv_lower_bound: нижняя граница hsv фильтра для цветного скана
        :keyword distance_camera2laser: расстояние между камерой и лазером
        :keyword camera_shift: смещение камеры по Y
        :keyword camera_angle: угол камеры от вертикали
        :keyword focal_length: фокусное расстояние камеры
        :keyword pixel_size: размер пикселя камеры
        :keyword table_length: длина сканируемой зоны
        :keyword table_width: ширина сканируемой зоны
        :keyword table_height: высота сканируемой зоны
        :keyword x0: начальная координата сканирования (начало стола) по X
        :keyword y0: начало стола по Y
        :keyword z0: начало стола по Z
    :return: карту высот формы (TOTAL_FRAMES, FRAME_WIDTH, 3), где для каждого пикселя записана [X, Y, Z] координата
    """
    FPS = cap.get(cv2.CAP_PROP_FPS)  # частота кадров видео
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в видео
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH)
    ksize = 29  # размер окна для laplace_of_gauss
    sigma = 4.45  # сигма для laplace_of_gauss
    THRESH_VALUE = 5
    LASER_ANGLE_TOLERANCE = tan(1 / 180 * pi)  # допуск стабильного отклонения угла лазера
    LASER_POS_TOLERANCE = 1  # допуск стабильного отклонения позиции лазера
    frame_idx = 0  # счетчик обработанных кадров
    stability_counter, laser_tangent, laser_row_pos = 0, 0, 0  # метрики стабильности нулевой линии лазера
    zero_level = None  # переменная для нулевой линии
    height_map = np.zeros((TOTAL_FRAMES - initial_frame_idx, FRAME_WIDTH, 3),
                          dtype='float16')  # карта высот
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_idx)  # читать видео с кадра initialFrameIdx
    start = time.time()
    if row_start >= row_stop and col_start >= col_stop:
        raise Exception('Incorrect bounds of image. min_row should be strictly less then max_row.')
    while cap.isOpened():  # пока видео открыто
        ret, frame = cap.read()
        if ret is True:  # пока кадры есть - сканировать
            roi = frame[row_start:row_stop, col_start:col_stop]  # обрезать кадр по зоне интереса
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # конвертировать в грейскейл
            if colored:  # сделать маску для выделения зоны лазера в кадре
                blur = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                mask = get_hsv_mask(blur, **kwargs)
            else:
                blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
                _, mask = cv2.threshold(blur, THRESH_VALUE, 255, cv2.THRESH_BINARY)
            derivative = laplace_of_gauss(gray, ksize, sigma)  # выделить точки похожие на лазер
            derivative = cv2.bitwise_and(derivative, derivative, mask=mask)  # убрать всё что точно не лазер
            derivative[derivative < 0] = 0  # отрицательная производная точно не лазер
            fine_laser_center = predict_laser(derivative, row_start, row_stop)  # расчитать субпиксельные позиции лазера
            # привести к ширине кадра для простоты
            fine_laser_center = np.pad(fine_laser_center, (col_start, FRAME_WIDTH - col_stop), mode='constant')
            # если по производной положений лазера есть всплески отклонение которых от среднего больше чем пять
            # среднеквадратичных, то считать эту точку невалидной и занулить её
            fine_laser_center_deriv = cv2.Sobel(fine_laser_center, -1, 0, 1, None, 1).flatten()
            fine_laser_center[
                abs(fine_laser_center_deriv.mean() - fine_laser_center_deriv) > 5 * fine_laser_center_deriv.std()] = 0
            # расчёт угла и положения нулевой линии
            if stability_counter < FPS:  # если параметры не стабильны в течении 1 секунды (FPS видео)
                # найти нулевую линию и её угол
                zero_level, tangent = predict_zero_level(fine_laser_center, FRAME_HEIGHT / 2 - 1)
                zero_level[(zero_level < row_start) | (zero_level > row_stop)] = row_stop
                #  если параметры линии отклоняются в допустимых пределах
                if abs(laser_tangent - tangent) < LASER_ANGLE_TOLERANCE and \
                        abs(laser_row_pos - zero_level[0]) < LASER_POS_TOLERANCE:
                    stability_counter += 1  # расчитать средние параметры линии по кадрам
                    laser_row_pos = (laser_row_pos * (stability_counter - 1) + zero_level[0]) / stability_counter
                    laser_tangent = (laser_tangent * (stability_counter - 1) + tangent) / stability_counter
                else:  # иначе принять найденные параметры за новые и обнулить счётчик
                    laser_row_pos, laser_tangent, stability_counter = zero_level[0], tangent, 0
                # TODO: вставить предупреждение если лазер долго нестабилен
                # расчитать нулевой уровень по расчитанным параметрам
                zero_level = np.array([-x * laser_tangent + laser_row_pos for x in range(fine_laser_center.size)])
            # занулить точки где положение "лазера" ниже нулевой линии
            fine_laser_center[fine_laser_center < zero_level] = zero_level[fine_laser_center < zero_level]
            try:  # расчитать физические координаты точек лазера
                height_map[frame_idx] = find_coords(frame_idx, fine_laser_center, zero_level, frame_shape=FRAME_SHAPE,
                                                    **kwargs)
            except OutOfScanArea:  # если точки вне зоны, значит закончить обработку
                cap.release()  # закрыть видео
                print('достигнута граница зоны сканирования')
            print(
                f'{frame_idx + initial_frame_idx + 1:{3}}/{TOTAL_FRAMES:{3}} кадров обрабтано за {time.time() - start:4.2f} с\n'
                f'  X: {height_map[frame_idx][0, X]:4.2f} мм; Zmax: {height_map[frame_idx][:, Z].max():4.2f} мм')
            frame_idx += 1
            ##########################################################################
            # """ for debug purposes """
            # for column, row in enumerate(fine_laser_center):
            #     frame[int(row), column] = (0, 255, 0)
            #     frame[int(zero_level[column]), column] = (255, 0, 0)
            # max_column = fine_laser_center.argmax()
            # max_row = int(fine_laser_center[max_column])
            # cv2.circle(frame, (max_column, max_row), 3, (0, 0, 255), -1)
            # cv2.imshow('frame', frame)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(15)
            ##########################################################################
        else:  # кадры кончились или побиты(?)
            cap.release()  # закрыть видео
    else:  # когда видео кончилось
        time_passed = time.time() - start
        print(f'Готово. Потрачено времени на анализ рельефа: {time_passed:3.2f} с\n')
        height_map[:, :, Z][height_map[:, :, Z] < 0] = 0  # везде где Z < 0 приравнять к нулю
        return height_map


def scan(path_to_video: str = globalValues.VID_PATH, colored: bool = False, **kwargs):
    """
    Функция обработки видео (сканирования)
    Находит начало области сканирования, и с этого момента обрабатывает видео поток, получает карту высот.
    Находит расположение объектов в зоне сканирования.
    Сохраняет:
        height_map.txt - карта высот как список координат точек
        height_map.png - карта высот как изображения без обработки и разметки найденных объектов
        cookies.png - визуализация обработанных данных с размеченными объектами

    :param str path_to_video: путь к видео, по умолчанию путь из settings.ini
    :param bool colored: видео цветное или нет
    :param kwargs: дополнительные параметры для сканирования
    :keyword thresh: параметр для детекта начала. thresh < 0 без детекта.
                    если видео цветное то поиск по мастер маске и 0 < thresh < 1 - степень схожести
                    если видео чб то поиск по пропаже/появлению линии и thresh - минимально количество точек на линии
    :keyword mirrored: ориентация сканирования. 0 слева - False, 0 справа - True
    :keyword reverse: направление сканирования. от нуля - False, к нулю - True
    :return: None
    """

    settings_values = globalValues.get_settings_values(**globalValues.settings_sections)  # параметры из конфига

    cap = cv2.VideoCapture(path_to_video)  # чтение видео
    calibrate_kx(cap.get(cv2.CAP_PROP_FPS))  # откалибровать kx согласно FPS

    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    if colored:
        thresh = kwargs.get('thresh', 0.6)
        detector = detect_start(cap, startMask, thresh)
    else:
        thresh = kwargs.get('thresh', 104)
        detector = detect_start3(cap, thresh)
    start = next(detector)
    while not start or start == -1:
        if start == -1:
            print('сканирование не удалось')
            cv2.destroyAllWindows()
            return None
        start = next(detector)
    initial_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    initial_frame_idx = initial_frame_idx - 1 if initial_frame_idx > 0 else 0
    print(f'Точка начала сканирования: {initial_frame_idx + 1: 3d} кадр')

    # сканировать от найденного кадра до конца
    height_map = scanning(cap, initial_frame_idx, **settings_values, **kwargs)
    globalValues.height_map = height_map

    # массив для нахождения позиций объектов
    height_map_z = np.dsplit(height_map, 3)[Z].reshape(height_map.shape[0], height_map.shape[1])
    height_map_8bit = (height_map_z * 10).astype(np.uint8)
    cookies, detected_contours = find_cookies(height_map_8bit, height_map)
    if len(cookies) != 0:
        globalValues.cookies = cookies
        print_objects(cookies, f'Объектов найдено: {len(cookies):{3}}')
        print()

    # сохранить карты
    cv2.imwrite('height_map.png', height_map_8bit)
    save_height_map(height_map)
    cv2.imwrite('cookies.png', detected_contours)

    cap.release()
    cv2.destroyAllWindows()
