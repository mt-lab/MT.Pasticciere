"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
from numpy import cos, sin, tan, sqrt, arctan, pi
import cv2
from typing import Tuple, List
from utilities import X, Y, Z, distance
import globalValues
from utilities import save_height_map
from cookie import *
import time
import imutils

# TODO: написать логи

# TODO: комментарии и рефактор

# TODO: для постобработки облака взять значения внутри найденных контуров (используя маску), найти среднее и отклонения
#       и обрезать всё что выше mean + std (или 2*std)


# масштабные коэффициенты для построения облака точек
kx = 1 / 3  # мм/кадр

# ширина изображения для обработки, пиксели
column_start = 0
column_end = 640
row_start = 100
row_end = 400
startMask = cv2.imread('startMask.png', 0)


def calculate_x(frame_idx):
    return frame_idx * kx


def calculate_yz(frameIdx, pxl_coords=(0, 0), zero_lvl_row=239, frame_shape=(480, 640), pixel_size=0, focal_length=0,
                 camera_angle=0,
                 distance_camera2laser=0, camera_shift=0, **kwargs):
    dpy0 = (zero_lvl_row - (frame_shape[0] / 2 - 1)) * pixel_size
    dpy = (pxl_coords[0] - (frame_shape[0] / 2 - 1)) * pixel_size
    dpx = (pxl_coords[1] - (frame_shape[1] / 2) - 1) * pixel_size
    tan_gamma = dpy0 / focal_length
    tan_theta = dpy / focal_length
    tan_rho = dpx / (focal_length * cos(camera_angle))
    tan_alphaPgamma = ((tan(camera_angle) + tan_gamma) / (1 - tan(camera_angle) * tan_gamma))
    tan_thetaMgamma = (tan_theta - tan_gamma) / (1 + tan_theta * tan_gamma)
    tan_alphaPtheta = (tan(camera_angle) + tan_theta) / (1 - tan(camera_angle) * tan_theta)
    camera_height = distance_camera2laser / tan_alphaPgamma
    try:
        sin_thetaMgamma = 1 / (sqrt(1 + (1 / tan_thetaMgamma) ** 2))
    except ZeroDivisionError:
        sin_thetaMgamma = 0
    try:
        sin_alphaPtheta = 1 / (sqrt(1 + (1 / tan_alphaPtheta) ** 2))
    except ZeroDivisionError:
        sin_alphaPtheta = 0
    try:
        cos_alphaPgamma = 1 / (sqrt(1 + tan_alphaPgamma ** 2))
    except ZeroDivisionError:
        cos_alphaPgamma = 0
    try:
        z = camera_height * sin_thetaMgamma / (sin_alphaPtheta * cos_alphaPgamma)  # высота точки от нулевого уровня
    except ZeroDivisionError:
        z = 0
    y = (camera_height - z) * tan_rho  # координата y точки относительно камеры
    y += camera_shift
    # x = calculate_x(frameIdx) # координата x точки высчитанная по номеру кадра
    return y, z


def find_laser_center(prev=(0, 0), middle=(0, 0), next=(0, 0), default=(240.0, 0)):
    # TODO: обработка багов связанных с вычислениями
    if prev[X] == middle[X] or middle[X] == next[X]:
        return middle
    a = ((middle[Y] - prev[Y]) * (prev[X] - next[X]) + (next[Y] - prev[Y]) * (middle[X] - prev[X])) / (
            (prev[X] - next[X]) * (middle[X] ** 2 - prev[X] ** 2) + (middle[X] - prev[X]) * (
            next[X] ** 2 - prev[X] ** 2))
    if a == 0:
        return middle
    b = ((middle[Y] - prev[Y]) - a * (middle[X] ** 2 - prev[X] ** 2))
    c = prev[Y] - a * prev[X] ** 2 - b * prev[X]

    xc = -b / (2 * a)
    yc = a * xc ** 2 + b * xc + c
    return xc, yc


def laplace_of_gauss(img, ksize, sigma, delta=0.0):
    """

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


def predict_laser(deriv: np.ndarray, row_start=0, row_end=480) -> np.ndarray:
    """

    :param deriv: laplace_of_gauss transformed img
    :return fine_laser_center: list of predicted laser subpixel positions
    """
    approx_laser_center = np.argmax(deriv, axis=0)
    approx_laser_center[approx_laser_center > (row_end - row_start - 1)] = 0
    fine_laser_center = np.zeros(approx_laser_center.shape)
    for column, row in enumerate(approx_laser_center):
        if row == 0:
            continue
        prevRow = row - 1
        nextRow = row + 1 if row < deriv.shape[0] - 1 else deriv.shape[0] - 1
        p1 = (1.0 * prevRow, deriv[prevRow, column])
        p2 = (1.0 * row, deriv[row, column])
        p3 = (1.0 * nextRow, deriv[nextRow, column])
        fine_laser_center[column] = find_laser_center(p1, p2, p3)[0] + row_start
    fine_laser_center[fine_laser_center > row_end - 1] = row_end
    return fine_laser_center


def predict_zero_level(array: np.ndarray, mid_row=239, row_start=0, row_end=479, img_to_mark=None, **kwargs) \
        -> Tuple[np.ndarray, float]:
    """

    :param array: list of points describing laser position
    :return:
    """
    zero_level = np.full_like(array, mid_row)
    tangent = .0
    nonzero_indices = array.nonzero()[0]
    if nonzero_indices.size:
        first_nonzero = nonzero_indices[0]
        last_nonzero = nonzero_indices[-1]
        tangent = (array[last_nonzero] - array[first_nonzero]) / (last_nonzero - first_nonzero)
        tangent = .0 if tangent == np.inf else tangent
        if img_to_mark is not None:
            # for debug purposes
            cv2.circle(img_to_mark, (first_nonzero, int(array[first_nonzero])), 3, (127, 0, 127), -1)
            cv2.circle(img_to_mark, (last_nonzero, int(array[last_nonzero])), 3, (127, 0, 127), -1)
        for column in range(zero_level.size):
            row = (column - first_nonzero) * tangent + array[first_nonzero]
            zero_level[column] = row if row_start < row < row_end else 0
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


def get_mask(img, hsv_lower_bound, hsv_upper_bound, zero_level=0):
    """
    Делает битовую маску лазера с изображения и применяет к ней lineThinner

    :param img: исходное изображение
    :param zero_level:
    :return: изображение после обработки
    """
    img = img[zero_level:, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower_bound), np.array(hsv_upper_bound))
    gauss = cv2.GaussianBlur(mask, (5, 5), 0)
    ret2, gauss_thresh = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gauss_thresh


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
        return 'Вы передали какую то дичь'

    if height_map is None:
        height_map = gray.copy() / 10

    gray[gray < gray.mean()] = 0

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
    result = cv2.bitwise_and(original, original, mask=sureBg)
    # расчет центров и поворотов контуров сразу в мм
    cookies = []
    for contour in contours:
        tmp = contour.copy()  # скопировать контур, чтобы не изменять оригинал
        tmp = np.float32(tmp)
        # перевести из пикселей в мм
        for point in tmp:
            px = int(point[0][1])
            py = int(point[0][0])
            point[0][1] = height_map[px, py, X]
            point[0][0] = height_map[px, py, Y]
        moments = cv2.moments(tmp)
        # найти центр контура и записать его в СК принтера
        M = cv2.moments(contour)
        Cx = int(M['m10'] / M['m00'])
        Cy = int(M['m01'] / M['m00'])
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        center = height_map[Cy, Cx]
        # найти угол поворота контура (главная ось)
        a = moments['m20'] / moments['m00'] - cx ** 2
        b = 2 * (moments['m11'] / moments['m00'] - cx * cy)
        c = moments['m02'] / moments['m00'] - cy ** 2
        theta = 1 / 2 * arctan(b / (a - c)) + (a < c) * pi / 2
        # угол поворота с учетом приведения в СК принтера
        rotation = theta + pi / 2
        maxHeight = get_max_height(contour, height_map)
        cookies.append(Cookie(center=center[:2], centerHeight=center[Z], rotation=rotation, maxHeight=maxHeight))
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


def detectStart(cap, mask, threshold=0.5):
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
    # работает сразу с видео потоком по принципу detectStart()
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


def detectStart3(cap, sensitivity=50):
    if sensitivity < 0:
        yield True
    start = False
    mirror = False
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    firstLine = False
    while True:
        frameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret != True or not cap.isOpened():
            yield -1
        if row_start <= row_end and column_start <= column_end:
            roi = frame[row_start:row_end, column_start:column_end]
        else:
            raise Exception('Incorrect bounds of image')
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        derivative = laplace_of_gauss(gray, 29, 4.45)
        blur = cv2.GaussianBlur(gray, (33, 33), 0)
        _, mask = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)
        derivative = cv2.bitwise_and(derivative, derivative, mask=mask)
        derivative[derivative < 0] = 0
        laser = predict_laser(derivative, row_start, row_end)
        thresh = np.zeros(frame.shape[:2], dtype='uint8')
        thresh[laser.astype(int), np.array([i for i in range(laser.size)])] = 255
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, sensitivity, None, roi.shape[1] * 0.8, 10)
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

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def scanning(cap, initial_frame_idx=0, colored=False, table_length=200, table_width=200, table_height=50, x0=0, y0=0,
             z0=0, **kwargs):
    # читать видео с кадра initialFrameIdx
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_idx)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    # карта высот
    height_map = np.zeros((total_frames - initial_frame_idx, column_end - column_start + 1, 3), dtype='float16')
    ksize = 29
    sigma = 4.45
    frame_counter, laser_tangent, laser_row_pos = 0, 0, 0

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        length = calculate_x(frame_idx)  # координата x точки высчитанная по номеру кадра
        if length < 0:
            frame_idx += 1
            continue
        if length >= table_length:
            # если отсканированная область по длине уже равна или больше рабочей, закончить сканирование
            print('Предел рабочей зоны.')
            ret = False

        # пока кадры есть - сканировать
        if ret is True:
            if row_start <= row_end and column_start <= column_end:
                roi = frame[row_start:row_end, column_start:column_end]
            else:
                raise Exception('Incorrect bounds of image')
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            derivative = laplace_of_gauss(gray, ksize, sigma)
            if colored:
                blur = cv2.GaussianBlur(roi, (33, 33), 0)
                # mask = getMask(blur, frame.shape[0] / 2 - 1)
                mask = get_mask(blur, row_start, **kwargs)
            else:
                blur = cv2.GaussianBlur(gray, (33, 33), 0)
                _, mask = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)
            derivative = cv2.bitwise_and(derivative, derivative, mask=mask)
            fine_laser_center = predict_laser(derivative, row_start, row_end)
            # расчёт угла и положения нулевой линии
            if frame_counter >= cap.get(cv2.CAP_PROP_FPS):  # если параметры стабильны в течении 1 секунды (FPS видео)
                # считать нулевой уровень по расчитанным параметрам
                zero_level = np.array([-x * laser_tangent + laser_row_pos for x in range(fine_laser_center.size)])
            else:
                # иначе найти нулевую линию и её угол
                zero_level, tangent = predict_zero_level(fine_laser_center, frame.shape[0] / 2 - 1, row_start, row_end,
                                                         **kwargs)
                # если параметры линии отклоняются в пределах 1 градуса и 1 пикселя
                if abs(laser_tangent - tangent) < tan(1 / 180 * pi) and abs(laser_row_pos - zero_level[0]) < 1:
                    # расчитать средние параметры линии по кадрам где отклонения в пределах
                    frame_counter += 1
                    laser_row_pos = (laser_row_pos * (frame_counter - 1) + zero_level[0]) / frame_counter
                    laser_tangent = (laser_tangent * (frame_counter - 1) + tangent) / frame_counter
                else:
                    # иначе принять найденные параметры за новые и обнулить счётчик
                    laser_row_pos, laser_tangent, frame_counter = zero_level[0], tangent, 0
            fine_laser_center[fine_laser_center < zero_level] = zero_level[fine_laser_center < zero_level]
            max_height = 0
            # расчитать физические координаты точек лазера
            for column, row in enumerate(fine_laser_center):
                zero = zero_level[column]
                width, height = calculate_yz(frame_idx, (row, column), zero, frame.shape, **kwargs)
                max_height = max(height, max_height)
                if 0 <= width <= table_width:
                    height_map[frame_idx, column, X] = length + x0
                    height_map[frame_idx, column, Y] = width + y0
                else:
                    # если координата по ширине не на столе, то не записывать точку
                    continue
                height_map[frame_idx, column, Z] = height + z0 if 0 <= height <= table_height else z0
            print(
                f'{frame_idx + initial_frame_idx + 1:{3}}/{total_frames:{3}} кадров обрабтано за {time.time() - start:4.2f} с\n'
                f'  X: {length:4.2f} мм; Zmax: {max_height:4.2f} мм')
            frame_idx += 1
            ##########################################################################
            """ for debug purposes """
            for column, row in enumerate(fine_laser_center):
                frame[int(row), column] = (0, 255, 0)
                frame[int(zero_level[column]), column] = (255, 0, 0)
            max_column = fine_laser_center.argmax()
            max_row = int(fine_laser_center[max_column])
            cv2.circle(frame, (max_column + column_start, max_row), 3, (0, 0, 255), -1)
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.waitKey(15)
            ##########################################################################
        else:
            # когда видео кончилось
            time_passed = time.time() - start
            print(f'Готово. Потрачено времени на анализ рельефа: {time_passed:3.2f} с\n')
            return height_map


def scan(path_to_video=globalValues.VID_PATH, sensitivity=104, colored=False, threshold=0.6):
    """
    Функция обработки видео (сканирования)
    :param path_to_video: путь к видео, по умолчанию путь из settings.ini
    :return: None
    """

    settings_values = globalValues.get_settings_values(**globalValues.settings_sections)

    cap = cv2.VideoCapture(path_to_video)  # чтение видео
    calibrate_kx(cap.get(cv2.CAP_PROP_FPS))

    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    if colored:
        detector = detectStart(cap, startMask, threshold)
    else:
        detector = detectStart3(cap, sensitivity)
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
    height_map = scanning(cap, initial_frame_idx, **settings_values)
    globalValues.height_map = height_map

    # массив для нахождения позиций объектов
    height_map_z = np.dsplit(height_map, 3)[Z].reshape(height_map.shape[0], height_map.shape[1])
    height_map_8bit = (height_map_z * 10).astype(np.uint8)
    cookies, detected_contours = find_cookies(height_map_8bit, height_map)
    if len(cookies) != 0:
        globalValues.cookies = cookies
        print(f'Объектов найдено: {len(cookies):{3}}')
        print('#############################################')
        for i, cookie in enumerate(cookies, 1):
            print(f'Объект №{i:3d}')
            print('#############################################')
            print(cookie)
            print('#############################################')
        print()

    # сохранить карты
    cv2.imwrite('height_map.png', height_map_8bit)
    save_height_map(height_map)
    cv2.imwrite('cookies.png', detected_contours)

    cap.release()
    cv2.destroyAllWindows()
