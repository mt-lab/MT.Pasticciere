"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
from numpy import cos, tan, sqrt, pi
import cv2
from typing import Tuple
from utilities import save_height_map, OutOfScanArea, Error, mid_idx, print_objects
import globalValues
from globalValues import get_settings_values, settings_sections
from cookie import *
import time

# TODO: написать логи

# TODO: комментарии и рефактор; удалить ненужные функции

# TODO: для постобработки облака взять значения внутри найденных контуров (используя маску), найти среднее и отклонения
#       и обрезать всё что выше mean + std (или 2*std)


# масштабные коэффициенты для построения облака точек
kx = 1 / 3  # мм/кадр

settings = ['hsv_upper_bound',
            'hsv_lower_bound',
            'distance_camera2laser',
            'camera_shift',
            'camera_angle',
            'focal_length',
            'pixel_size',
            'table_length',
            'table_width',
            'table_height',
            'x0',
            'y0',
            'z0',
            ]


def find_coords(frame_idx: int, laser_points: np.ndarray, zero_level: np.ndarray,
                frame_shape: Tuple = (480, 640),
                mirrored: bool = False, reverse: bool = False,
                distance_camera2laser: float = 94,
                camera_shift: float = 113,
                camera_angle: float = pi / 6,
                focal_length: float = 2.9,
                pixel_size: float = 0.005,
                table_length: float = 200, table_width: float = 200, table_height: float = 50,
                x0: float = 0, y0: float = 0, z0: float = 0) -> np.ndarray:
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
        if tan_thetaMgamma == 0 or tan_alphaPgamma == 0 or tan_alphaPtheta == 0:
            z = 0
        else:
            try:
                sin_thetaMgamma = 1 / (sqrt(1 + (1 / tan_thetaMgamma) ** 2))
                sin_alphaPtheta = 1 / (sqrt(1 + (1 / tan_alphaPtheta) ** 2))
                cos_alphaPgamma = 1 / (sqrt(1 + tan_alphaPgamma ** 2))
                z = camera_height * sin_thetaMgamma / (sin_alphaPtheta * cos_alphaPgamma)
            except ZeroDivisionError:
                z = 0.0
        y = (camera_height - z) * tan_rho
        y = camera_shift + y if not mirrored else camera_shift - y
        x = frame_idx * kx
        x = x if not reverse else - x
        if x0 + x < 0 or abs(x) > table_length:  # если x вне зоны сканирования райзнуть ошибку
            raise OutOfScanArea(pos=x + x0, bounds=table_length)
        if 0 <= z <= table_height:
            positions[column] = np.array([x + x0, y + y0, z + z0])
        else:
            positions[column] = np.array([x + x0, y + y0, z0])
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
    if p[0] == m[0] or m[0] == n[0]:  # если точки совпадают, аппроксимация не получится, вернуть среднюю
        return m
    a = .5 * (n[1] + p[1]) - m[1]
    if a == 0:  # если а = 0, то получилась линия, вершины нет, вернуть среднюю точку
        return m
    b = (m[1] - p[1]) - a * (2 * m[0] - 1)
    c = p[1] - p[0] * (a * p[0] + b)
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


def gray_gravity(img: np.ndarray, row_start=0, row_stop=480) -> np.ndarray:
    ggm = img.copy() / np.amax(img)
    centers = np.sum(ggm * (np.mgrid[:ggm.shape[0], :ggm.shape[1]][0] + 1), axis=0) / np.sum(ggm,
                                                                                             axis=0) + row_start - 1
    centers[centers > row_stop - 1] = row_stop - 1
    centers[np.isinf(centers) | np.isnan(centers)] = 0
    return centers


def predict_laser(deriv: np.ndarray, min_row=0, max_row=None) -> np.ndarray:
    # TODO: написать варианты не использующие LoG:
    #       1. применять квадратичную аппроксимацию сразу на изображение (отсутствие возможности цветного скана или в ярком освещении)
    #       2. применять GGM (возможно ухудшение точности)
    #       3. применять IGGM (возможно замедление работы алгоритма)
    #       4. применять фильтр по фону на LoG и всё бышеперечисленное
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
        p1 = (1. * prevRow, 1. * deriv[prevRow, column])
        p2 = (1. * row, 1. * deriv[row, column])
        p3 = (1. * nextRow, 1. * deriv[nextRow, column])
        fine_laser_center[column] = find_laser_center(p1, p2, p3)[0] + min_row
    fine_laser_center[fine_laser_center > max_row - 1] = max_row
    return fine_laser_center


def predict_zero_level(laser: np.ndarray, mid_row: Union[int, float] = 239) -> Tuple[np.ndarray, float]:
    """
    Расчитывает положение нулевой линии и её угол по крайним точкам из массива

    :param np.ndarray laser: массив точек описывающих положение лазера
    :param mid_row: средний ряд кадра, значение по умолчанию если расчёт не получится
    :return: массив точек нулевой линии и тангенс наклона линии от горизонтали
    """
    zero_level = np.full_like(laser, mid_row)
    tangent = .0
    nonzero_indices = laser.nonzero()[0]
    if nonzero_indices.size:
        first_nonzero = nonzero_indices[0]
        last_nonzero = nonzero_indices[-1]
        tangent = (laser[last_nonzero] - laser[first_nonzero]) / (last_nonzero - first_nonzero)
        tangent = .0 if tangent == np.inf else tangent
        zero_level = np.array(
            [(column - first_nonzero) * tangent + laser[first_nonzero] for column in range(laser.size)])
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


def get_hsv_mask(img, hsv_lower_bound, hsv_upper_bound):
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


def normalize(img, value=1):
    array = img.copy().astype(np.float64)
    array = (array - array.min()) / (array.max() - array.min()) * value
    return array


def detect_start3(cap, threshhold=50, roi=None):
    if threshhold < 0:
        yield True
    start = False
    mirror = False
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    firstLine = False
    row_start, row_stop, col_start, col_stop = roi if roi is not None else (0, FRAME_HEIGHT, 0, FRAME_WIDTH)
    if row_start >= row_stop and col_start >= col_stop:
        raise Exception('Incorrect bounds of image. min_row should be strictly less then max_row.')
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
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshhold, None, roi.shape[1] * 0.6, 10)
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
            print(f'{frameIdx + 1:{3}}/{TOTAL_FRAMES:{3}} кадр. Начало сканирования')
            yield True
        print(f'{frameIdx + 1:{3}}/{TOTAL_FRAMES:{3}} кадров пропущенно в ожидании точки старта')
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


def scanning(cap: cv2.VideoCapture, initial_frame_idx: int = 0, **kwargs) -> np.ndarray:
    """

    :param cv2.VideoCapture cap:    видеопоток для обработки
    :param int initial_frame_idx:   начальный кадр
    :param kwargs:                  дополнительные параметры для расчётов
    Параметры для сканирования:
        :keyword mirrored: ориентация сканирования
            (default) False - ноль слева
                      True - ноль справа
        :keyword reverse: направление сканирования
            (default) False - от нуля
                      True - к нулю
        :keyword colored: изображение цветное или нет
            (default) False - черно-белое
                      True - цветное
        :keyword extraction_mode: способ нахождения примерного центра лазера
            (default) 'max_peak' - по максимальной интенсивности на изображении с аппроксимацией по параболе
                      'log' - максимальное значние лапласиана гаусса с аппроксимацией по параболе
                      'ggm' - gray gravity method
                      'iggm' - # TODO: improved gray gravity method
        :keyword threshold: значение трэшхолда для маски
            (default) thresh <= 0 - OTSU
                      thresh > 0 - обычный трешхолд
        :keyword avg_time: время в секундах для усреднения нулевой линии с учетом стабильности
            (default) avg_time <= 0 без усреднения, считать в каждом кадре
                      avg_time > 0 с усреднением
        :keyword laser_angle_tol: допуск отклонения угла лазера в градусах при усреднении
            (default) 0.1 градуса
        :keyword laser_pos_tol: допуск отклонения положения лазера в пикселях при усреднении
            (default) 0.1 пикселя
        :keyword roi:   область интереса в кадре в формате (row_start, row_stop, col_start, col_stop)
            (default) по всей области изображения
        :keyword debug: флаг отладки
            (default) False - отключена
                      True - включена визуализауция процесса
    Параметры из конфига:
        :keyword hsv_upper_bound:       верхняя граница hsv фильтра для цветного скана
        :keyword hsv_lower_bound:       нижняя граница hsv фильтра для цветного скана
        :keyword distance_camera2laser: расстояние между камерой и лазером
        :keyword camera_shift:          смещение камеры по Y
        :keyword camera_angle:          угол камеры от вертикали
        :keyword focal_length:          фокусное расстояние камеры
        :keyword pixel_size:            размер пикселя камеры
        :keyword table_length:          длина сканируемой зоны
        :keyword table_width:           ширина сканируемой зоны
        :keyword table_height:          высота сканируемой зоны
        :keyword x0:                    начальная координата сканирования (начало стола) по X
        :keyword y0:                    начало стола по Y
        :keyword z0:                    начало стола по Z
    Дополнительные параметры:
        :keyword log_k:     размер окна для 'log'
            (default) 29
        :keyword log_sigma: среднеквардратичное отклонение для 'log'
            (default) 4.45
    :return: карту высот формы (TOTAL_FRAMES, col+stop-col_start, 3), где каждый пиксель это [X, Y, Z] координата
    """
    FPS = cap.get(cv2.CAP_PROP_FPS)  # частота кадров видео
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в видео
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH)
    REVERSE = kwargs.pop('reverse', False)
    MIRRORED = kwargs.pop('mirrored', False)
    EXTRACTION_MODE = kwargs.pop('extraction_mode', 'max_peak').lower()  # метод для расчёта лазера
    THRESH_VALUE = kwargs.pop('threshold', 0)  # пороговое значение для создания маски с простым трешхолдом
    AVG_TIME = round(kwargs.pop('avg_time', 0) * FPS)  # время на усреднение стабильного лазера; при <=0 то не усреднять
    LASER_ANGLE_TOLERANCE = np.deg2rad(kwargs.pop('laser_angle_tol', 0.1))  # допуск стабильного отклонения угла лазера
    LASER_POS_TOLERANCE = kwargs.pop('laser_pos_tol', 0.1)  # допуск стабильного отклонения позиции лазера
    ksize = kwargs.pop('log_k', 29)  # размер окна для laplace_of_gauss
    sigma = kwargs.pop('log_sigma', 4.45)  # сигма для laplace_of_gauss
    colored = kwargs.pop('colored', False)
    hsv_upper_bound = kwargs.pop('hsv_upper_bound', (0, 0, 0))
    hsv_lower_bound = kwargs.pop('hsv_lower_bound', (255, 255, 255))
    verbosity = kwargs.pop('verbosity', 0)
    row_start, row_stop, col_start, col_stop = kwargs.pop('roi', (0, FRAME_HEIGHT, 0, FRAME_WIDTH))
    DEBUG = kwargs.pop('debug', False)
    kwargs = {k: kwargs[k] for k in kwargs if k in settings}
    frame_idx = 0  # счетчик обработанных кадров
    avg_counter, laser_tangent, laser_row_pos = 0, 0, 0  # метрики стабильности нулевой линии лазера
    zero_level = None  # переменная для нулевой линии
    fine_laser_center = None  # переменная для лазера
    if row_start >= row_stop and col_start >= col_stop:
        raise Exception('Incorrect bounds of image. min_row should be strictly less then max_row.')
    height_map = np.zeros((TOTAL_FRAMES - initial_frame_idx, col_stop - col_start, 3), dtype='float32')  # карта высот
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_idx)  # читать видео с кадра initialFrameIdx
    start = time.time()
    while cap.isOpened():  # пока видео открыто
        ret, frame = cap.read()
        if ret is True:  # пока кадры есть - сканировать
            roi = frame[row_start:row_stop, col_start:col_stop]  # обрезать кадр по зоне интереса
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # конвертировать в грейскейл
            if colored:  # сделать маску для выделения зоны лазера в кадре
                blur = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                mask = get_hsv_mask(blur, hsv_upper_bound, hsv_lower_bound)
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            else:
                blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
                if THRESH_VALUE > 0:
                    _, mask = cv2.threshold(blur, THRESH_VALUE, 255, cv2.THRESH_BINARY)
                else:
                    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ############################################################################################################
            # ВЫБОР МЕТОДА ПОИСКА ЛАЗЕРА
            ############################################################################################################
            if EXTRACTION_MODE == 'max_peak':
                blur = cv2.bitwise_and(blur, blur, mask=mask)
                fine_laser_center = predict_laser(blur, row_start, row_stop)
            elif EXTRACTION_MODE == 'log':
                derivative = laplace_of_gauss(gray, ksize, sigma)  # выделить точки похожие на лазер
                derivative = cv2.bitwise_and(derivative, derivative, mask=mask)  # убрать всё что точно не лазер
                derivative[derivative < 0] = 0  # отрицательная производная точно не лазер
                fine_laser_center = predict_laser(derivative, row_start,
                                                  row_stop)  # расчитать субпиксельные позиции лазера
            elif EXTRACTION_MODE == 'ggm':
                blur = cv2.bitwise_and(blur, blur, mask=mask)
                fine_laser_center = gray_gravity(blur, row_start, row_stop)
            elif EXTRACTION_MODE == 'iggm':
                pass
            else:
                raise Error(f'Unknown extraction mode {EXTRACTION_MODE}')
            ############################################################################################################
            # если по производной положений лазера есть всплески отклонение которых от среднего больше чем пять
            # среднеквадратичных, то считать эту точку невалидной и занулить её
            fine_laser_center_deriv = cv2.Sobel(fine_laser_center, -1, 0, 1, None, 1).flatten()
            fine_laser_center[
                abs(fine_laser_center_deriv.mean() - fine_laser_center_deriv) > 5 * fine_laser_center_deriv.std()] = 0
            fine_laser_center = np.pad(fine_laser_center, (col_start, FRAME_WIDTH - col_stop), 'constant')
            ############################################################################################################
            # РАСЧЁТ ПОЛОЖЕНИЯ И УГЛА НУЛЕВОЙ ЛИНИИ #
            ############################################################################################################
            if AVG_TIME <= 0:  # если не задано усреднять лазер, то считать нулевой уровень в каждом кадре
                zero_level, _ = predict_zero_level(fine_laser_center, FRAME_HEIGHT // 2 - 1)
            elif avg_counter < AVG_TIME:  # если задано усреднять и лазер ещё не усреднён
                # найти нулевую линию, её угол и отклонение от предыдущего значения
                zero_level, tangent = predict_zero_level(fine_laser_center, FRAME_HEIGHT / 2 - 1)
                angle_error = np.abs(np.arctan(laser_tangent) - np.arctan(tangent))
                pos_error = abs(laser_row_pos - zero_level[0])
                #  если параметры линии отклоняются в допустимых пределах
                if angle_error < LASER_ANGLE_TOLERANCE and pos_error < LASER_POS_TOLERANCE:
                    avg_counter += 1  # расчитать средние параметры линии по кадрам
                    laser_row_pos = (laser_row_pos * (avg_counter - 1) + zero_level[0]) / avg_counter
                    laser_tangent = (laser_tangent * (avg_counter - 1) + tangent) / avg_counter
                else:  # иначе принять найденные параметры за новые и обнулить счётчик
                    laser_row_pos, laser_tangent, avg_counter = zero_level[0], tangent, 0
                # TODO: вставить предупреждение если лазер долго нестабилен
                # расчитать нулевой уровень по расчитанным параметрам и обрезать по roi
                zero_level = np.array([x * laser_tangent + zero_level[0] for x in range(fine_laser_center.size)])
            zero_level[(zero_level < row_start) | (zero_level > row_stop - 1)] = row_stop - 1
            ############################################################################################################
            # занулить точки где положение "лазера" ниже нулевой линии
            fine_laser_center[fine_laser_center < zero_level] = zero_level[fine_laser_center < zero_level]
            try:  # расчитать физические координаты точек лазера
                height_map[frame_idx] = find_coords(frame_idx, fine_laser_center, zero_level, frame_shape=FRAME_SHAPE,
                                                    reverse=REVERSE, mirrored=MIRRORED, **kwargs)[col_start:col_stop]
            except OutOfScanArea:  # если точки вне зоны, значит закончить обработку
                cap.release()  # закрыть видео
                print('достигнута граница зоны сканирования')
            print(
                f'{frame_idx + initial_frame_idx + 1:{3}}/{TOTAL_FRAMES:{3}} кадров обрабтано за {time.time() - start:4.2f} с\n'
                f'  X: {height_map[frame_idx][0, X]:4.2f} мм; Zmax: {height_map[frame_idx][:, Z].max():4.2f} мм')
            frame_idx += 1
            ############################################################################################################
            # ВЫВОД НА ЭКРАН ИЗОБРАЖЕНИЙ ДЛЯ ОТЛАДКИ
            ############################################################################################################
            if DEBUG:
                frame[row_start:row_stop, [col_start, col_stop - 1]] = (255, 0, 255)
                frame[[row_start, row_stop - 1], col_start:col_stop] = (255, 0, 255)
                frame[fine_laser_center.astype(np.int)[col_start:col_stop], np.mgrid[col_start:col_stop]] = (0, 255, 0)
                frame[zero_level.astype(np.int)[col_start:col_stop], np.mgrid[col_start:col_stop]] = (255, 0, 0)
                cv2.circle(frame, (fine_laser_center.argmax(), int(np.amax(fine_laser_center))), 3,
                           (0, 0, 255), -1)
                cv2.imshow('frame', frame)
                cv2.imshow('mask', mask)
                cv2.imshow('height map', height_map.copy()[..., Z] / np.amax(height_map[..., Z]))
                cv2.waitKey(15)
            ############################################################################################################
        else:  # кадры кончились или побиты(?)
            cap.release()  # закрыть видео
    else:  # когда видео кончилось
        time_passed = time.time() - start
        print(f'Готово. Потрачено времени на анализ рельефа: {time_passed:3.2f} с\n')
        height_map[..., Z][height_map[..., Z] < 0] = 0  # везде где Z < 0 приравнять к нулю
        if REVERSE:  # если скан с конца области, ориентировать массив соответственно
            height_map = np.flipud(height_map)
        if MIRRORED:  # если скан зеркальный, ориентировать массив соответственно
            height_map = np.fliplr(height_map)
        return height_map


def scan(path_to_video: str, colored: bool = False, **kwargs):
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
    :keyword start_thresh: параметр для детекта начала. start_thresh < 0 без детекта.
                    если видео цветное то поиск по мастер маске и 0 < start_thresh < 1 - степень схожести
                    если видео чб то поиск по пропаже/появлению линии и start_thresh - минимально количество точек на линии
    :keyword mirrored: ориентация сканирования. 0 слева - False, 0 справа - True
    :keyword reverse: направление сканирования. от нуля - False, к нулю - True
    :keyword debug: флаг отладки для функций
    :return: None
    """
    col_start = 30
    col_stop = 610
    row_start = 100
    row_stop = 400

    settings_values = get_settings_values(
        **{k: settings_sections[k] for k in settings if k in settings_sections})  # параметры из конфига

    cap = cv2.VideoCapture(path_to_video)  # чтение видео
    calibrate_kx(cap.get(cv2.CAP_PROP_FPS))  # откалибровать kx согласно FPS
    if 'roi' not in kwargs:
        kwargs.update({'roi': (row_start, row_stop, col_start, col_stop)})

    kwargs = {**settings_values, **kwargs}
    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    if colored:
        # TODO: либо отказаться от цвета совсем, либо уже придумать что то
        start_thresh = kwargs.pop('start_thresh', 104)
        detector = detect_start3(cap, start_thresh)
    else:
        start_thresh = kwargs.pop('start_thresh', 104)
        detector = detect_start3(cap, start_thresh)
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
    height_map = scanning(cap, initial_frame_idx, colored=colored, **kwargs)
    globalValues.height_map = height_map

    # массив для нахождения позиций объектов
    height_map_z = height_map[..., Z]
    height_map_8bit = (height_map_z / np.amax(height_map_z) * 255).astype(np.uint8)

    height_map_8bit[height_map_z < 1] = 0
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
