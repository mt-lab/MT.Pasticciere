"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
from numpy import cos, sin, tan, pi, arctan
import cv2
from typing import Tuple
from utilities import save_height_map, OutOfScanArea, Error, mid_idx, print_objects
import globalValues
from globalValues import get_settings_values, settings_sections
from cookie import *
import time

# TODO: написать логи

# масштабные коэффициенты для построения облака точек
kx = 1 / 3  # мм/кадр

settings = ['hsv_upper_bound',
            'hsv_lower_bound',
            'reverse',
            'mirrored',
            'extraction_mode',
            'avg_time',
            'laser_angle_tol',
            'laser_pos_tol',
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


def find_coords(frame_idx: int, laser: np.ndarray, zero_level: np.ndarray,
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
    :param np.ndarray laser: массив длины frame_shape[1], позиции лазера
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
    # TODO: посчитать расстояние между лазером и соплом
    # TODO: добавить переключение взаимного расположения камеры и лазера (добавлять координату или убавлять)
    row_mid, col_mid = frame_shape[0] / 2 - 1, frame_shape[1] / 2 - 1
    row = mid_idx(zero_level)[0]  # найти центр лазера
    if isinstance(row, np.ndarray):
        row = row.mean()
    # найти высоту камеры по центральной точке
    dpy0 = (row - row_mid) * pixel_size  # отклонение середины нулевой линии от центра кадра по вертикали
    gamma = arctan(dpy0 / focal_length)  # угол отклонения середины нулевой линии от оси камеры
    camera_height = distance_camera2laser / tan(camera_angle + gamma)  # высота камеры
    dpy0 = (zero_level - row_mid) * pixel_size  # массив отклонений точек нулевой линии от центра кадра по вертикали
    dpy = (laser - row_mid) * pixel_size  # массив отклонений точек лазера от центра кадра по вертикале
    dpx = (np.mgrid[:laser.size] - col_mid) * pixel_size  # массив отклоненний точек лазера от центра по горизонтали
    gamma = arctan(dpy0 / focal_length)  # массив углов отклонения нулевой линии от оси камеры
    theta = arctan(dpy / focal_length)  # массив углом отклонения точек лазера от оси камеры
    z = camera_height * sin(theta - gamma) / (sin(camera_angle + theta) * cos(camera_angle + gamma))  # высоты
    z[(z < 0) | (z > table_height)] = 0  # всё, что вне диапазона занулить
    y = (camera_height - z) * dpx / (focal_length * cos(camera_angle))  # массив Y координат
    y = camera_shift + y if not mirrored else camera_shift - y  # если зеркально, то отзеркалить
    x = frame_idx * kx - camera_height * tan(camera_angle + gamma)  # массив X координат
    x = x if not reverse else - x  # развернуть если скан наоборот
    if np.any(abs(x) > table_length):  # проверка конца сканирования
        raise OutOfScanArea
    return np.column_stack([x + x0, y + y0, z + z0])


def find_laser_center(p=(0, 0), m=(0, 0), n=(0, 0)) -> Tuple[float, float]:
    """
    Аппроксимирует по трём точкам параболу и находит её вершину
    Таким образом более точно находит позицию лазера в изображении

    :param Tuple[int, float] p: предыдущая точка от m (m-1)
    :param Tuple[int, float] m: точка с максимальной интенсивностью, (ряд, интенсивность)
    :param Tuple[int, float] n: следующая точка от m (m+1)
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
    ggm = img.copy().astype(np.float32) / np.amax(img)
    centers = np.sum(ggm * (np.mgrid[:ggm.shape[0], :ggm.shape[1]][0] + 1), axis=0) / np.sum(ggm,
                                                                                             axis=0) + row_start - 1
    centers[centers > row_stop - 1] = row_stop - 1
    centers[np.isinf(centers) | np.isnan(centers)] = 0
    return centers


def predict_laser(deriv: np.ndarray, row_start=0, row_stop=None) -> np.ndarray:
    # TODO: написать варианты не использующие LoG:
    #       3. применять IGGM (возможно замедление работы алгоритма)
    """

    :param deriv: preprocessed img
    :param row_start: минимально возможный ряд
    :param row_stop: максимально возможный ряд
    :return fine_laser: list of predicted laser subpixel positions
    """
    laser = np.argmax(deriv, axis=0)
    laser[laser > (row_stop - row_start - 1)] = 0
    fine_laser = np.zeros(laser.shape)
    for column, row in enumerate(laser):
        if row == 0:
            continue
        prevRow = row - 1
        nextRow = row + 1 if row < deriv.shape[0] - 1 else deriv.shape[0] - 1
        p1 = (1. * prevRow, 1. * deriv[prevRow, column])
        p2 = (1. * row, 1. * deriv[row, column])
        p3 = (1. * nextRow, 1. * deriv[nextRow, column])
        fine_laser[column] = find_laser_center(p1, p2, p3)[0] + row_start
    fine_laser[fine_laser > row_stop - 1] = row_stop
    return fine_laser


def predict_zero_level(laser: np.ndarray, mid_row: Union[int, float] = 239, col_start=0, col_stop=640, padl=50,
                       padr=50) -> Tuple[np.ndarray, float]:
    """
    Расчитывает положение нулевой линии и её угол по крайним точкам из массива

    :param np.ndarray laser: массив точек описывающих положение лазера
    :param mid_row: средний ряд кадра, значение по умолчанию если расчёт не получится
    :return: массив точек нулевой линии и тангенс наклона линии от горизонтали
    """
    if padl < 0 or padr < 0:
        raise Error
    data_x = laser.nonzero()[0]
    data_x = data_x[(data_x < col_start + padl) | (data_x > col_stop - padr)]
    if data_x.size:
        data_y = laser[data_x]
        k = np.polyfit(data_x, data_y, 1)
        line = np.poly1d(k)
        zero_level = line(np.mgrid[:laser.size])
        tangent = k[0]
    else:
        zero_level = np.full_like(laser, mid_row)
        tangent = 0
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


def detect_start3(cap, threshhold=50, roi=None, verbosity=0, debug=False):
    if threshhold < 0:
        yield True
    start = False
    mirror = False
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    angle_tol = tan(1 / 180 * pi)
    firstLine = False
    row_start, row_stop, col_start, col_stop = roi if roi is not None else (0, FRAME_HEIGHT, 0, FRAME_WIDTH)
    if row_start >= row_stop and col_start >= col_stop:
        raise Error('Incorrect bounds of image. row_start should be strictly less then row_stop.')
    while True:
        frameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret is not True or not cap.isOpened():
            yield -1
        roi = frame[row_start:row_stop, col_start:col_stop]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (33, 33), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        derivative = laplace_of_gauss(gray, 29, 4.45)  # выделить точки похожие на лазер
        derivative = cv2.bitwise_and(derivative, derivative, mask=mask)  # убрать всё что точно не лазер
        derivative[derivative < 0] = 0  # отрицательная производная точно не лазер
        laser = predict_laser(derivative, row_start, row_stop)  # расчитать позиции лазера
        thresh = np.zeros(frame.shape[:2], dtype='uint8')
        thresh[laser.astype(int), np.mgrid[col_start:col_stop]] = 255
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        # TODO: проверку стабильности пропажи и появления лазера
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshhold, None, roi.shape[1] * 0.9, 10)
        if lines is not None:
            for count, line in enumerate(lines):
                for x1, y1, x2, y2 in line:
                    if (y2 - y1) / (x2 - x1) > angle_tol:
                        del lines[count]
                        continue
                    #####################################################
                    # """ for debug purposes """
                    if debug:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #################################################################
        # """ for debug purposes """
        if debug:
            cv2.imshow('thresh', thresh)
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.waitKey(15)
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
            print(f'{frameIdx + 1:{3}}/{TOTAL_FRAMES:{3}} кадр. Начало зоны сканирования')
            cv2.destroyAllWindows()
            yield True
        if verbosity >= 1:
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
        :keyword verbosity: подробность вывода информации
            (default) 0 - только конечный результат
                      1 - + сколько кадров обработано
                      2 - + координаты и время обработки
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
    :return: карту высот формы (TOTAL_FRAMES, col_stop - col_start, 3), где каждый пиксель это [X, Y, Z] координата
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
    zero_level_padl, zero_level_padr = kwargs.pop('zero_level_zone', (10, 10))
    debug = kwargs.pop('debug', False)
    kwargs = {k: kwargs[k] for k in kwargs if k in settings_sections}
    frame_idx = 0  # счетчик обработанных кадров
    avg_counter, laser_tangent, laser_row_pos = 0, 0, 0  # метрики стабильности нулевой линии лазера
    zero_level = None  # переменная для нулевой линии
    if row_start >= row_stop and col_start >= col_stop:
        raise Exception('Incorrect bounds of image. row_start should be strictly less then row_stop.')
    height_map = np.zeros((TOTAL_FRAMES - initial_frame_idx, col_stop - col_start, 3), dtype='float32')  # карта высот
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_idx)  # читать видео с кадра initialFrameIdx
    print('Идёт обработка данных...')
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
                if THRESH_VALUE == 0:
                    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif THRESH_VALUE > 0:
                    _, mask = cv2.threshold(blur, THRESH_VALUE, 255, cv2.THRESH_BINARY)
                else:
                    mask = np.full_like(blur, 255, np.uint8)
            ############################################################################################################
            # ВЫБОР МЕТОДА ПОИСКА ЛАЗЕРА
            ############################################################################################################
            if EXTRACTION_MODE == 'max_peak':
                blur = cv2.bitwise_and(blur, blur, mask=mask)
                laser = predict_laser(blur, row_start, row_stop)
            elif EXTRACTION_MODE == 'log':
                derivative = laplace_of_gauss(gray, ksize, sigma)  # выделить точки похожие на лазер
                derivative = cv2.bitwise_and(derivative, derivative, mask=mask)  # убрать всё что точно не лазер
                derivative[derivative < 0] = 0  # отрицательная производная точно не лазер
                laser = predict_laser(derivative, row_start, row_stop)  # расчитать субпиксельные позиции лазера
            elif EXTRACTION_MODE == 'ggm':
                blur = cv2.bitwise_and(blur, blur, mask=mask)
                laser = gray_gravity(blur, row_start, row_stop)
            elif EXTRACTION_MODE == 'iggm':
                raise NotImplemented('IGGM laser extraction.')
            else:
                raise Error(f'Unknown extraction mode {EXTRACTION_MODE}')
            ############################################################################################################
            # если по производной положений лазера есть всплески отклонение которых от среднего больше чем пять
            # среднеквадратичных, то считать эту точку невалидной и занулить её
            laser_deriv = cv2.Sobel(laser, -1, 0, 1, None, 1).flatten()
            laser[abs(laser_deriv.mean() - laser_deriv) > 5 * laser_deriv.std()] = 0
            # сделать паддинг для правильного нахождения Y координаты в дальнейшем
            laser = np.pad(laser, (col_start, FRAME_WIDTH - col_stop), 'constant')
            ############################################################################################################
            # РАСЧЁТ ПОЛОЖЕНИЯ И УГЛА НУЛЕВОЙ ЛИНИИ #
            ############################################################################################################
            if AVG_TIME <= 0:  # если не задано усреднять лазер, то считать нулевой уровень в каждом кадре
                zero_level, _ = predict_zero_level(laser, FRAME_HEIGHT // 2 - 1, col_start, col_stop,
                                                   zero_level_padl, zero_level_padr)
            elif avg_counter < AVG_TIME:  # если задано усреднять и лазер ещё не усреднён
                # найти нулевую линию, её угол и отклонение от предыдущего значения
                zero_level, tangent = predict_zero_level(laser, FRAME_HEIGHT // 2 - 1, col_start, col_stop,
                                                         zero_level_padl, zero_level_padr)
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
                zero_level = np.mgrid[:laser.size] * laser_tangent + laser_row_pos
            zero_level[(zero_level < row_start) | (zero_level > row_stop - 1)] = row_stop - 1
            ############################################################################################################
            # занулить точки где положение "лазера" ниже нулевой линии
            laser[laser < zero_level] = zero_level[laser < zero_level]
            try:  # расчитать физические координаты точек лазера
                height_map[frame_idx] = find_coords(frame_idx, laser, zero_level, frame_shape=FRAME_SHAPE,
                                                    reverse=REVERSE, mirrored=MIRRORED, **kwargs)[col_start:col_stop]
            except OutOfScanArea:  # если точки вне зоны, значит закончить обработку
                height_map = height_map[:frame_idx]
                cap.release()  # закрыть видео
                print('Достигнута граница зоны сканирования')
            if verbosity == 1:
                print(f'{frame_idx + initial_frame_idx + 1:{3}}/{TOTAL_FRAMES:{3}} кадров обрабтано')
            elif verbosity == 2:
                print(
                    f'{frame_idx + initial_frame_idx + 1:{3}}/{TOTAL_FRAMES:{3}} кадров обрабтано за {time.time() - start:4.2f} с\n'
                    f'  X: {height_map[frame_idx][0, X]:4.2f} мм; Zmax: {np.amax(height_map[frame_idx][:, Z]):4.2f} мм')
            frame_idx += 1
            ############################################################################################################
            # ВЫВОД НА ЭКРАН ИЗОБРАЖЕНИЙ ДЛЯ ОТЛАДКИ
            ############################################################################################################
            if debug:
                frame[row_start:row_stop, [col_start, col_stop - 1]] = (255, 0, 255)
                frame[row_start:row_stop, [col_start + zero_level_padl, col_stop - zero_level_padr]] = (127, 255, 127)
                frame[[row_start, row_stop - 1], col_start:col_stop] = (255, 0, 255)
                frame[laser.astype(np.int)[col_start:col_stop], np.mgrid[col_start:col_stop]] = (0, 255, 0)
                frame[zero_level.astype(np.int)[col_start:col_stop], np.mgrid[col_start:col_stop]] = (255, 0, 0)
                cv2.circle(frame, (laser.argmax(), int(np.amax(laser))), 3,
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


def scan(path_to_video: str, colored: bool = False, debug=False, verbosity=0, **kwargs):
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

    settings_values = get_settings_values(**{k: settings_sections[k] for k in settings_sections if
                                             settings_sections[k][0] in ['Scanner', 'Camera',
                                                                         'Table']})  # параметры из конфига

    cap = cv2.VideoCapture(path_to_video)  # чтение видео
    calibrate_kx(cap.get(cv2.CAP_PROP_FPS))  # откалибровать kx согласно FPS
    if 'roi' not in kwargs:
        kwargs.update({'roi': (row_start, row_stop, col_start, col_stop)})

    kwargs = {**settings_values, **kwargs}
    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    start_thresh = kwargs.pop('start_thresh', 104)
    detector = detect_start3(cap, start_thresh, verbosity=verbosity, debug=debug)
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
    height_map = scanning(cap, initial_frame_idx, colored=colored, verbosity=verbosity, debug=debug, **kwargs)
    globalValues.height_map = height_map

    # массив для нахождения позиций объектов
    height_map_z = height_map[..., Z]
    height_map_8bit = (height_map_z / np.amax(height_map_z) * 255).astype(np.uint8)

    height_map_8bit[height_map_z < 1] = 0
    factory = np.abs(height_map[0, 0, Y] - height_map[0, -1, Y]) / height_map.shape[1]
    factorx = np.abs(height_map[0, 0, X] - height_map[-1, 0, X]) / height_map.shape[0]
    scale_factor = factory / factorx
    height_map_8bit_real = cv2.resize(height_map_8bit, None, fx=scale_factor, fy=1)
    cookies, detected_contours = find_cookies(height_map_8bit, height_map)
    cookies, detected_contours = procecc_cookies(cookies, height_map, img=detected_contours)
    detected_contours_real = cv2.resize(detected_contours, None, fx=scale_factor, fy=1)
    print('Положения печений найдены.')
    if len(cookies) != 0:
        globalValues.cookies = cookies
        print_objects(cookies, f'Объектов найдено: {len(cookies):{3}}')
        print()

    # сохранить карты
    cv2.imwrite('height_map.png', height_map_8bit)
    cv2.imwrite('height_map_real.png', height_map_8bit_real)
    cv2.imwrite('cookies_real.png', detected_contours_real)
    cv2.imwrite('cookies.png', detected_contours)
    save_height_map(height_map)

    cap.release()
    cv2.destroyAllWindows()
