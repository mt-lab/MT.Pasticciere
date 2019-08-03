"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
import cv2
from configValues import focal, pxlSize, cameraAngle, cameraHeight, tableWidth, tableLength, tableHeight, X0, Y0, Z0, \
    hsvLowerBound, hsvUpperBound, markPicture, markCenter, accuracy, VID_PATH
from math import atan, sin, cos, pi, sqrt, tan
from utilities import X, Y, Z, distance
import globalValues
from cookie import *
import time
import imutils

# TODO: написать логи

# TODO: комментарии и рефактор

# TODO: для постобработки облака взять значения внутри найденных контуров (используя маску), найти среднее и отклонения
#       и обрезать всё что выше mean + std (или 2*std)


# масштабные коэффициенты для построения облака точек
# Kz = 9 / 22  # мм/пиксель // теперь расчёт по формуле
Kx = 1 / 3  # мм/кадр // уточнить коэффициент, по хорошему должно быть tableLength/(frameCount-initialFrameIdx)
# Ky = 100 / 447  # мм/пиксель // теперь расчёт по формуле

# ширина изображения для обработки, пиксели
Xnull = 0
Xend = 640
startMask = cv2.imread('startMask.png', 0)


def findDistanceToLaser(midPoint=239, zeroLevel=239):
    theta = atan((zeroLevel - midPoint) * pxlSize / focal)  # угол смещения нулевого уровня
    distanceToLaser = cameraHeight / cos(cameraAngle + theta)  # расстояние от линзы до лазера
    return distanceToLaser, theta


def calculateZ(pxl, midPoint=239, zeroLevel=239, distanceToLaser=None, theta=None):
    """
    Функция расчета положения точки по оси Z относительно уровня стола
    :param pxl: номер пикселя по вертикали на картинке
    :param midPoint: номер серединного пикселя по вертикали
    :return height: высота точки
    """
    dp = (pxl - midPoint) * pxlSize
    dp0 = (zeroLevel - midPoint) * pxlSize
    theta = atan(dp / focal)
    beta = atan(dp0 / focal)
    phi = theta - beta
    alpha = cameraAngle
    height = cameraHeight * sin(phi) / (sin(alpha + beta + phi) * cos(alpha + beta))
    # if distanceToLaser is None or theta is None:
    #     distanceToLaser, theta = findDistanceToLaser(midPoint, zeroLevel)
    # chi = atan((pxl - midPoint) * pxlSize / focal)  # угловая координата пикселя
    # phi = chi - theta  # угол изменения высоты
    # beta = pi / 2 - cameraAngle - theta  # угол между плоскостью стола и прямой между линзой и лазером
    # height = distanceToLaser * sin(phi) / cos(beta - phi)
    return height


def calculateY(pxl, z=0.0, columnMidPoint=319, rowMidPoint=239, midWidth=tableWidth / 2, zeroLevel=239,
               distanceToLaser=None):
    """
    Функция расчета положения точки по оси Y относительно середины обзора камеры (соответственно середины стола)
    :param pxl: номер пикселя по горизонтали на картинке
    :param z: высота, на которой находится точка
    :param columnMidPoint: номер серединного пикселя по горизонтали
    :param midWidth: расстояние до середины стола
    :return width: расстояние до точки от начала стола
    """
    if distanceToLaser is None:
        distanceToLaser, _ = findDistanceToLaser(rowMidPoint, zeroLevel)
    dW = (pxl - columnMidPoint) * pxlSize * (distanceToLaser - z * cameraHeight / distanceToLaser) / focal
    width = midWidth + dW
    return width


def calculateX(frameIdx):
    return frameIdx * Kx


def calculateCoordinates(frameIdx=0, pixelCoordinate=(0, 0), rowMidPoint=239, columnMidPoint=319, zeroLevel=239,
                         midWidth=tableWidth / 2, distanceToLaser=None, theta=None):
    row = pixelCoordinate[0]
    column = pixelCoordinate[1]

    if distanceToLaser is None or theta is None:
        distanceToLaser, theta = findDistanceToLaser(rowMidPoint, zeroLevel)

    height = calculateZ(row, rowMidPoint, zeroLevel, distanceToLaser, theta)  # высота точки относительно стола

    width = calculateY(column, height, columnMidPoint, midWidth,
                       distanceToLaser)  # координата Y относительно начала стола

    length = calculateX(frameIdx)  # координата точки по X относительно начала стола

    return (length, width, height)


def findLaserCenter(prev=(0, 0), middle=(0, 0), next=(0, 0), default=(240.0, 0)):
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
    return (xc, yc)


def LoG(img, ksize, sigma, delta=0.0):
    kernelX = cv2.getGaussianKernel(ksize, sigma)
    kernelY = kernelX.T
    gauss = -cv2.sepFilter2D(img, cv2.CV_64F, kernelX, kernelY, delta=delta)
    laplace = cv2.Laplacian(gauss, cv2.CV_64F)
    return laplace


def calibrateKx(videoFPS: 'frame per sec', printerVelocity: 'mm per minute' = 300):
    """
    Функция калибровки коэффициента Kx
    :param videoFPS:
    :param printerVelocity:
    :return:
    """
    kx = (printerVelocity / 60) / (videoFPS / 2)
    return kx


def generatePly(pointsArray, filename='cloud.ply'):
    """
    Генерирует файл облака точек

    :param pointsArray - массив точек с координатами
    :param filename - имя файла для записи, по умолчанию cloud.ply
    :return: None
    """
    print('Generating point cloud...')
    start = time.time()
    ply = []
    # если точка лежит по высоте за пределами рабочей зоны, включая нулевой уровень, то пропустить точку
    for count, point in enumerate(pointsArray, 1):
        if point[Z] <= Z0 or point[Z] > (Z0 + tableHeight):
            continue
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
    timePassed = time.time() - start
    print(f'Done for {timePassed:3.2f} sec\n')


def findZeroLevel(img):
    """
    Находит индекс строки на изображении с максимальным количеством белых пикселей,
    то есть нулевой уровень лазера на изображении

    :param img - изображение
    :return: индекс строки
    """
    rowSum = img.sum(axis=1)
    return np.argmax(rowSum)


def lineThinner(img, upperBound=0):
    """
    Оставляет нижний край лазера на изображении в виде линии толщиной в один пиксель

    :param img: исходное изображение
    :param upperBound: верхняя граница, выше которой алгоритм применять бессмысленно
    :return: полученное изображение
    """
    newImg = np.zeros(img.shape, dtype="uint8")
    for x in range(img.shape[1]):
        for y in range(img.shape[0] - 1, upperBound, -1):
            if (img.item(y, x) == 255) and (img.item(y, x) == img.item(y - 1, x)):
                newImg.itemset((y, x), 255)
                break
    return newImg


def getMask(img, zero_level=0):
    """
    Делает битовую маску лазера с изображения и применяет к ней lineThinner

    :param img: исходное изображение
    :param zero_level:
    :return: изображение после обработки
    """
    img = img[zero_level:, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsvLowerBound), np.array(hsvUpperBound))
    gauss = cv2.GaussianBlur(mask, (5, 5), 0)
    ret2, gaussThresh = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gaussThin = lineThinner(gaussThresh)
    return gaussThin


def findCookies(imgOrPath, heightMap=None, distanceToLaser=cameraHeight / cos(cameraAngle)):
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
    if isinstance(imgOrPath, str):
        original = cv2.imread(imgOrPath)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif isinstance(imgOrPath, np.ndarray):
        if imgOrPath.ndim == 3:
            original = imgOrPath.copy()
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif imgOrPath.ndim == 2:
            original = cv2.merge((imgOrPath.copy(), imgOrPath.copy(), imgOrPath.copy()))
            gray = imgOrPath.copy()
    else:
        return 'Вы передали какую то дичь'

    if heightMap is None:
        heightMap = gray.copy() / 10

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
    cntParm = []
    for contour in contours:
        tmp = contour.copy()  # скопировать контур, чтобы не изменять оригинал
        tmp = np.float32(tmp)
        # перевести из пикселей в мм
        for point in tmp:
            px = int(point[0][1])
            py = int(point[0][0])
            point[0][1] = calculateX(px) + X0
            point[0][0] = calculateY(py, z=heightMap[px, py], distanceToLaser=distanceToLaser) + Y0
        moments = cv2.moments(tmp)
        # найти центр контура и записать его в СК принтера
        M = cv2.moments(contour)
        Cx = int(M['m10'] / M['m00'])
        Cy = int(M['m01'] / M['m00'])
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        centerHeight = heightMap[Cy, Cx]
        center = (cy, cx)
        # найти угол поворота контура (главная ось)
        a = moments['m20'] / moments['m00'] - cx ** 2
        b = 2 * (moments['m11'] / moments['m00'] - cx * cy)
        c = moments['m02'] / moments['m00'] - cy ** 2
        theta = 1 / 2 * atan(b / (a - c)) + (a < c) * pi / 2
        # угол поворота с учетом приведения в СК принтера
        rotation = theta + pi / 2
        cntParm.append((center, centerHeight, rotation))
    cookies = []
    for contour in cntParm:
        centerHeight = contour[1]
        center = contour[0]
        rotation = contour[2]
        cookies.append(Cookie(center=center, centerHeight=centerHeight, rotation=rotation))

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


def normalize(img):
    array = img.copy().astype(np.float64)
    array = (array - array.min()) / (array.max() - array.min())
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


def detectStart2(cap, contourPath='', threshold=0.5):
    # копия findCookies заточеная под поиск конкретного контура и его положение с целью привязки к глобальной СК
    # работает сразу с видео потоком по принципу detectStart()
    # TODO: разделить на функции, подумать как обобщить вместе с findCookies()

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
                if distance(markCenter, candidateCenter) <= 2:
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh', thresh)
        thresh = skeletonize(thresh)
        lines = cv2.HoughLines(thresh, 1, np.pi / 2, sensitivity)
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow('skeleton', frame)
        # cv2.waitKey(15)
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


def scanning(cap, initialFrameIdx=0, tolerance=0.1):
    # TODO: отделить наполнение массива облака точек от наполнения карты высот для более удобной постобработки

    # читать видео с кадра initialFrameIdx
    cap.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIdx)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameIdx = 0
    # количество точек в облаке
    numberOfPoints = (Xend - Xnull) * (totalFrames - initialFrameIdx)
    pointNumber = 0
    # массив с координатами точек в облаке
    ply = np.zeros((numberOfPoints, 3))
    # карта высот
    heightMap = np.zeros((totalFrames - initialFrameIdx, Xend - Xnull), dtype='float16')
    zeroLevel = 240  # ряд пикселей принимаемый за ноль высоты
    row_max = 0
    global Kx
    Kx = calibrateKx(cap.get(cv2.CAP_PROP_FPS))
    ksize = 29
    sigma = 4.45
    distanceToLaser = cameraHeight / cos(cameraAngle)
    theta = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        # пока кадры есть - сканировать
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            derivative = LoG(gray, ksize, sigma)
            _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            derivative = cv2.bitwise_and(derivative, derivative, mask=mask)
            derivative[:240, :] = 0
            apprxLaserCenter = np.argmax(derivative, axis=0)
            fineLaserCenter = np.zeros(apprxLaserCenter.shape)
            for column, row in enumerate(apprxLaserCenter):
                prevRow = row - 1 if row > 0 else 0
                nextRow = row + 1 if row < frame.shape[0] - 1 else frame.shape[0] - 1
                p1 = (1.0 * prevRow, derivative[prevRow, column])
                p2 = (1.0 * row, derivative[row, column])
                p3 = (1.0 * nextRow, derivative[nextRow, column])
                fineLaserCenter[column], _ = findLaserCenter(p1, p2, p3)
            # print(f'fineLaserZeroLvl {(fineLaserCenter[0] + fineLaserCenter[-1]) / 2}')
            row_max = calculateZ(fineLaserCenter.max(), zeroLevel=zeroLevel, distanceToLaser=distanceToLaser,
                                 theta=theta)
            print(f'max_row {row_max}')
            # при первом кадре найти нулевой уровень и соответствующее смещение по Z
            # if frameIdx + initialFrameIdx == initialFrameIdx:
            #     zeroLevel = fineLaserCenter.mean()
            #     if zeroLevel < 240:
            #         initialFrameIdx +=1
            #         continue
            #     distanceToLaser, theta = findDistanceToLaser(zeroLevel=zeroLevel)
            #     print(
            #         f'Ряд соответствующий нулевому уровню: {zeroLevel:3.1f} ряд')
            #     cv2.waitKey(0)
            zeroLevel = fineLaserCenter[0]
            distanceToLaser, theta = findDistanceToLaser(zeroLevel=zeroLevel)
            fineLaserCenter[fineLaserCenter < zeroLevel] = zeroLevel
            for column, row in enumerate(fineLaserCenter):
                frame[int(row), column] = (0, 255, 0)
            frame[int(zeroLevel)] = (255, 0, 0)
            frame[int(fineLaserCenter.max())] = (0, 0, 255)
            # cv2.imshow('frame', frame)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(15)
            for column, row in enumerate(fineLaserCenter):
                length, width, height = calculateCoordinates(frameIdx, (row, column), zeroLevel=zeroLevel,
                                                             distanceToLaser=distanceToLaser, theta=theta)
                if row < zeroLevel or zeroLevel < frame.shape[0] / 2:
                    continue
                ply[pointNumber, X] = length + X0
                ply[pointNumber, Y] = width + Y0
                if accuracy <= height and height <= tableHeight:
                    heightMap[frameIdx, column] = height
                    ply[pointNumber, Z] = height + Z0
                elif height > tableHeight:
                    ply[pointNumber, Z] = tableHeight
                else:
                    ply[pointNumber, Z] = Z0
                pointNumber += 1

            print(
                f'{frameIdx + 1 + initialFrameIdx:{3}}/{totalFrames:{3}} processed for {time.time() - start:4.2f} sec')
            frameIdx += 1
        else:
            # когда видео кончилось
            timePassed = time.time() - start
            print(f'Done. Time passed {timePassed:3.2f} sec\n')
            return ply, heightMap, distanceToLaser


def scan(pathToVideo=VID_PATH, sensitivity=104, tolerance=0.1):
    """
    Функция обработки видео (сканирования)
    :param pathToVideo: путь к видео, по умолчанию путь из settings.ini
    :return: None
    """

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео

    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    detector = detectStart3(cap, sensitivity)
    start = next(detector)
    while not start or start == -1:
        if start == -1:
            print('сканирование не удалось')
            cv2.destroyAllWindows()
            return None
        start = next(detector)
    initialFrameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    print(f'Точка начала сканирования: {initialFrameIdx + 1: 3d} кадр')

    # сканировать от найденного кадра до конца
    ply, heightMap, distanceToLaser = scanning(cap, initialFrameIdx, tolerance)
    globalValues.heightMap = heightMap
    globalValues.distanceToLaser = distanceToLaser

    # массив для нахождения позиций объектов
    heightMap8bit = (heightMap * 10).astype(np.uint8)
    cookies, detectedContours = findCookies(heightMap8bit, heightMap, distanceToLaser)
    if len(cookies) != 0:
        for cookie in cookies:
            print(cookie.center, cookie.centerHeight, cookie.rotation)

    # сохранить карты
    cv2.imwrite('height_map.png', heightMap8bit)
    cv2.imwrite('cookies.png', detectedContours)

    # сгенерировать файл облака точек
    generatePly(ply)

    cap.release()
    cv2.destroyAllWindows()
