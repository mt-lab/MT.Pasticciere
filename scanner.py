"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
import cv2
from configValues import focal, pxlSize, cameraAngle, distanceToLaser, tableWidth, tableLength, tableHeight, X0, Y0, Z0, \
    hsvLowerBound, hsvUpperBound, accuracy, VID_PATH
from math import atan, sin, cos
from utilities import X, Y, Z
from cookie import *
import time
import imutils

# TODO: написать логи


# масштабные коэффициенты для построения облака точек
# Kz = 9 / 22  # мм/пиксель // теперь расчёт по формуле
Kx = 74 / 214  # мм/кадр // уточнить коэффициент, по хорошему должно быть tableLength/frameCount
# Ky = 100 / 447  # мм/пиксель // теперь расчёт по формуле

# ширина изображения для обработки, пиксели
Xnull = 0
Xend = 640


def calculateZ(pxl, midPoint=240):
    """
    Функция расчета положения точки по оси Z относительно уровня стола
    :param pxl: номер пикселя по вертикали на картинке
    :param midPoint: номер серединного пикселя по вертикали
    :return height: высота точки
    """
    phi = atan((pxl - midPoint) * pxlSize / focal)
    height = distanceToLaser * sin(phi) / cos(cameraAngle + phi)
    return height


def calculateY(pxl, z=0, midPoint=320, midWidth=tableWidth / 2):
    """
    Функция расчета положения точки по оси Y относительно середины обзора камеры (соответственно середины стола)
    :param pxl: номер пикселя по горизонтали на картинке
    :param z: высота, на которой находится точка
    :param midPoint: номер серединного пикселя по горизонтали
    :param midWidth: расстояние до середины стола
    :return width: расстояние до точки от начала стола
    """
    dW = (pxl - midPoint) * pxlSize * (distanceToLaser - z * cos(cameraAngle)) / focal
    width = midWidth + dW
    return width


def calibrate(video, width: 'in mm', length: 'in mm', height: 'in mm'):
    """
    Функция калибровки коэффициентов
    :param video:
    :param width:
    :param length:
    :param height:
    :return:
    """
    kx = 0
    ky = 0
    kz = 0

    zeroLevel = findZeroLevel()
    shiftZ = calculateZ(zeroLevel)
    return kx, ky, kz, zeroLevel


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
        if point[Z] < Z0 + accuracy or point[Z] > (Z0 + tableHeight) - accuracy:
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


def findCookies(imgOrPath):
    """
    Функция нахождения расположения и габаритов объектов на столе из полученной карты высот
    :param img (np arr, str): карта высот
    :return cookies, result, rectangles, contours: параметры печенек, картинка с визуализацией, параметры боксов
            ограничивающих печеньки, контура границ печенья
    """
    # TODO: подробнее посмотреть происходящее в функции, где то тут баги
    # TODO: отредактировать вывод функции, слишком много всего
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
            pass
    else:
        return 'Вы передали какую то дичь'
    # избавление от минимальных шумов с помощью гауссова фильтра и отсу трешхолда
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, gausThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # нахождение замкнутых объектов на картинке с помощью морфологических алгоритмов
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(gausThresh, cv2.MORPH_OPEN, kernel, iterations=3)
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
    # находим прямоугольники минимальной площади в которые вписываются печеньки
    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    rectanglesCoords = [np.int0(cv2.boxPoints(rect)) for rect in rectangles]
    pic = original.copy()
    for idx, rect in enumerate(rectanglesCoords):
        cv2.drawContours(pic, [rect], 0, (0, 0, 255), 2)
    # определить положение печенек в мм и поворот
    cookies = []
    for rect in rectangles:
        height = gray.item(int(rect[0][Y]), int(rect[0][X])) / 10
        center = (
            rect[0][Y] * Kx + X0, calculateY(rect[0][X], z=height) + Y0)  # позиция печеньки на столе в СК принтера в мм
        width = calculateY(rect[0][X] + rect[1][X] / 2, z=height) - \
                calculateY(rect[0][X] - rect[1][X] / 2, z=height)  # размер печеньки вдоль оси Y в СК принтера в мм
        length = rect[1][Y] * Kx  # размер печеньки вдоль оси X в СК принтера в мм
        rotation = rect[2]  # вращение прямоугольника в углах
        cookies.append(Cookie(center, width, length, height, rotation))
    # сохранить изображение с отмеченными контурами
    cv2.imwrite('cookies.png', pic)
    return cookies, result, rectangles, contours


def compare(img, mask, threshold=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv, np.array([75,92,32]),np.array([95,215,65]))
    # получить чб из img по фильтру
    compare = cv2.bitwise_and(hsv, hsv, mask=mask)
    cv2.imshow('compare', compare)
    similarity = np.sum(compare == 255) / np.sum(mask == 255)
    if similarity >= threshold:
        return True
    else:
        return False


startMask = cv2.imread('/home/bedlamzd/Reps/MT.Pasticciere/startMask1.png',0)


def detectStart(cap, mask, threshold=0.5):
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
        print(f'{frameIdx:{3}}/{frameCount:{3}} frames skipped waiting for starting point')
        yield False


def scanning(cap, initialFrameIdx=0):
    cap.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIdx)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameIdx = initialFrameIdx
    numberOfPoints = (Xend - Xnull) * (frameCount - frameIdx)
    pointIdx = 0
    ply = np.zeros((numberOfPoints, 3))
    heightMap = np.zeros((frameCount - initialFrameIdx, Xend - Xnull), dtype='uint8')
    zeroLevel = 240
    shiftZ = 0
    zmax = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = getMask(frame)
            if frameIdx == initialFrameIdx:
                zeroLevel = findZeroLevel(img)
                shiftZ = calculateZ(zeroLevel)
                print(f'Ряд соответствующий нулевому уровню: {zeroLevel:3d}')
            for imgX in range(Xnull, Xend):
                for imgY in range(zeroLevel, img.shape[0]):
                    # если пиксель белый
                    if img.item(imgY, imgX):
                        # рассчитать соответствующую ему высоту над столом
                        height = calculateZ(imgY) - shiftZ
                        zmax = max(zmax, height)
                        # если высота над столом больше погрешности и находится в пределах рабочей высоты принтера
                        if height > accuracy and height < tableHeight:
                            ply[pointIdx, X] = frameIdx * Kx + X0
                            ply[pointIdx, Y] = calculateY(imgX, z=height) + Y0
                            ply[pointIdx, Z] = height + Z0
                            # заполнение карты высот
                            heightMap[frameIdx, imgX] = round(10 * height) if height < 25.5 else 255
                            break
                        # если высота над столом меньшье погрешности
                        elif height < accuracy:
                            height = ply[pointIdx - 1 if pointIdx > 0 else 0, Z] - Z0
                            ply[pointIdx, X] = frameIdx * Kx + X0
                            ply[pointIdx, Y] = calculateY(imgX, z=height) + Y0
                            ply[pointIdx, Z] = height + Z0
                            # заполнение карты высот
                            heightMap[frameIdx, imgX] = round(10 * height) if height < 25.5 else 255
                            break
                else:
                    # если белых пикселей в стобце нет, то записать нулевой уровень
                    ply[pointIdx, X] = frameIdx * Kx + X0
                    ply[pointIdx, Y] = calculateY(imgX, z=0) + Y0
                    ply[pointIdx, Z] = Z0
                pointIdx += 1
            frameIdx += 1
            print(f'{frameIdx:{3}}/{frameCount:{3}} processed for {(time.time() - start):3.2f} sec')
        else:
            timePassed = time.time() - start
            print(f'Done. Time passed {timePassed:3.2f} sec\n')
            return ply, heightMap


def scan(pathToVideo=VID_PATH, threshold = 0.5):
    """
    Функция обработки видео (сканирования)
    :param pathToVideo: путь к видео, по умолчанию путь из settings.ini
    :return: None
    """

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео

    # найти кадр начала сканирования
    print('Ожидание точки старта...')
    detector = detectStart(cap, startMask, threshold)
    start = next(detector)
    while not start:
        if start == -1:
            return 'сканирование не удалось'
        start = next(detector)
    initialFrameIdx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
    print(f'Точка начала сканирования: {initialFrameIdx+1: 3d} кадр')

    # сканировать от найденного кадра до конца
    ply, heightMap = scanning(cap, initialFrameIdx)

    # раскрасить карту высот
    coloredHeightMap = cv2.applyColorMap(heightMap, cv2.COLORMAP_BONE)
    cookies = findCookies(coloredHeightMap)[0]
    if len(cookies) != 0:
        print(cookies[0].center, cookies[0].height)

    # сохранить карту высот
    cv2.imwrite('colored_height_map.png', coloredHeightMap)

    # сгенерировать файл облака точек
    generatePly(ply)

    cap.release()
    cv2.destroyAllWindows()
