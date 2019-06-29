"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
import cv2
from configValues import hsvLowerBound, hsvUpperBound, accuracy, VID_PATH
from utilities import X, Y, Z
# from open3d import *  # only for visuals
import time
import imutils

# TODO: написать логи

# координаты фактического начала стола относительно глобальных координат принтера в мм
X_0 = 49
Y_0 = 21.5
Z_0 = 5

# максимальная высота на которую может подняться сопло, мм
Z_MAX = 30

# масштабные коэффициенты для построения облака точек
# TODO: сделать автоматический расчёт коэффициентов или привязанный к глобальным параметрам принтера
Kz = 9 / 22  # мм/пиксель
Kx = 74 / 214  # мм/кадр
Ky = 100 / 447  # мм/пиксель

# ширина изображения для обработки, пиксели
Xnull = 0
Xend = 640


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
    return kx, ky, kz


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
    for count, point in enumerate(pointsArray, 1):
        if point[Z] < Z_0 + accuracy or point[Z] > Z_MAX - accuracy:
            continue
        ply.append(f'{point[X]:.3f} {point[Y]:.3f} {point[Z]:.3f}\n')
        # print(f'{count:{6}}/{len(pointsArray):{6}} points processed')
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
            # print(f'{count:{6}}/{len(ply):{6}} points recorded')
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
    # TODO: переписать под обработку "сверху" и "снизу"
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
    gaussThin = lineThinner(gaussThresh, zero_level)
    # cv2.imshow('w', gaussThin)
    # cv2.waitKey(15)
    return gaussThin


def findCookies(imgOrPath='scanned.png'):
    """
    Функция нахождения расположения и габаритов объектов на столе из полученной карты высот
    :param img (np arr, str): карта высот
    :return cookies, result, rectangles, contours: параметры печенек, картинка с визуализацией, параметры боксов
            ограничивающих печеньки, контура границ печенья
    """
    # проверка параметр строка или нет
    if isinstance(imgOrPath, str):
        original = cv2.imread(imgOrPath)
    else:
        original = imgOrPath
    # избавление от минимальных шумов с помощью гауссова фильтра и отсу трешхолда
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
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
    # количество печенек на столе (уникальные маркеры минус фони что то ещё)
    numOfCookies = len(np.unique(markers)) - 2
    # вырезаем ненужный контур всей картинки
    blankSpace = np.zeros(gray.shape, dtype='uint8')
    blankSpace[markers == 1] = 255
    blankSpace = cv2.bitwise_not(blankSpace)
    blankSpaceCropped = blankSpace[2:blankSpace.shape[0] - 2, 2:blankSpace.shape[1] - 2]
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
    for idx, rect in enumerate(rectanglesCoords):
        cv2.drawContours(original, [rect], 0, (0, 0, 255), 2)
    # определить положение печенек в мм и поворот
    cookies = []
    for rect in rectangles:
        center = (rect[0][Y] * Kx + X_0, rect[0][X] * Ky + Y_0)  # позиция печеньки на столе в СК принтера в мм
        width = rect[1][X] * Ky  # размер печеньки вдоль оси Y в СК принтера в мм
        length = rect[1][Y] * Kx  # размер печеньки вдоль оси X в СК принтера в мм
        rotation = rect[2]  # вращение прямоугольника в углах
        cookies.append((center, width, length, rotation))
    # сохранить изображение с отмеченными контурами
    cv2.imwrite('cookies.png', original)
    return cookies, result, rectangles, contours


def scan(pathToVideo=VID_PATH):
    """
    Функция обработки видео (сканирования)
    :param pathToVideo: путь к видео, по умолчанию путь из settings.ini
    :return: None
    """
    # TODO: разделить на несколько функций
    frameIdx = 0  # счётчик кадров
    pointIdx = 0  # счётчик точек

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в файле

    numberOfPoints = (Xend - Xnull) * frameCount  # количество точек в облаке
    ply = np.zeros((numberOfPoints, 3))  # массив облака точек
    newPly = np.zeros((frameCount, Xend - Xnull))  # массив карты глубины
    zeroLevel = 275  # нулевой уровень в пикселях
    # zmax = 0 # максимальное отклонение по z в пикселях

    start = time.time()
    # открыть видео и сканировать пока видео не закончится
    while (cap.isOpened()):
        ret, frame = cap.read()
        # если с кадром всё в порядке, то сканировать
        if ret == True:
            img = getMask(frame)
            # для первого кадра получить индекс нулевого уровня
            if frameIdx == 70:
                zeroLevel = findZeroLevel(img)
                # print(zeroLevel)
            # обработка изображения по столбцам затем строкам
            for imgX in range(Xnull, Xend):
                for imgY in range(zeroLevel + 1, img.shape[0]):
                    # если пиксель белый и высота больше погрешности,
                    # то посчитать координаты точки, записать в массив
                    # TODO: разделить генерацию облака и карты высоты
                    # TODO: написать фильтрацию облака точек с помощью open3d или подобного
                    if img.item(imgY, imgX) and (imgY - zeroLevel) * Kz > accuracy:
                        # zmax = max(zmax, imgY - zeroLevel)
                        ply[pointIdx, X] = frameIdx * Kx + X_0
                        ply[pointIdx, Y] = (imgX - Xnull) * Ky + Y_0
                        ply[pointIdx, Z] = (imgY - zeroLevel) * Kz + Z_0
                        # заполнение карты глубины
                        newPly[frameIdx, imgX] = 10 * int(ply[pointIdx, Z]) if ply[pointIdx, Z] < 255 else 255
                        break
                    elif img.item(imgY, imgX) and (imgY - zeroLevel) * Kz < accuracy:
                        ply[pointIdx, X] = frameIdx * Kx + X_0
                        ply[pointIdx, Y] = (imgX - Xnull) * Ky + Y_0
                        ply[pointIdx, Z] = ply[pointIdx - 1 if pointIdx > 0 else 0, Z] + Z_0
                        # заполнение карты глубины
                        newPly[frameIdx, imgX] = 10 * int(ply[pointIdx, Z]) if ply[pointIdx, Z] < 255 else 255
                else:
                    # если белых пикселей в стобце нет или высота меньше погрешности, записать предыдущий уровень
                    # TODO: обработку точек по ближайшим соседям или другому алгоритму
                    ply[pointIdx, X] = frameIdx * Kx + X_0
                    ply[pointIdx, Y] = (imgX - Xnull) * Ky + Y_0
                    ply[pointIdx, Z] = Z_0
                    # заполнение карты глубины
                    newPly[frameIdx, imgX] = Z_0
                pointIdx += 1
            frameIdx += 1
            print(f'{frameIdx:{3}}/{frameCount:{3}} processed for {time.time() - start:3.2f} sec')
        else:
            timePassed = time.time() - start
            print(f'Done. Time passed {timePassed:3.2f} sec\n')
            break
    cap.release()
    cv2.imwrite('scanned.png', newPly)
    # print(zmax)
    # сгенерировать файл облака точек
    generatePly(ply)
    cookies = findCookies('scanned.png')[0]
    print(cookies[0][0])
    cv2.destroyAllWindows()
