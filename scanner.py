"""
scanner.py
Author: bedlamzd of MT.lab

Обработка видео в облако точек и нахождение расположения объектов в рабочей области
"""

import numpy as np
import cv2
from global_variables import *
# from open3d import *  # only for visuals
import time
import imutils
import os

# координаты фактического начала стола относительно глобальных координат принтера в мм
X_0 = 0
Y_0 = 0
Z_0 = 0

# максимальная высота на которую может подняться сопло, мм
Z_MAX = 30

# масштабные коэффициенты для построения облака точек, мм/пиксель
Kz = 6 / 74
Kx = 60 / 23
Ky = 70 / 334
# print(Kx, Ky, Kz)

# ширина изображения для обработки, пиксели
Xnull = 0
Xend = 640


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
    print(f'Done for {timePassed:.2f} sec\n')


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
    mask = cv2.inRange(hsv, hsvLowerBound, hsvUpperBound)
    blur = cv2.medianBlur(mask, 3, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = lineThinner(th3, zero_level)
    return mask


def findCookies(imgOrPath):
    """
    Функция нахождения расположения и габаритов объектов на столе из полученной карты глубины
    :param img:
    :return:
    """
    if isinstance(imgOrPath, str):
        img = cv2.imread(imgOrPath, 0)
    else:
        img = imgOrPath
    # cv2.imshow('picture', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    blur = cv2.medianBlur(img, 3, 0)
    ret2, median = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('picture', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel, iterations=3)
    onlyTable = opening
    # выделяем область, которая точно является задним фоном
    sureBg = cv2.dilate(opening, kernel, iterations=4)
    # находим область, которая точно является печеньками
    distTransform = cv2.distanceTransform(onlyTable, cv2.DIST_L2, 3)
    ret, sureFg = cv2.threshold(distTransform, 0.1 * distTransform.max(), 255, 0)
    # Находим область в которой находятся края печенек.
    sureFg = np.uint8(sureFg)
    unknown = cv2.subtract(sureBg, sureFg)
    cv2.imshow('picture', distTransform)
    cv2.imshow('picture1', sureBg)
    cv2.imshow('picture2', sureFg)
    cv2.imshow('picture3', unknown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Marker labelling
    ret, markers = cv2.connectedComponents(sureFg)
    # cv2.imshow('picture', markers)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(median, markers)
    cv2.imshow('picture', markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blankSpace = np.zeros(median.shape, dtype="uint8")
    blankSpace[markers == 1] = 255
    blankSpace = cv2.bitwise_not(blankSpace)
    cv2.imshow('picture', blankSpace)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours = cv2.findContours(blankSpace.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.imshow('picture', contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    table = cv2.bitwise_and(img, img, mask=blankSpace)
    table = cv2.bitwise_and(img, img, mask=sureBg)
    cv2.imshow('picture', table)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contours, table


def scan(pathToVideo=VID_PATH):
    """
    Функция обработки видео (сканирования)
    :param pathToVideo: путь к видео, по умолчанию путь из settings.ini
    :return: None
    """
    frameIdx = 0  # счётчик кадров
    pointIdx = 0  # счётчик точек

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в файле

    numberOfPoints = (Xend - Xnull) * frameCount  # количество точек в облаке
    ply = np.zeros((numberOfPoints, 3))  # массив облака точек
    newPly = np.zeros((frameCount, Xend - Xnull))  # массив карты глубины
    zeroLevel = 10  # нулевой уровень в пикселях
    # zmax = 0 # максимальное отклонение по z в пикселях

    start = time.time()
    # открыть видео и сканировать пока видео не закончится
    while (cap.isOpened()):
        ret, frame = cap.read()
        # если с кадром всё в порядке, то сканировать
        if ret == True:
            img = getMask(frame)
            # для первого кадра получить индекс нулевого уровня
            if frameIdx == 0:
                zeroLevel = findZeroLevel(img)
                # print(zeroLevel)
            # обработка изображения по столбцам затем строкам
            for imgX in range(Xnull, Xend):
                for imgY in range(zeroLevel + 1, img.shape[0]):
                    # если пиксель белый и высота больше погрешности,
                    # то посчитать координаты точки, записать в массив
                    if img.item(imgY, imgX) and (imgY - zeroLevel) * Kz > accuracy:
                        # zmax = max(zmax, imgY - zeroLevel)
                        ply[pointIdx, X] = frameIdx * Kx + X_0
                        ply[pointIdx, Y] = (imgX - Xnull) * Ky + Y_0
                        ply[pointIdx, Z] = (imgY - zeroLevel) * Kz + Z_0
                        # заполнение карты глубины
                        newPly[frameIdx, imgX] = 10 * int(ply[pointIdx, Z]) if ply[pointIdx, Z] < 255 else 255
                        break
                else:
                    # если белых пикселей в стобце нет или высота меньше погрешности, записать уровень стола
                    ply[pointIdx, X] = frameIdx * Kx + X_0
                    ply[pointIdx, Y] = (imgX - Xnull) * Ky + Y_0
                    ply[pointIdx, Z] = Z_0
                pointIdx += 1
            frameIdx += 1
            print(f'{frameIdx:{3}}/{frameCount:{3}} processed for {time.time() - start:{3}} sec')
        else:
            timePassed = time.time() - start
            print(f'Done. Time passed {timePassed:{3}} sec\n')
            break
    cap.release()
    cv2.imwrite('scanned.png', newPly)
    # print(zmax)
    # сгенерировать файл облака точек
    generatePly(ply)
    cv2.destroyAllWindows()
