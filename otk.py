"""
otk.py
Author: Anton Mayorov of MT.lab

Файл otk.py отвечает за автоматическую систему технического контроля.

содержит 2 основные функции:
    getMask - возвращает маску рисунка в формате png снятую с печенья "оригинала".
Значение фильтра, для выделения маски задается вручную, после завершения программы
оно записывается в файл settings.ini
    mancompare - сравнивает маску, полученную функцией getMask со всеми печеньями
находящимися на столе.
"""

import numpy as np
import cv2
import configparser
import imutils
from matplotlib import pyplot as plt

path = "settings.ini"


def get_config(path):
    """
    Выбираем файл настроек
    """

    config = configparser.ConfigParser()
    config.read(path)
    return config


def update_setting(path, section, setting, value):
    """
    Обновляем параметр в настройках
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)


def nothing(x):
    pass

def segmentation(image):

    """
    Выделяет с изображения область со столом. Находит контуры всех печенек на столе.
    Подсчитывает автоматически количество печенек на столе.
    Возвращает массив с контурами и изображение печенек на черном фоне.

    Параметры:
    image (array) – трехканальное изображение BGR.
    """

    table = image[2:716, 275:1100]
    #перевожу область со столом в gtayscale
    grayTable = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    #выделяю поверхность стола по цвету
    lowerColor = np.array([0,73,141])
    upperColor = np.array([197,210,255])
    hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
    onlyTable = cv2.inRange(hsv,lowerColor,upperColor)
    onlyTable = cv2.bitwise_not(onlyTable)
    # убираем шумы, чтобы добиться полного выделения поверхности стола
    blur = cv2.medianBlur(onlyTable, 7)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(blur,cv2.MORPH_OPEN,kernel, iterations = 2)
    onlyTable = opening
    # выделяем область, которая точно является задним фоном
    sureBg = cv2.dilate(opening,kernel,iterations=3)
    # находим область, которая точно является печеньками
    distTransform = cv2.distanceTransform(onlyTable,cv2.DIST_L2,3)
    ret, sureFg = cv2.threshold(distTransform,0.1*distTransform.max(),255,0)
    # Находим область в которой находятся края печенек.
    sureFg = np.uint8(sureFg)
    unknown = cv2.subtract(sureBg,sureFg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sureFg)
    # Add one to all labels so that sure bqackground is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(table,markers)
    numberOfCokies = len(np.unique(markers)) - 2

    blankSpace = np.zeros(grayTable.shape, dtype="uint8")
    blankSpace[markers == 1] = 255
    blankSpace = cv2.bitwise_not(blankSpace)

    height, width = blankSpace.shape
    blankSpaceCropped = blankSpace[2:height-2, 2:width-2]

    contours = cv2.findContours(blankSpaceCropped.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:numberOfCokies]
    table = cv2.bitwise_and(table,table,mask = blankSpace)
    table = cv2.bitwise_and(table,table,mask = sureBg)
    return contours, table


def getPattern(image,threshlevel):
    """
    Выделяет рисунок нанесенный белой глазурью.

    Параметры:
    image (array) – одноканальное изображение в grayscale.
    threshlevel (int) - число от 0 до 255. Значение порога для threshold
    """
    ret, thresh = cv2.threshold(image, int(threshlevel), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    median = cv2.medianBlur(opening, 5)
    return median

def getMainContour(mask):
    """
    Находит самый большой замкнутый контур.

    Параметры:
    mask(array) – одноканальное изображение в grayscale.
    """
    cnt= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = imutils.grab_contours(cnt)
    contoursNumber = len(cnt)
    cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:1]
    mainCnt = cnt[0]
    return mainCnt, contoursNumber

def cropAndRotation(cnt,table):
    """
    Вырезает печенье со стола и поворачивает его.

    Параметры:
    cnt(vector of 2D points) – контур отдельновзятой печеньки.
    table(array) – трехканальное изображение печенек на черном фоне.
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mult = 1.1
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    rotated = False
    angle = rect[2]
    if W < H:
        angle+=90
        rotated = True
    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.getRectSubPix(table, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = W if not rotated else H
    croppedH = H if not rotated else W
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))
    return croppedRotated,box

def cropAndRotation2(cnt,table,widthOriginal,heightOriginal):
    """
    Вырезает печенье со стола и поворачивает его.

    Параметры:
    cnt(vector of 2D points) – контур отдельновзятой печеньки.
    table(array) – трехканальное изображение печенек на черном фоне.
    widthOriginal(int) - ширина эталонного изображения
    heightOriginal(int) - высота эталонного изображения
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mult = 1.1
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    x1Corrected = x1-int(((x2-x1)*(mult-1))/2)
    x2Corrected = x2+int(((x2-x1)*(mult-1))/2)
    y1Corrected = y1-int(((y2-y1)*(mult-1))/2)
    y2Corrected = y2+int(((y2-y1)*(mult-1))/2)

    x1Corrected = 0 if x1Corrected<0 else x1Corrected
    x2Corrected = 824 if x2Corrected>824 else x2Corrected
    y1Corrected = 0 if y1Corrected<0 else y1Corrected
    y2Corrected = 714 if y2Corrected>714 else y2Corrected

    for i in range (y1Corrected,y2Corrected):
        for j in range(x1Corrected,x2Corrected):
            dist_from_contour = cv2.pointPolygonTest(cnt, (j, i), True)
            if dist_from_contour < 0:
                table[i,j] = 0;

    rotated = False
    angle = rect[2]
    if W < H:
        angle+=90
        rotated = True
    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.getRectSubPix(table, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = widthOriginal
    croppedH = heightOriginal
    croppedRotated = cv2.getRectSubPix(cropped, (int(widthOriginal), int(heightOriginal)), (size[0]/2, size[1]/2))
    return croppedRotated,box

def getMask():
    """
    Со снимка эталонного печенья создает маску. Уровень фильтра threshold выбирается вручную,
    его значение сохраняется в settings.ini
    """
    original = cv2.imread("cookie1/origin.jpg",1)
    contours, table = segmentation(original)
    cnt = contours[0];
    table_copy = table.copy();
    croppedRotated,box = cropAndRotation(cnt,table_copy)
    cv2.namedWindow('threshholding')
    cv2.createTrackbar('Tlevel', 'threshholding', 0, 255, nothing)
    gray = cv2.cvtColor(croppedRotated, cv2.COLOR_BGR2GRAY)
    while True:
        threshlevel = cv2.getTrackbarPos('Tlevel', 'threshholding')
        median = getPattern(gray,threshlevel)
        cv2.imshow("threshholding", median)
        k = cv2.waitKey(1) & 0xFF
        if  (k == ord('q')) | (k == 27):
            break
    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", median)
    update_setting(path, "OTK", "threshlevel", str(threshlevel))

def mancompare(threshlevel):
    """
    Сравнивает маску, полученную функцией getMask со рисунками на каждом печенье.

    Параметры:
    threshlevel (int) - число от 0 до 255. Значение порога для threshold, применяемое для эталонного изображения
    """
    #Читаем нужные изображения
    original = cv2.imread("cookie1/12.jpg",1)
    mask = cv2.imread("mask.png", 0)
    widthOriginal = mask.shape[1]
    heightOriginal  = mask.shape[0]
    #Основные параметры сравнения:
    numPixelsTolerance = 0.09
    contourLengthTolerance = 0.03
    shapeCompareTolerance = 0.05
    #создаем окно для отображения каждой печеньки
    fig=plt.figure(figsize=(10,5))
    columns = 3
    rows = 2
    subplot_counter = 1
    #----------------------------------------------
    defectsCounter = 0 #сюда записывается количество печенья-брака на столе
    cnt1, contoursNumberOriginal = getMainContour(mask)
    mainСontourLegthOriginal = cv2.arcLength(cnt1,True)
    n_white_pix_original = np.sum(mask == 255)
    ax = fig.add_subplot(rows, columns, subplot_counter)
    ax.set_title("origin")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    subplot_counter+=1
    print("ideal number of white pixels",n_white_pix_original)
    print("ideal length",mainСontourLegthOriginal)
    print("ideal number of contours", contoursNumberOriginal)
    contours, table = segmentation(original)
    original_table = table.copy()

    for cnt in contours:
        table_copy = table.copy();
        croppedRotated,box = cropAndRotation2(cnt,table_copy,widthOriginal,heightOriginal)
        #получение маски с каждой печеньки
        gray = cv2.cvtColor(croppedRotated, cv2.COLOR_BGR2GRAY)
        median = getPattern(gray,threshlevel)
        #нахождение внешнего контура, его длины и оценка формы
        cnt2, contoursNumber = getMainContour(median)
        main_contour_legth = cv2.arcLength(cnt2,True)
        mask_RGB=cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(mask_RGB,[cnt2],0,(0,0,255),2)
        n_white_pix = np.sum(median == 255)
        ax = fig.add_subplot(rows, columns, subplot_counter)
        ax.set_title(subplot_counter-1)
        plt.axis("off")
        #plt.imshow(cv2.cvtColor(median, cv2.COLOR_GRAY2RGB))
        plt.imshow(mask_RGB)
        print("-------------------------------------")
        print("Cookie number:",subplot_counter-1)
        print("main_contour_legth:",main_contour_legth)
        print("number of white pixels:",n_white_pix)
        print("contoursNumber",contoursNumber)
        print("-------------------------------------")
        subplot_counter+=1
        if ((abs(n_white_pix_original - n_white_pix)>numPixelsTolerance*n_white_pix_original) | (abs(mainСontourLegthOriginal - main_contour_legth)>contourLengthTolerance*mainСontourLegthOriginal)|(contoursNumberOriginal != contoursNumber)):
            defectsCounter +=1
            cv2.drawContours(original_table,[box],0,(0,0,255),2)
        else:
            cv2.drawContours(original_table,[box],0,(0,255,0),2)

    if defectsCounter == 0:
        answer = "yes"
    else:
        answer = "no"
        cv2.imshow("Table with result",original_table)
    print(answer)
    print("Number of cookies with defect:",defectsCounter)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return answer
