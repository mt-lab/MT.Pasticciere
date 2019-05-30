import numpy as np
import cv2
import configparser
import imutils

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


def manmask():
    original = cv2.imread("photo_1.jpg",1)
    #вырезаю область со столом
    table = original[13:700, 260:1000]
    #вырезаю область со столом
    gray_table = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    #выделяю персиковый цвет стола по хуевилу
    lower_color = np.array([133,20,0])
    upper_color = np.array([215,155,255])
    hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
    only_table = cv2.inRange(hsv,lower_color,upper_color)
    only_table = cv2.bitwise_not(only_table)
    # убираем шумы
    blur = cv2.medianBlur(only_table, 7)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(blur,cv2.MORPH_OPEN,kernel, iterations = 4)
    only_table = opening
    # выделяем область, которая точно является задним фоном
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # находим область, которая точно является печеньками
    dist_transform = cv2.distanceTransform(only_table,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(table,markers)



    blank_space = np.zeros(gray_table.shape, dtype="uint8")
    #blank_space[markers == -1] = 255
    blank_space[markers == 1] = 255
    blank_space = cv2.bitwise_not(blank_space)


    height, width = blank_space.shape
    blank_space_cropped = blank_space[2:height-2, 2:width-2]

    contours = cv2.findContours(blank_space_cropped.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    table = cv2.bitwise_and(table,table,mask = blank_space)
    table = cv2.bitwise_and(table,table,mask = sure_bg)

    for cnt in contours:

            #cv2.drawContours(table, [cnt], -1, (0, 255, 255), 2)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(table,[box],0,(0,0,255),2)
            mult = 1

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

            if angle < -45:
                angle+=90
                rotated = True

            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
            #cv2.circle(table, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

            cropped = cv2.getRectSubPix(table, size, center)
            cropped = cv2.warpAffine(cropped, M, size)

            croppedW = W if not rotated else H
            croppedH = H if not rotated else W

            croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))


    cv2.namedWindow('threshholding')
    cv2.createTrackbar('Tlevel', 'threshholding', 0, 255, nothing)
    gray = cv2.cvtColor(croppedRotated, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 15, 17, 17)
    blur = cv2.medianBlur(blur, 3)

    while True:
        threshlevel = cv2.getTrackbarPos('Tlevel', 'threshholding')
        ret, thresh = cv2.threshold(gray, threshlevel, 255, cv2.THRESH_BINARY)
        kernel1 = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
        median = cv2.medianBlur(opening, 9)
        cv2.imshow("threshholding", median)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", median)
    update_setting(path, "OTK", "threshlevel", str(threshlevel))

def mancompare(threshlevel):
    counter = 0

    mask = cv2.imread("mask.png", 0)
    photo = cv2.imread("cookie_changed.png", 1)
    photo = cv2.imread("2.png", 1)

    rows, cols = mask.shape
    print(rows, cols)

    compairing_result = open('compairing_result.txt', 'w')

    photo = cv2.resize(photo, None, fx=0.8, fy=0.8,
                       interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 15, 17, 17)
    blur = cv2.medianBlur(blur, 3)
    ret, thresh = cv2.threshold(gray, int(threshlevel), 255, cv2.THRESH_BINARY)
    kernel1 = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    median = cv2.medianBlur(opening, 9)
    mask_from_photo = median

    cv2.imshow("mask", mask)
    cv2.imshow("mask_from_photo", mask_from_photo)

    ret = cv2.matchShapes(mask, mask_from_photo, 1, 0.0)

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] != mask_from_photo[i, j]:
                counter += 1
                photo[i, j] = [0, 0, 255]
            if (mask[i, j] == mask_from_photo[i, j]) & (mask[i, j] == 255):
                photo[i, j] = [0, 255, 0]

    cv2.imshow("photo", photo)
    print(counter)
    print(ret)

    if ret < 0.001:
        answer = "yes"
    else:
        answer = "no"

    cv2.waitKey(0)

    compairing_result.write(answer)
    cv2.destroyAllWindows()
    compairing_result.close()
