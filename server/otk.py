import numpy as np
import cv2
import configparser

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
    cv2.namedWindow('threshholding')
    cv2.createTrackbar('Tlevel','threshholding',0,255,nothing)
    while True:
        original = cv2.imread("cookie.png", 1)
        original = cv2.resize(original,None,fx=0.8, fy=0.8, interpolation =  cv2.INTER_CUBIC)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 15, 17, 17)
        blur = cv2.medianBlur(blur,3)
        threshlevel = cv2.getTrackbarPos('Tlevel', 'threshholding')
        ret,thresh = cv2.threshold(gray,threshlevel,255,cv2.THRESH_BINARY)
        kernel1 = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
        median = cv2.medianBlur(opening,9)
        cv2.imshow("blur", blur)
        #cv2.imshow("threshholding", opening)
        cv2.imshow("gray", gray)
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

    rows,cols = mask.shape
    print(rows,cols)

    compairing_result = open('compairing_result.txt', 'w')

    photo = cv2.resize(photo,None,fx=0.8, fy=0.8, interpolation =  cv2.INTER_CUBIC)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 15, 17, 17)
    blur = cv2.medianBlur(blur,3)
    ret,thresh = cv2.threshold(gray,int(threshlevel),255,cv2.THRESH_BINARY)
    kernel1 = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    median = cv2.medianBlur(opening,9)
    mask_from_photo = median

    cv2.imshow("mask", mask)
    cv2.imshow("mask_from_photo", mask_from_photo)

    ret = cv2.matchShapes(mask,mask_from_photo,1,0.0)

    for i in range(rows):
        for j in range(cols):
            if mask[i,j] != mask_from_photo[i,j]:
                counter+=1
                photo[i,j] = [0,0,255]
            if (mask[i,j] == mask_from_photo[i,j]) & (mask[i,j] == 255):
                photo[i,j] = [0,255,0]


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
