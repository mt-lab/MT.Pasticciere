import numpy as np
import cv2


def mancompare(threshlevel):
    counter = 0

    mask = cv2.imread("mask.png", 0)
    photo = cv2.imread("cookie_changed.png", 1)
    photo = cv2.imread("2.png", 1)

    rows,cols = mask.shape
    print(rows,cols)

#    threshlevel_read = open('threshlevel.txt', 'r')
#    threshlevel = threshlevel_read.read()
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
#    threshlevel_read.close()
    compairing_result.close()

# mancompare()
