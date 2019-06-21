import numpy as np
import cv2
from global_variables import *
# from open3d import *  # only for visuals
import time
import imutils
import os

X_0 = 0
Y_0 = 0
Z_0 = 0
error = 0.5
Z_MAX = 30

Kz = 6/74
Kx = 60/23
Ky = 70/334
print(Kx, Ky, Kz)

Ynull = 0
Yend = 640


def generate_PLY(arr):
    print('Generating point cloud...')
    start = time.time()
    ply = []
    for row in arr:
        s = ''
        coord = row.tolist()
        if coord[Z] < Z_0 + accuracy or coord[Z] > Z_MAX:
            continue
        for element in coord:
            s = s + str(round(element, 3)) + ' '
        s += '\n'
        ply.append(s)
    f = open("cloud.ply", "w+")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % len(ply))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for point in ply:
        f.write(point)
    f.close()
    time_passed = time.time() - start
    print('Done for %03d sec\n' % time_passed)


def find_z_zero_lvl(img):
    row_sum = img.sum(axis=1)
    return np.argmax(row_sum)

def lineThinner(img, upperBound):
    newImg = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]-1, upperBound, -1):
            if (img.item(y,x) == 255) and (img.item(y,x) == img.item(y-1,x)):
                newImg.itemset((y,x), 255)
                break
    return newImg

def get_mask(img, zero_level=0):
    img = img[zero_level:, :]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_bound = np.array([0, 0, 129])
    up_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, low_bound, up_bound)
    blur = cv2.medianBlur(mask,3,0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = lineThinner(th3,zero_level)
    cv2.imshow('mask', mask)
    cv2.waitKey(50)
    return mask

def findCookies(img, contoursNumber=1):

    # cv2.imshow('picture', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    blur = cv2.medianBlur(img,3,0)
    ret2,median = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
    cv2.imshow('picture', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(median,cv2.MORPH_OPEN,kernel, iterations = 3)
    onlyTable = opening
    # выделяем область, которая точно является задним фоном
    sureBg = cv2.dilate(opening,kernel,iterations=4)
    # находим область, которая точно является печеньками
    distTransform = cv2.distanceTransform(onlyTable,cv2.DIST_L2,3)
    ret, sureFg = cv2.threshold(distTransform,0.1*distTransform.max(),255,0)
    # Находим область в которой находятся края печенек.
    sureFg = np.uint8(sureFg)
    unknown = cv2.subtract(sureBg,sureFg)
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
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(median,markers)
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
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:contoursNumber]
    cv2.imshow('picture', contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    table = cv2.bitwise_and(img,img,mask = blankSpace)
    table = cv2.bitwise_and(img,img,mask = sureBg)
    cv2.imshow('picture', table)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contours, table

def scan(pathToVideo=VID_PATH):
    frameNumber = 0  # счётчик кадров
    p = 0  # счётчик точек

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # всего кадров в файле

    numberOfPoints = (Yend - Ynull) * frameCount  # количество точек в облаке
    ply = np.zeros((numberOfPoints, 3))  # массив облака точек
    new_ply = np.zeros((frameCount, Yend - Ynull)) # массив карты глубины
    zero_lvl = 10  # нулевой уровень в пикселях
    zmax = 0

    start = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = get_mask(frame)
            if frameNumber == 0:
                zero_lvl = find_z_zero_lvl(img)
                print(zero_lvl)
            # if frameNumber == 0:
            #     cv2.imwrite('sample0.png', frame)
            for pxlY in range(Ynull, Yend):
                for pxlX in range(zero_lvl + 1, img.shape[0]):
                    if img.item(pxlX, pxlY):
                        # new_row = np.array([(pxlY - Xnull) * Kx + X_0, frameNumber * Ky + Y_0, (pxlX - zero_lvl) * Kz + Z_0])
                        # ply = np.append(ply, [new_row], axis=0)
                        zmax = max(zmax, pxlX-zero_lvl)
                        ply[p, X] = frameNumber * Kx + X_0
                        ply[p, Y] = (pxlY - Ynull) * Ky + Y_0
                        ply[p, Z] = (pxlX - zero_lvl) * Kz + Z_0
                        if ply[p,Z] > (Z_0 + accuracy):
                            new_ply[frameNumber, pxlY] = 10*int(ply[p,Z]) if ply[p,Z] < 255 else 255
                        break
                else:
                    # new_row = np.array([(pxlY - Xnull) * Kx + X_0, frameNumber * Ky + Y_0, Z_0])
                    # ply = np.append(ply, [new_row], axis=0)
                    ply[p, X] = frameNumber * Kx + X_0
                    ply[p, Y] = (pxlY - Ynull) * Ky + Y_0
                    ply[p, Z] = Z_0
                p += 1
            frameNumber += 1
            print('%03d/%03d processed for %03d sec' % (frameNumber, frameCount, time.time() - start))
        else:
            time_passed = time.time() - start
            print('Done. Time passed %03d sec\n' % time_passed)
            break
    cap.release()
    # Otsu's thresholding after Gaussian filtering
    cv2.imwrite('scanned.png', new_ply)
    print(zmax)
    # scanned = cv2.imread('scanned.png', 0)
    # blur = cv2.GaussianBlur(scanned,(7,7),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('contour', th3)
    # cv2.waitKey(0)
    generate_PLY(ply)
    cv2.destroyAllWindows()
    # xmax = max([_[X] for _ in ply])
    # xmin = min([_[X] for _ in ply])
    # ymax = max([_[Y] for _ in ply])
    # ymin = min([_[Y] for _ in ply])
    #
    # X_C = (xmax+xmin)/2
    # Y_C = (ymax+ymin)/2
    # width = abs(xmax - xmin)
    # height = abs(ymax-ymin)
    # print(X_C, Y_C)
    # print(width, height)

# pcd = read_point_cloud('cloud.ply')
# draw_geometries([pcd])
# scan(VID_PATH)
