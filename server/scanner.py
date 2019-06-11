import numpy as np
import cv2
from global_variables import *
# from open3d import *  # only for visuals
import time
import os

X_0 = 0
Y_0 = 0
Z_0 = 0
error = 0.5
Z_MAX = 30

IMG_PATH = './img/'
VID_PATH = 'project.avi'

Kz = 0.45
Kx = 0.303
Ky = 0.725

Xnull = 0
Xend = 640



def generate_PLY(arr):
    print('Generating point cloud...')
    start = time.time()
    ply = []
    for row in arr:
        s = ''
        coord = row.tolist()
        if coord[Z] < Z_0 + error or coord[Z] > Z_MAX:
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


def get_img(path):
    im = cv2.imread(path)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    low_bound = np.array([0, 0, 240])
    up_bound = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, low_bound, up_bound)
    # for x in range(mask.shape[1]):
    #     for y in range(mask.shape[0]):
    #         if (mask.item(y, x) == 255) and (mask.item(y + 1, x) == 255):
    #             mask.itemset((y, x), 0)
    return mask


def get_mask(img, zero_level = 0):
    img = img[zero_level:,:]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_bound = np.array([0, 0, 240])
    up_bound = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, low_bound, up_bound)
    return mask


def scan(pathToVideo):
    Y1 = 0  # счётчик кадров
    p = 0  # счётчик точек

    cap = cv2.VideoCapture(pathToVideo)  # чтение видео
    propId_frameCount = 7
    frameCount = int(cap.get(propId_frameCount))  # всего кадров в файле

    num_pnts = (Xend - Xnull) * frameCount  # количество точек в облаке
    ply = np.zeros((num_pnts, 3))  # массив облака точек
    zero_lvl = 0  # нулевой уровень в пикселях

    start = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = get_mask(frame, zero_lvl)
            if Y1 == 0:
                zero_lvl = find_z_zero_lvl(img)
            for x in range(Xnull, Xend):
                for y in range(zero_lvl + 1, img.shape[0]):
                    if img.item(y, x):
                        # new_row = np.array([(x - Xnull) * Kx + X_0, Y1 * Ky + Y_0, (y - zero_lvl) * Kz + Z_0])
                        # ply = np.append(ply, [new_row], axis=0)
                        ply[p, X] = (x - Xnull) * Kx + X_0
                        ply[p, Y] = Y1 * Ky + Y_0
                        ply[p, Z] = (y - zero_lvl) * Kz + Z_0
                        break
                else:
                    # new_row = np.array([(x - Xnull) * Kx + X_0, Y1 * Ky + Y_0, Z_0])
                    # ply = np.append(ply, [new_row], axis=0)
                    ply[p, X] = (x - Xnull) * Kx + X_0
                    ply[p, Y] = Y1 * Ky + Y_0
                    ply[p, Z] = Z_0
                p += 1
            Y1 += 1
            print('%03d/%03d processed for %03d sec' % (Y1, frameCount, time.time() - start))
        else:
            time_passed = time.time() - start
            print('Done. Time passed %03d sec\n' % time_passed)
            break
    cap.release()
    generate_PLY(ply)
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

"""
старый кусок кода для работы с картинками

files = os.listdir(IMG_PATH)
files.sort()

num_pnts = (Xend - Xnull) * len(files)

ply = np.zeros((num_pnts, 3))

img = get_img(IMG_PATH + files[0])
zero_lvl = find_z_zero_lvl(img)

for file in files:
    img = get_img(IMG_PATH + file)
    for x in range(Xnull, Xend):
        for y in range(zero_lvl + 1, img.shape[0]):
            if img.item(y, x):
                # new_row = np.array([(x - Xnull) * Kx + X_0, Y1 * Ky + Y_0, (y - zero_lvl) * Kz + Z_0])
                # ply = np.append(ply, [new_row], axis=0)
                ply[p, X] = (x - Xnull) * Kx + X_0
                ply[p, Y] = Y1 * Ky + Y_0
                ply[p, Z] = (y - zero_lvl) * Kz + Z_0
                break
        else:
            # new_row = np.array([(x - Xnull) * Kx + X_0, Y1 * Ky + Y_0, Z_0])
            # ply = np.append(ply, [new_row], axis=0)
            ply[p, X] = (x - Xnull) * Kx + X_0
            ply[p, Y] = Y1 * Ky + Y_0
            ply[p, Z] = Z_0
        p += 1
    Y1 += 1
    print('%03d/%03d processed for %03d sec' % (Y1, len(files), time.time() - start))
"""
