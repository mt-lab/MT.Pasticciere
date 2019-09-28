import cv2
import numpy as np
from math import pi, atan
from scanner import LoG, findLaserCenter, calculateCoordinates


def nothing(*arg):
    pass


def predict_zero_level(array: np.ndarray, mid_row=239, row_start=0, row_end=479):
    """

    :param array: list of points describing laser position
    :return:
    """
    zero_level = np.full_like(array, mid_row)
    nonzero_indices = array.nonzero()[0]
    if nonzero_indices.size:
        first_nonzero = nonzero_indices[0]
        last_nonzero = nonzero_indices[-1]
        tangent = (fineLaserCenter[last_nonzero] - fineLaserCenter[first_nonzero]) / (last_nonzero - first_nonzero)
        tangent = 0 if tangent == np.inf else tangent
        for column in range(zero_level.size):
            zero_level[column] = (column - first_nonzero) * tangent + fineLaserCenter[first_nonzero]
    return zero_level, tangent


def predict_laser(img, mid_row=239, row_start=0, row_end=480, column_start=0, column_end=640):
    """

    :param img: grayscale img
    :return:
    """
    apprxLaserCenter = np.argmax(derivative, axis=0)
    apprxLaserCenter[apprxLaserCenter > (row_end - row_start - 1)] = 0
    fineLaserCenter = np.zeros(apprxLaserCenter.shape)
    for column, row in enumerate(apprxLaserCenter):
        if row == 0:
            continue
        prevRow = row - 1
        nextRow = row + 1 if row < derivative.shape[0] - 1 else derivative.shape[0] - 1
        p1 = (1.0 * prevRow, derivative[prevRow, column])
        p2 = (1.0 * row, derivative[row, column])
        p3 = (1.0 * nextRow, derivative[nextRow, column])
        fineLaserCenter[column] = findLaserCenter(p1, p2, p3)[0] + row_start
    fineLaserCenter[fineLaserCenter > row_end - 1] = row_end
    return fineLaserCenter


# cv2.namedWindow("result")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек
# cv2.namedWindow("settings2")  # создаем окно настроек

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('Ynull', 'settings', 0, 480, nothing)
cv2.createTrackbar('Yend', 'settings', 480, 480, nothing)
cv2.createTrackbar('Xnull', 'settings', 0, 640, nothing)
cv2.createTrackbar('Xend', 'settings', 640, 640, nothing)
# cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('k', 'settings2', 16, 100, nothing)
# cv2.createTrackbar('sigma', 'settings2', 0, 3000, nothing)
# cv2.createTrackbar('k2', 'settings2', 16, 100, nothing)
# cv2.createTrackbar('2', 'settings2', 0, 3000, nothing)

# cap = cv2.VideoCapture(2)

ksize = 29
sigma = 4.45
laserAngle = [0, 0]

while True:
    cap = cv2.VideoCapture(r'/home/bedlamzd/MT.Pasticciere/scanner.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        Yend = cv2.getTrackbarPos('Yend', 'settings')
        Ynull = cv2.getTrackbarPos('Ynull', 'settings')
        Xnull = cv2.getTrackbarPos('Xnull', 'settings')
        Xend = cv2.getTrackbarPos('Xend', 'settings')
        if ret != True:
            break
        roi = frame[Ynull:Yend, Xnull:Xend] if not Ynull >= Yend and not Xnull >= Xend else np.zeros(frame.shape,
                                                                                                     dtype='uint8')
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        derivative = LoG(gray, ksize, sigma)
        blur = cv2.GaussianBlur(gray, (33, 33), 0)
        _, mask = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
        derivative = cv2.bitwise_and(derivative, derivative, mask=mask)
        derivative[derivative < 0] = 0
        # smth = derivative.copy()
        # smth[smth<0] =0
        # smth = cv2.normalize(smth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        apprxLaserCenter = np.argmax(derivative, axis=0)
        apprxLaserCenter[apprxLaserCenter > (Yend - Ynull - 1)] = 0
        fineLaserCenter = np.zeros(apprxLaserCenter.shape)
        for column, row in enumerate(apprxLaserCenter):
            if row == 0:
                continue
            prevRow = row - 1
            nextRow = row + 1 if row < derivative.shape[0] - 1 else derivative.shape[0] - 1
            p1 = (1.0 * prevRow, derivative[prevRow, column])
            p2 = (1.0 * row, derivative[row, column])
            p3 = (1.0 * nextRow, derivative[nextRow, column])
            fineLaserCenter[column] = findLaserCenter(p1, p2, p3)[0] + Ynull
        fineLaserCenter[fineLaserCenter > Yend - 1] = Yend
        #########################################################
        # zeroLevel = fineLaserCenter[fineLaserCenter > Ynull]  #
        # if len(zeroLevel) != 0:                               #
        #     zeroLevel = zeroLevel[0]                          #
        #########################################################
        zeroLevel = np.zeros(fineLaserCenter.shape)

        nzIdc = np.nonzero(fineLaserCenter)
        if len(nzIdc[0]) != 0:
            fnzIdx = nzIdc[0][0]
            lnzIdx = nzIdc[0][-1]
            ####################################################################################
            cv2.circle(frame, (fnzIdx, int(fineLaserCenter[fnzIdx])), 5, (127, 0, 127), -1)  #
            cv2.circle(frame, (lnzIdx, int(fineLaserCenter[lnzIdx])), 5, (127, 0, 127), -1)  #
            ####################################################################################
            tangent = (fineLaserCenter[lnzIdx] - fineLaserCenter[fnzIdx]) / (lnzIdx - fnzIdx)
            # if laserAngle[1] >= cap.get(cv2.CAP_PROP_FPS):
            #     lA = laserAngle[0]
            # lA = (laserAngle[0] * laserAngle[1] + tangent) / (laserAngle[1] + 1)
            lA = tangent
            if lA == np.inf:
                lA = 0
            laserAngle[0] = lA
            laserAngle[1] += 1
            print(lA * 200, atan(lA) * 180 / pi)
            for column, _ in enumerate(zeroLevel):
                zeroLevel[column] = (column - fnzIdx) * lA + fineLaserCenter[fnzIdx]
                zeroLevel[zeroLevel < Ynull] = Ynull
                zeroLevel[zeroLevel > Yend - 1] = Yend - 1
                if fineLaserCenter[column] < zeroLevel[column]:
                    fineLaserCenter[column] = zeroLevel[column]
        ######################################################################
        else:  #
            zeroLevel = int(frame.shape[0] / 2 - 1)  #
            fineLaserCenter[fineLaserCenter < zeroLevel] = zeroLevel  #
        ##########################################################################
        for column, row in enumerate(fineLaserCenter):  #
            frame[int(row), column] = (0, 255, 0)  #
            if isinstance(zeroLevel, int) or isinstance(zeroLevel, float):  #
                frame[int(zeroLevel), column] = (255, 0, 0)  #
            else:  #
                frame[int(zeroLevel[column]), column] = (255, 0, 0)  #
        frame[:Ynull] = (100, 100, 100)
        frame[:, :Xnull] = (100, 100, 100)
        frame[Yend:] = (100, 100, 100)
        frame[:, Xend:] = (100, 100, 100)
        frame[int(frame.shape[0] / 2) - 1, :] = (150, 0, 200)
        frame[:, int(frame.shape[1] / 2) - 1] = (150, 0, 200)
        frame[int(fineLaserCenter.max())] = (0, 0, 255)  #
        # cv2.imshow('smth', smth)
        cv2.imshow('frame', frame)  #
        cv2.imshow('mask', mask)  #
        cv2.waitKey(15)  #
        ##########################################################################
        ch = cv2.waitKey(15)
        if ch == 27:
            break
    ch = cv2.waitKey(30)
    if ch == 27:
        break

cv2.destroyAllWindows()
