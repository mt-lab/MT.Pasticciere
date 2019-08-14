import cv2
import numpy as np
from scanner import LoG, findLaserCenter, calculateCoordinates


def nothing(*arg):
    pass


cv2.namedWindow("result")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек
cv2.namedWindow("settings2")  # создаем окно настроек

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 123, 255, nothing)
cv2.createTrackbar('s1', 'settings', 40, 255, nothing)
cv2.createTrackbar('v1', 'settings', 42, 255, nothing)
cv2.createTrackbar('h2', 'settings', 213, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
cv2.createTrackbar('k', 'settings2', 16, 100, nothing)
cv2.createTrackbar('sigma', 'settings2', 0, 3000, nothing)
cv2.createTrackbar('k2', 'settings2', 16, 100, nothing)
cv2.createTrackbar('2', 'settings2', 0, 3000, nothing)

while True:
    cap = cv2.VideoCapture(r'C:\Users\yktma\Desktop\Studying\repos\MT.Pasticciere\files\х98у89.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')
        k = cv2.getTrackbarPos('k', 'settings2')
        sigma = cv2.getTrackbarPos('sigma', 'settings2')
        k2 = cv2.getTrackbarPos('k2', 'settings2')
        sigma2 = cv2.getTrackbarPos('sigma2', 'settings2')

        # параметры гауссова фильтра
        k = 2 * int(k) + 1
        sigma = sigma / 100
        k2 = 2 * int(k2) + 1
        sigma2 = sigma2 / 100

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # ищем границы и делаем маску
        derivative = LoG(gray, k, sigma)
        hsv = cv2.GaussianBlur(hsv, (k2, k2), sigma2)
        mask = cv2.inRange(hsv, h_min, h_max)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        result = cv2.bitwise_and(derivative, derivative, mask=mask)
        apprxLaserCenter = np.argmax(result, axis=0)
        apprxLaserCenter[apprxLaserCenter < frame.shape[0] / 2] = 0
        fineLaserCenter = np.zeros(apprxLaserCenter.shape)
        for column, row in enumerate(apprxLaserCenter):
            if row == 0:
                continue
            prevRow = row - 1
            nextRow = row + 1 if row < frame.shape[0] - 1 else frame.shape[0] - 1
            p1 = (1.0 * prevRow, derivative[prevRow, column])
            p2 = (1.0 * row, derivative[row, column])
            p3 = (1.0 * nextRow, derivative[nextRow, column])
            frame[row, column] = (0, 255, 0)
            fineLaserCenter[column], _ = findLaserCenter(p1, p2, p3)
        zeroLvl = fineLaserCenter[np.nonzero(fineLaserCenter)]
        if zeroLvl.size != 0:
            zeroLvl = zeroLvl.min() #(zeroLvl[0]+zeroLvl[-1])/2
        else:
            zeroLvl = frame.shape[0] / 2 - 1
        frame[int(zeroLvl)] = (255, 0, 0)
        cv2.imshow('frame', frame)
        cv2.imshow('result', result)
        ch = cv2.waitKey(15)
        if ch == 27:
            break
    ch = cv2.waitKey(30)
    if ch == 27:
        break

cv2.destroyAllWindows()
