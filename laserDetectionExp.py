import cv2
import numpy as np
from scanner import startMask



if __name__ == '__main__':
    def nothing(*arg):
        pass

X, Y =0, 1

def normalize(img):
    array = img.copy().astype(np.float64)
    array = (array-array.min())/(array.max() - array.min())
    return array

def findLaserCenter(p1 = (0,0), p2 = (0,0), p3 = (0,0)):
    a = ((p2[Y]-p1[Y])*(p1[X]-p3[X])+(p3[Y]-p1[Y])*(p2[X]-p1[X]))/((p1[X]-p3[X])*(p2[X]**2-p1[X]**2)+(p2[X]-p1[X])*(p3[X]**2-p1[X]**2))
    b = ((p2[Y]-p1[Y])-a*(p2[X]**2-p1[X]**2))
    c = p1[Y] - a*p1[X]**2-b*p1[X]
    xc = -b/(2*a)
    yc = a*xc**2 + b*xc+c
    return (xc, yc)

def secondOrderInvertGaussDerivative(img, ksize, sigma, delta=0.0):
    kernelX = cv2.getGaussianKernel(ksize, sigma)
    kernelY = kernelX.T
    filteredImg = -cv2.sepFilter2D(img, cv2.CV_64F, kernelX, kernelY, delta=delta)
    derivative = cv2.Laplacian(filteredImg, cv2.CV_64F)
    return derivative, filteredImg

cv2.namedWindow("settings")  # создаем окно настроек

# path = input('enter path\n')
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 125, 255, nothing)
cv2.createTrackbar('s1', 'settings', 96, 255, nothing)
cv2.createTrackbar('v1', 'settings', 97, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 198, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('e1', 'settings', 0, 255, nothing)
# cv2.createTrackbar('e2', 'settings', 0, 255, nothing)
# cv2.createTrackbar('ek', 'settings', 0, 2, nothing)
cv2.createTrackbar('krl', 'settings', 2, 100, nothing)
cv2.createTrackbar('opK', 'settings', 0, 50, nothing)
cv2.createTrackbar('iter', 'settings', 0, 10, nothing)
cv2.createTrackbar('sigma', 'settings', 0, 3000, nothing)
cv2.createTrackbar('sigma2', 'settings', 0, 3000, nothing)
cv2.createTrackbar('gauk', 'settings', 0, 100, nothing)
cv2.createTrackbar('delta', 'settings', 0, 1000, nothing)
# crange = [0, 0, 0, 0, 0, 0]
# scan = cv2.imread('scanned.png', 0)

def compare(img, mask, hl, hu, threshold=0.5):
    """
    Побитовое сравнение по маске по количеству белых пикселей
    :param img: изображение для сравнения
    :param mask: применяемая маска
    :param threshold: порог схожести
    :return: True/False в зависимости от схожести
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv, hl, hu)
    compare = cv2.bitwise_and(hsv, hsv, mask=mask)
    cv2.imshow('compare', compare)
    cv2.imshow('hsv', hsv)
    # cv2.waitKey(30)
    similarity = np.sum(compare == 255) / np.sum(mask == 255)
    if similarity >= threshold:
        print(f'{similarity:4.2f}')
        return True
    else:
        return False
ch = 0
ret = False
checker = np.array([[2,0,0,0,2],
                    [2,2,0,2,2],
                    [2,2,0,2,2],
                    [2,2,0,2,2],
                    [2,0,0,0,2]])

while True:
    cap = cv2.VideoCapture('/home/bedlamzd/Reps/MT.Pasticciere/Files/х98у89.mp4')
    print('\n\ncap opened')
    while (cap.isOpened()):
        # print('trying to show')
        if ch != 102:
            ret, frame = cap.read()
        if ret == True:
            frame = frame[240: 400, :]
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _,_,gray = cv2.split(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # считываем значения бегунков
            # h1 = cv2.getTrackbarPos('h1', 'settings')
            # s1 = cv2.getTrackbarPos('s1', 'settings')
            # v1 = cv2.getTrackbarPos('v1', 'settings')
            # e1 = cv2.getTrackbarPos('e1', 'settings')
            # h2 = cv2.getTrackbarPos('h2', 'settings')
            # s2 = cv2.getTrackbarPos('s2', 'settings')
            # v2 = cv2.getTrackbarPos('v2', 'settings')
            # e2 = cv2.getTrackbarPos('e2', 'settings')
            # ek = 2 * cv2.getTrackbarPos('ek', 'settings') + 3
            # k = 2 * cv2.getTrackbarPos('krl', 'settings') + 1
            # opK = 2 * cv2.getTrackbarPos('opK', 'settings') + 1
            # it = cv2.getTrackbarPos('iter', 'settings') + 1
            gauk = 2 * cv2.getTrackbarPos('gauk', 'settings') + 1
            sigma = cv2.getTrackbarPos('sigma', 'settings')/100
            # sigma2 = cv2.getTrackbarPos('sigma2', 'settings')/100
            delta = cv2.getTrackbarPos('delta', 'settings')/1000

            # формируем начальный и конечный цвет фильтра
            # h_min = np.array((h1, s1, v1), np.uint8)
            # h_max = np.array((h2, s2, v2), np.uint8)
            # kernel = np.ones((opK, opK), np.uint8)

            # накладываем фильтр на кадр в модели HSV
            # denoise = cv2.fastNlMeansDenoisingColored(img,None, e1,e2,k,opK)
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.inRange(hsv, h_min, h_max)
            # img = cv2.bitwise_and(frame, frame, mask=thresh)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # gauss = cv2.GaussianBlur(thresh, (k, k), sigma2)
            # ret2, gaussThresh = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # gaussOp = cv2.morphologyEx(gaussThresh, cv2.MORPH_OPEN, kernel, iterations=it)
            # gaussThin = lineThinner(gaussOp)

            # median = cv2.medianBlur(thresh,k,0)
            # ret3,medianThresh = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # medianOp = cv2.morphologyEx(medianThresh, cv2.MORPH_OPEN, kernel, iterations=it)
            # medianThin = lineThinner(medianOp)

            # edges = cv2.Canny(gaussThresh, e1, e2, True)
            # edges2 = cv2.Canny(gaussOp, e1, e2, True)

            # cmp = compare(frame, startMask, h_min, h_max,0.7)
            # if cmp:
            #     print(cmp)
            #     print(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # gray = cv2.bitwise_and(gray, gray, mask=gaussOp)
            derivative = secondOrderInvertGaussDerivative(gray, gauk, sigma, delta)[0]
            result = np.zeros(frame.shape)
            b, g,r = cv2.split(result)
            # derivative = normalize(derivative)
            # derivative[derivative < delta] = 0
            for c, column in enumerate(derivative.T):
                if abs(derivative.max() - column.max()) < 2*sigma:
                    row = column.argmax()
                    g[row, c] = 255
            avgNonzeroColumn = int(np.sum(g.nonzero()[0])/len(g.nonzero()[0])) if len(g.nonzero()[0]) != 0 else 0
            b[avgNonzeroColumn, :] = 255
            result = cv2.merge((b,g,r))
            result[avgNonzeroColumn + 10*int(sigma):,:] =0
            # result[:265, :] = 0
            # b,g,r = cv2.split(result)
            # nim = cv2.filter2D(g/255,-1, checker)
            # g[nim // 2 < 2] = 0
            # result = cv2.merge((b,g*255,r),result)
            cv2.imshow('derivative', derivative)
            cv2.imshow('result', result)
            cv2.imshow('raw', frame)
            # cv2.imshow('mask', thresh)
            # cv2.imshow('denoise', denoise)
            cv2.imshow('gray', gray)
            # cv2.imshow('median', medianThresh)
            # cv2.imshow('gaussian', gaussThresh)
            # cv2.imshow('gauMorph', gaussOp)
            # cv2.imshow('threshEdge', edges)
            # cv2.imshow('morphEdge', edges2)
            # # cv2.imshow('medMorph', medianOp)
            # cv2.imshow('resGau', gaussThin)
            # cv2.imshow('resMed', medianThin)
            # cv2.imshow('masked', img)

        ch = cv2.waitKey(30)
        if ch == 27 or ret == False:
            break
    ch = cv2.waitKey(30)
    if ch == 27:
        break

cv2.destroyAllWindows()
