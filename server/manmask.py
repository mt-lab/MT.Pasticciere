import numpy as np
import cv2

def nothing(x):
     pass

cv2.namedWindow('threshholding')
cv2.createTrackbar('Tlevel','threshholding',0,255,nothing)
while True:
    original = cv2.imread("cookie.jpg", 1)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    threshlevel = cv2.getTrackbarPos('Tlevel','threshholding')
    ret,thresh = cv2.threshold(gray,threshlevel,255,cv2.THRESH_BINARY)
    kernel1 = np.ones((4,4),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    median = cv2.medianBlur(opening,7)
    #cv2.imshow("original", original)
    #cv2.imshow("threshholding", opening)
    #cv2.imshow("closing", closing)
    cv2.imshow("threshholding", median)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.imwrite("mask.png", median)
threshlevel_save = open('threshlevel.txt', 'w')
threshlevel_save.write(str(threshlevel))
threshlevel_save.close()
