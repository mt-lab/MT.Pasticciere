import numpy as np
import cv2

counter = 0

mask = cv2.imread("mask.png", 0)
photo = cv2.imread("cookie_changed.jpg", 1)

rows,cols = mask.shape
print(rows,cols)

threshlevel_read = open('threshlevel.txt', 'r')
threshlevel = threshlevel_read.read()
compairing_result = open('compairing_result.txt', 'w')

gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,int(threshlevel),255,cv2.THRESH_BINARY)
kernel1 = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
median = cv2.medianBlur(opening,7)
mask_from_photo = median

cv2.imshow("mask", mask)
cv2.imshow("photo", median)

ret = cv2.matchShapes(mask,mask_from_photo,1,0.0)

for i in range(rows):
    for j in range(cols):
        if mask[i,j] != mask_from_photo[i,j]:
            counter+=1

print(counter)

if ret < 0.01:
    answer = "yes"
else:
    answer = "no"

cv2.waitKey(0)

compairing_result.write(answer)
cv2.destroyAllWindows()
threshlevel_read.close()
compairing_result.close()