import cv2
import os

img = cv2.imread("samples/2A2X.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #gray
# cv2.imwrite("test.jpg", img)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

kernal =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))
erode = cv2.erode(thresh, kernel=kernal, iterations=1)
kernal =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
dilate = cv2.dilate(erode, kernel=kernal,  iterations=1)
cv2.imwrite("dilate.png", dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts)==2 else cnts[1]
cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (36, 255, 12), 2)
    # if(w>15 and h>15):
        # print(x, y, w,  h)

cv2.imwrite("result.png", img)

