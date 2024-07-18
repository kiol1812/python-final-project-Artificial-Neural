import os
import cv2
import numpy as np
import pandas as pd


def split(imgName):
    splits = np.zeros((4, 288))
    img = cv2.imread("samples/"+imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #gray
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernal =  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erode = cv2.erode(thresh, kernel=kernal, iterations=1)
    kernal =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(erode, kernel=kernal,  iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])
    index=0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if(x>0): x-=1
        splits[index] = gray[0:24, x:x+12].reshape(288)#top:bottom, left:right
        index+=1
    return splits

dir = os.fsencode("samples")
count=0
data = np.zeros((9955*4, 288))
labels = np.zeros((9955*4, 1))
for file in os.listdir(dir):
    filename = os.fsdecode(file)
    splits = split(filename)
    index=0
    for s in splits:
        data[count] = s
        labels[count, 0] = ord(filename[index])
        count+=1
        index+=1
    # if count>400:
    #     break

# np.savetxt("train/train.csv", data, delimiter=",")
result = np.concatenate((labels, data), axis=1)
pd.DataFrame(result).to_csv("train/sample.csv")