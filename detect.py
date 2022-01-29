import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread(r'C:\\Users\\admin\\Desktop\\BarcodesDetect\\barcodes.jpg', cv2.IMREAD_GRAYSCALE)
im_out = cv2.imread(r'C:\\Users\\admin\\Desktop\\BarcodesDetect\\barcodes.jpg')
# cv2.imshow("image",im)


#黑帽运算，提取物体内部的“黑洞”
kernel = np.ones((1, 3), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow("blackhat",im)

#二值化操作:可以通过阈值的设定来提取出我们感兴趣的部分
# src：表示的是图片源
# thresh：表示的是阈值（起始值）
# maxval：表示的是最大值
# type：表示的是这里划分的时候使用的是什么类型的算法**，常用值为0（cv2.THRESH_BINARY）**
thresh, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)


#先膨胀，之后执行闭操作。闭操作是为了填充物体内的“黑洞”
kernel = np.ones((1, 5), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2)  # dilatazione
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)  # chiusura

#执行开操作，消除物体之外的小区域
kernel = np.ones((21, 35), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.namedWindow("temp",cv2.WINDOW_NORMAL)
cv2.imshow("temp",im)
# 提取

iimage,contours ,hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours!=None:
    for contour in contours:
        if cv2.contourArea(contour) <= 10000:
            continue
        cv2.drawContours(im_out,contours ,-1,(0,255,0),5)

cv2.namedWindow("cur",cv2.WINDOW_NORMAL)
cv2.imshow("cur",im_out)
cv2.waitKey(0)
