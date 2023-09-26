import sys

import cv2
import numpy as np
from matplotlib.pyplot import *
from pylab import *


def ReadImage(imgpath):
    img = cv2.imread(imgpath)
    g = img[:, :, 0]
    b = img[:, :, 1]
    r = img[:, :, 2]
    # cv2.imshow("g",g)
    # cv2.imshow("b", b)
    # cv2.imshow("r", r)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return b, img


def zuixiaojuxing(b_img, image):
    ret, thresh = cv2.threshold(b_img, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # To draw an individual contour, say 4th contour:
    cv2.drawContours(image, contours, 3, (0, 255, 0), 3)
    # But most of the time, below method will be useful:
    cnt = contours[4]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)

    cv2.imshow("new img",image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def Tongji(grayimg):
    grayvalue = np.zeros((256))
    h, w = grayimg.shape
    for i in range(h):
        for j in range(w):
            val = grayimg[i, j]
            grayvalue[val] += 1

    return grayvalue

def upsample(img, maxw, maxh):
    # img = cv2.imread(imgpath)
    (h, w) = img.shape[:2]

    # newimg = rotate_bound(img,10)
    # cv2.imshow("oldimg", img)
    newimg = img.copy()
    while (h < maxh or w < maxw):
        newimg = cv2.pyrUp(newimg, dstsize=(w * 2, h * 2))
        (h, w) = newimg.shape[:2]
    # cv2.imshow("newimg", newimg)
    return newimg
def fenduantongji(tongjival):
    fenduan = np.array_split(tongjival, 5)
    vallst = []
    for i in fenduan:
        sumval = i.sum()
        vallst.append(sumval)
    return vallst

def JYZ_Fenxi(imagePath):
    # 获取照片
    retdict = {}
    try:
        img = cv2.imread(imagePath)
        img = upsample(img, 224, 224)
        # 缩放
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        # 获取灰度照片
        # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray = img[:, :, 1]
        SizeAll = gray.shape[0] * gray.shape[1]
        # cv2.imshow('gray', gray)
        # 获取二值照片
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow('binary', binary)
        # 获取轮廓
        contours, hie = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        totle_area = 0
        totle_num = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 80:
                # print("x = {} , y = {} , w = {} , h = {}, 面积 = {}".format(x, y, w, h, w * h))
                totle_area = totle_area + w * h
                totle_num += 1
        faremianji = totle_area / SizeAll
        # print("发热区域占比：{}".format(totle_area / SizeAll))
        # 画出轮廓
        # draw_img = img.copy()
        # cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)
        # imgs = np.hstack([img, draw_img])
        # cv2.imshow('imgs', imgs)
        # cv2.waitKey()
        retdict["totle_num"] = totle_num
        retdict["totle_area"] = totle_area
        if 0<totle_num <= 3 and faremianji <= 0.10:
            retdict["quexian"] = True
            # return True
        else:
            retdict["quexian"] = False
            # return False
    except BaseException as e :
        retdict = {}
        retdict["totle_num"] = -1
        retdict["totle_area"] = -1
        retdict["quexian"] = False
    finally:
        return retdict

def XianJia_Fenxi(imagePath):
    retdict = {}
    # 获取照片
    img = cv2.imread(imagePath)
    img = upsample(img, 300, 300)
    # 缩放
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # 获取灰度照片
    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = img[:, :, 1]
    SizeAll = gray.shape[0] * gray.shape[1]
    # cv2.imshow('gray', gray)
    # 获取二值照片
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary)
    # 获取轮廓
    contours, hie = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    totle_area = 0
    totle_num = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > 49:
            # print("x = {} , y = {} , w = {} , h = {}, 面积 = {}".format(x, y, w, h, w * h))
            totle_area = totle_area + w * h
            totle_num += 1
    retdict["totle_num"] = totle_num
    retdict["totle_area"] = totle_area
    faremianji = totle_area / SizeAll
    # print("发热区域占比：{}".format(totle_area / SizeAll))
    # 画出轮廓
    # draw_img = img.copy()
    # cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)
    # imgs = np.hstack([img, draw_img])
    # cv2.imshow('imgs', imgs)
    # cv2.waitKey()
    if 1<= totle_num < 3 and faremianji <= 0.30:
        retdict["quexian"] = True
    else:
        retdict["quexian"] = True
    return retdict


if __name__ == '__main__':
    # flag = JYZ_Fenxi("D:\\python_prj\\study\\studytorch\\yuanshituxiang\\1_8bf2ab64f097a0856f569e99c2dbe8ba.png")
    # flag = XianJia_Fenxi("D:\\python_prj\\study\\studytorch\\shudiandeyang3316\\datasets\\6_192efe04985bee8607be13c90e5bf349.png")
    # print(flag)
    # print(sys.argv)
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        # print(filepath)
        flag = XianJia_Fenxi(filepath)
        print(flag)
    else:
        retdict = {}
        retdict["totle_num"] = -1
        retdict["totle_area"] = -1
        retdict["quexian"] = False
        print(retdict)
