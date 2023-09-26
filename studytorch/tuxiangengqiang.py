from PIL import Image
import cv2
import os
import hashlib
import numpy as np
import time
import math
from random import sample


def CreatImage(max_w, max_h):
    img = np.zeros([max_w, max_h, 3], np.uint8)  # 创建一个图片，规定长、宽、通道数 数值，类型
    img[:, :, 0] = np.ones([max_w, max_h]) * 0  # 设定第1个通道数值
    img[:, :, 1] = np.ones([max_w, max_h]) * 0  # 设定第2个通道数值
    img[:, :, 2] = np.ones([max_w, max_h]) * 0  # 设定第3个通道数值
    # cv2.imshow("new image", img)
    # cv2.waitKey(0)  # 保持界面框
    # cv2.destroyWindow('new image')  # 清除内存
    return img


def FullImageToCreateImg(imgPath, max_w, max_h):
    """
    创建一个图像，并把imgPath图像填充到新创建的图像中去
    :param imgPath: 图像地址
    :param max_w: 创建图像的最大宽
    :param max_h: 创建图像的最大高
    :return: 返回新创建的图像
    """
    img1 = cv2.imread(imgPath)
    total_w = img1.shape[1]
    total_h = img1.shape[0]
    if total_w % 2 != 0:
        img1 = cv2.resize(img1, (total_w + 1, total_h))
        total_w += 1
    if total_h % 2 != 0:
        img1 = cv2.resize(img1, (total_w, total_h + 1))
        total_h += 1
    if total_w > max_w:
        img1 = cv2.resize((max_w, total_h))
        total_w = max_w
    if total_h > max_h:
        img1 = cv2.resize((total_w, max_h))
        total_h = max_h
    cimg = CreatImage(max_w, max_h)
    center_x = int(max_w / 2)
    center_y = int(max_h / 2)
    W = int(total_w / 2)
    H = int(total_h / 2)
    # 1.求(X0,Y0)左上角顶点坐标
    X0 = center_x - W
    Y0 = center_y - H
    # 2.求(X1,Y1)右下角顶点坐标
    X1 = center_x + W
    Y1 = center_y + H
    cimg[Y0:Y1, X0:X1] = img1
    return cimg


def FullImageToImg(img_src, max_w, max_h):
    """
    创建一个图像，并把imgPath图像填充到新创建的图像中去
    :param imgPath: 图像地址
    :param max_w: 创建图像的最大宽
    :param max_h: 创建图像的最大高
    :return: 返回新创建的图像
    """
    total_w = img_src.shape[1]
    total_h = img_src.shape[0]
    if total_h < (max_h / 3):
        img_src = cv2.resize(img_src, (total_w, 3 * total_h))
        total_h = 3 * total_h
    if total_w < (max_w / 3):
        img_src = cv2.resize(img_src, (3 * total_w, total_h))
        total_w = 3 * total_w
    if total_w % 2 != 0:
        img_src = cv2.resize(img_src, (total_w + 1, total_h))
        total_w += 1
    if total_h % 2 != 0:
        img_src = cv2.resize(img_src, (total_w, total_h + 1))
        total_h += 1
    if total_w > max_w:
        img_src = cv2.resize(img_src, (max_w, total_h))
        total_w = max_w
    if total_h > max_h:
        img_src = cv2.resize(img_src, (total_w, max_h))
        total_h = max_h

    cimg = CreatImage(max_w, max_h)
    center_x = int(max_w / 2)
    center_y = int(max_h / 2)
    W = int(total_w / 2)
    H = int(total_h / 2)
    # 1.求(X0,Y0)左上角顶点坐标
    X0 = center_x - W
    Y0 = center_y - H
    # 2.求(X1,Y1)右下角顶点坐标
    X1 = center_x + W
    Y1 = center_y + H
    cimg[Y0:Y1, X0:X1] = img_src
    return cimg


def GenerateImage(imgpath, number):
    img = cv2.imread(imgpath)
    jiaodu = 360.0 / number
    for i in range(number):
        newimg = rotate_bound(img, i * jiaodu)
        # newimg = cv2.resize(newimg,(128,128))
        newimg = FullImageToImg(newimg, 128, 128)
        cv2.imshow("newimg", newimg)
        cv2.waitKey()
        cv2.destroyAllWindows()


TianChong = [
    (113, 0, 69),
    (126, 0, 137),
    (91, 0, 123),
    (141, 0, 144)
]


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    # R_channel = 0
    # G_channel = 0
    # B_channel = 0
    # B_channel = B_channel + np.sum(image[:, :, 0])
    # G_channel = G_channel + np.sum(image[:, :, 1])
    # R_channel = R_channel + np.sum(image[:, :, 2])
    (h, w) = image.shape[:2]
    bgrlst = []
    bgr1 = image[0, 0]
    bgrlst.append(bgr1)
    bgr2 = image[0,w-1]
    bgrlst.append(bgr2)
    bgr3 = image[h-1, 0]
    bgrlst.append(bgr3)
    bgr4 = image[h-1, w-1]
    bgrlst.append(bgr4)
    # print(bgr1, bgr2, bgr3, bgr4)
    bgrchoice = sample(bgrlst,1)
    # r_color = int(R_channel / (h*w))
    # g_color = int(G_channel / (h*w))
    # b_color = int(B_channel / (h * w))
    # boadval = (b_color ,g_color,r_color)
    boadval = (int(bgrchoice[0][0]),int(bgrchoice[0][1]),int(bgrchoice[0][2]))
    (cX, cY) = (w / 2, h / 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # boadval = sample(TianChong, 1)
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=tuple(boadval))

def shuipingfanzhuan(img=None,imgpath=None):
    retimg = None
    if img is None:
        pass
    else:
        retimg = cv2.flip(img, 1)
    if imgpath is None:
        pass
    else:
        retimg = cv2.imread(imgpath)
        retimg = cv2.flip(retimg, 1)
    return retimg

def chuizhifanzhuan(img=None,imgpath=None):
    retimg = None
    if img is None:
        pass
    else:
        retimg = cv2.flip(img,0)
    if imgpath is None:
        pass
    else:
        retimg = cv2.imread(imgpath)
        retimg = cv2.flip(retimg,0)
    return retimg
def xuanzhuanimage(img=None,imgpath=None):
    retimg = None
    if img is None:
        pass
    else:
        retimg = cv2.flip(img, 1)
    if imgpath is None:
        pass
    else:
        retimg = cv2.imread(imgpath)
        retimg = cv2.flip(retimg, 1)
    return retimg
# def upcaiyang(imgpath, maxw, maxh):
#     img = cv2.imread(imgpath)
#     (h, w) = img.shape[:2]
#     if h >= maxh:
#         pass
#     if w >= maxw:
#         pass


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
    # new1img = cv2.resize(newimg, (maxw, maxh))
    # cv2.imshow("newimg1", new1img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # GenerateImage("D:\\python_prj\\study\\studytorch\\yuanshituxiang\\5_86ef32ff269eb5f1109350438be0e82c.png", 360)
    upsample("D:\\python_prj\\study\\studytorch\\Datasets\\728347708_0.562.png", 224, 224)
    # img = cv2.imread("D:\\python_prj\\study\\studytorch\\yuanshituxiang\\0_0d0c7bd9d2e2c80282092ce4f4eb6827.png")
    # img = cv2.resize(img,dsize=None,fx=4,fy=4)
    # (h, w) = img.shape[:2]
    # newimg = rotate_bound(img, 45)
    # cv2.imshow("oldimg", img)
    # # newimg = cv2.pyrUp(img,dstsize=(w*2,h*2))
    # cv2.imshow("newimg", newimg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cimg = FullImageToCreateImg(
    #     imgPath="D:\\python_prj\\study\\studytorch\\yuanshituxiang\\0_00ba09ad17dd5b7b0b7d87d855274c69.png",
    #     max_w=244,
    #     max_h=244)
    # cv2.imshow("img", cimg)
    # cv2.waitKey()
    # cv2.destroyWindow("img")
