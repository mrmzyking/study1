import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

from Google_SPP import Google_SPP_Net
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional
import cv2
import math


class SquarePad:
    def __call__(self, image):
        imgsize = image.size()
        c = imgsize[0]
        w = imgsize[2]
        h = imgsize[1]
        max_wh = max(w, h)
        lr = int((max_wh - w) / 2)
        tb = int((max_wh - h) / 2)
        padding = (lr, lr, tb, tb)
        newimg = torch.nn.functional.pad(input=image, pad=padding, mode='constant')
        return newimg


img_transforms = transforms.Compose([transforms.ToTensor(),
                                     SquarePad(),
                                     transforms.Resize([244, 244]),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])
                                     ])
import numpy as np


def load_model(path: str):
    """
    加载训练模型
    :param path: 模型地址
    :return:
    """
    model = Google_SPP_Net()
    model.load_state_dict(path)
    model.eval()
    return model


def DuleImage(imgpath):
    img = cv2.imread(imgpath)
    total_w = img.shape[1]
    total_h = img.shape[0]
    minval = min(total_w, total_h)
    if minval <= 7:
        return None
    fxy = math.ceil(256 / minval)
    if fxy <= 2:
        newimg = cv2.resize(img, dsize=None, fx=2, fy=2)
    else:
        newimg = cv2.resize(img, dsize=None, fx=fxy, fy=fxy)
    return newimg


def precitc(model, imgpath):
    devicetype = ""
    devicetype_qx = False
    # 1.处理图像
    image = DuleImage(imgpath)
    if image is None:
        print("Err : 图像的水平或垂直像素≤7，无法进行分析")
        return None
    iimage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    imgout = img_transforms(iimage)
    # # 1.4 新增维度
    img1 = imgout.view(1, 3, 244, 244)
    # 2.预测输出值
    y = model(img1)
    print(y)
    outputs = torch.argmax(y, dim=1)
    if outputs == 0:
        devicetype = "JueYuanZi"
        devicetype_qx = False
    elif outputs == 1:
        devicetype = "JueYuanZi"
        devicetype_qx = True
    elif outputs == 2:
        devicetype = "XianJia"
        devicetype_qx = False
    elif outputs == 3:
        devicetype = "XianJia"
        devicetype_qx = True
    # elif outputs == 4 :
    #     devicetype = "XianJia_NY"
    #     devicetype_qx = False
    # elif outputs == 5 :
    #     devicetype = "XianJia_NY"
    #     devicetype_qx = True
    # elif outputs == 6 :
    #     devicetype = "XianJia_XieXing_NXL"
    #     devicetype_qx = False
    # elif outputs == 7 :
    #     devicetype = "XianJia_XieXing_NXL"
    #     devicetype_qx = True
    else:
        devicetype = "XianLu"
        devicetype_qx = True
    return (devicetype, devicetype_qx)


if __name__ == '__main__':
    model = Google_SPP_Net()
    model.load_state_dict(torch.load("D:\\python_prj\\study\\studytorch\\myFirstModel.pth"))
    model.eval()
    # (devicetype,qx) = precitc(model,"D:\\python_prj\\study\\studytorch\\shudan\\shudian-QX\\datasets\\5_58df05671637967a27eee6101fdc85b7.png")
    (devicetype, qx) = precitc(model,
                               "D:\\python_prj\\study\\studytorch\\Datasets\\1.png")
    print(devicetype, qx)
    (devicetype, qx) = precitc(model,
                               "D:\\python_prj\\study\\studytorch\\Datasets\\2.png")
    print(devicetype, qx)
    (devicetype, qx) = precitc(model,
                               "D:\\python_prj\\study\\studytorch\\Datasets\\3.png")
    print(devicetype, qx)
    (devicetype, qx) = precitc(model,
                               "D:\\python_prj\\study\\studytorch\\Datasets\\4.jpg")
    print(devicetype, qx)
