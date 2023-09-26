import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

from ShuDianHW_JYZ_Model import ShuDianHongWai
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional
import cv2
import math

from studytorch.tuxiangengqiang import upsample


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
                                     transforms.Resize([224, 224]),
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
    model = ShuDianHongWai()
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
    newimg = upsample(img,256,256)
    # fxy = math.ceil(256 / minval)
    # if fxy <= 2:
    #     newimg = cv2.resize(img, dsize=None, fx=2, fy=2)
    # else:
    #     newimg = cv2.resize(img, dsize=None, fx=fxy, fy=fxy)
    cv2.imshow("new1", newimg)
    # newimg = cv2.transpose(newimg)
    # 对比度增强
    # cv2.imshow("new2",newimg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img


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
    img1 = imgout.view(1, 3, 224, 224)
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
    model = ShuDianHongWai()
    model.load_state_dict(torch.load("D:\\python_prj\\study\\studytorch\\JYZ_MODEL.pth"))
    model.eval()
    # (devicetype, qx) = precitc(model,
    #                            "D:\\python_prj\\study\\studytorch\\yuanshituxiang\\1_c946b9a1f5cf170a3d360f693c441e1a.png")
    # print(devicetype, qx)
    # (devicetype, qx) = precitc(model,
    #                            "D:\\python_prj\\study\\studytorch\\Datasets\\728347708_0.562.png")
    # print(devicetype, qx)

    (devicetype, qx) = precitc(model,
                               "D:\\python_prj\\study\\studytorch\\shudiandeyang3316\\datasets\\2_47abcd0e7031f652282f381019e156d9.png")
    print(devicetype, qx)
    # (devicetype, qx) = precitc(model,
    #                            "D:\\python_prj\\study\\studytorch\\Datasets\\523875087_0.358.png")
    # print(devicetype, qx)
    # (devicetype, qx) = precitc(model,
    #                            "D:\\python_prj\\study\\studytorch\\Datasets\\593628118_0.562.png")
    # print(devicetype, qx)
    # (devicetype, qx) = precitc(model,
    #                            "D:\\python_prj\\study\\studytorch\\Datasets\\1996063452_0.611.jpg")
    # print(devicetype, qx)
