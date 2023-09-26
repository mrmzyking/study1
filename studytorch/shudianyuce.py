import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from studytorch.googlenet_myself import GoogLeNet_myself
from studytorch.tuxiangengqiang import *
img_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5,0.5,0.5],
                                                            std=[0.5,0.5,0.5])
                                       ])
import numpy as np

def load_model(path: str):
    """
    加载训练模型
    :param path: 模型地址
    :return:
    """
    model = GoogLeNet_myself()
    model.load_state_dict(path)
    model.eval()
    return model


def precitc(model, imgpath):
    devicetype = ""
    devicetype_qx = False
    # 1.处理图像
    image = FullImageToCreateImg(imgpath,64,64)
    # cv2.imshow("image",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    iimage = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    # img1 = np.array([img])
    # input = torch.from_numpy(img).unsqueeze(0)
    # input = torch.Tensor(input)
    imgout = img_transforms(iimage)
    # # 1.4 新增维度
    img1 = imgout.view(1, 3, 64, 64)
    # 2.预测输出值
    y = model(img1)
    outputs = torch.argmax(y, dim=1)
    if outputs == 0 :
        devicetype = "JueYuanZi"
        devicetype_qx = False
    elif outputs == 1 :
        devicetype = "JueYuanZi"
        devicetype_qx = True
    elif outputs == 2 :
        devicetype = "XianJia_LuoShuanShi"
        devicetype_qx = False
    elif outputs == 3 :
        devicetype = "XianJia_LuoShuanShi"
        devicetype_qx = True
    elif outputs == 4 :
        devicetype = "XianJia_NY"
        devicetype_qx = False
    elif outputs == 5 :
        devicetype = "XianJia_NY"
        devicetype_qx = True
    elif outputs == 6 :
        devicetype = "XianJia_XieXing_NXL"
        devicetype_qx = False
    elif outputs == 7 :
        devicetype = "XianJia_XieXing_NXL"
        devicetype_qx = True
    else:
        devicetype = "XianLu"
        devicetype_qx = True
    return (devicetype,devicetype_qx)

if __name__ == '__main__':
    model = GoogLeNet_myself()
    model.load_state_dict(torch.load("D:\\python_prj\\study\\studytorch\\myFirstModel.pth"))
    model.eval()
    (devicetype,qx) = precitc(model,"D:\\python_prj\\study\\studytorch\\shudian-300\\datasets\\5_e7c29ebc782ecf7084863739a91721ce.png")
    print(devicetype,qx)
