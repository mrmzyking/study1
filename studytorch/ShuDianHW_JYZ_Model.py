import math

import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2

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
    fxy = math.ceil(256 / minval)
    if fxy <= 2:
        newimg = cv2.resize(img, dsize=None, fx=2, fy=2)
    else:
        newimg = cv2.resize(img, dsize=None, fx=fxy, fy=fxy)
    return newimg

class ShuDianHongWai(nn.Module):
    def __init__(self):
        super(ShuDianHongWai,self).__init__()
        self.Lay1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32*7*7,out_features=1000),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(in_features=1000,out_features=100),
            nn.Dropout(0.4),
            nn.Sigmoid(),
            nn.Linear(in_features=100,out_features=2),
            nn.Softmax(),
        )

    def forward(self,x):
        y = self.Lay1(x)
        return y

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
    else:
        devicetype = "JueYuanZi"
        devicetype_qx = True
    return (devicetype, devicetype_qx)


train_transforms = transforms.Compose([transforms.ToTensor(),
                                       SquarePad(),
                                       transforms.Resize([244,244]),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])
                                       ])
class MyDataSets(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = self.get_img_info(data_dir)
        self.transforms = transforms

    def __getitem__(self, item):
        path_img, label = self.data_info[item]
        image = Image.open(path_img).convert('RGB')
        labels = torch.zeros(2, dtype=torch.float32)
        labels[int(label)] = 1.0
        if self.transforms is not None:
            image = self.transforms(image)
        return image, labels

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        # path_dir = os.path.join(data_dir, 'train_dataset.txt')
        self.data_info = []
        with open(data_dir) as file:
            lines = file.readlines()
            for line in lines:
                self.data_info.append(line.strip('\n').split(' '))
        return self.data_info

if __name__ == '__main__':
    # 1.加载训练数据集
    Train_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\JYZ\\train\\datasets.txt",
                               transforms=train_transforms)
    Train_dataloader = DataLoader(Train_dataset, batch_size=10, shuffle=True, num_workers=3)
    # calculateWeights()
    # 2.加载测试数据集
    Test_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\JYZ\\test\\datasets.txt",
                              transforms=train_transforms)
    Test_dataloader = DataLoader(Test_dataset, batch_size=10, shuffle=False, num_workers=3)
    # 3.定义网络模型
    model = ShuDianHongWai()
    # 4.定义损失函数
    criterion = torch.nn.MSELoss()
    # 5.定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 6.训练数据集
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for i, (images, labels) in enumerate(Train_dataloader, 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            labels = torch.tensor(labels, dtype=torch.float32)
            # 梯度清零
            optimizer.zero_grad()
            # 预测数据
            outputs = model(images)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            # 计算每100张照片的预测准确率
            running_loss += loss.item()  # extract the loss value
            if i % 20 == 19:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 20))
                # zero the loss
                running_loss = 0.0

        # 利用测试数据集进行测试
        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in Test_dataloader:
                images, labels = data
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                labels = torch.tensor(labels, dtype=torch.float32)
                # run the model on the test set to predict labels
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
                # the label with the highest energy will be our prediction
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (outputs == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            path = "./JYZ_MODEL.pth"
            torch.save(model.state_dict(), path)
            best_accuracy = accuracy
    print('Finished Training')
