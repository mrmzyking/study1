import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

from googlenet_myself import GoogLeNet_myself
from torch.utils.data.sampler import WeightedRandomSampler

train_transforms = transforms.Compose([transforms.ToTensor(),
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
        labels = torch.zeros(6, dtype=torch.float32)
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


def calculateWeights(label_dict, d_set):
    arr = []
    for label, count in label_dict.items():
        weight = len(d_set) / count
        arr.append(weight)
    return arr





if __name__ == '__main__':
    # 1.加载训练数据集
    # samples_weight = [1.0 / 2258, 1.0 / 128, 1.0 / 67, 1.0 / 13, 1.0 / 317, 1.0 / 165]
    # weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # samples_weight = [1.0 / (2258 / 2839), 1.0 / (128/ 2839), 1.0 / (67/ 2839), 1.0 / (13/ 2839), 1.0 / (317/ 2839), 1.0 / (165/ 2839)]
    # samples_weight = [0.0041, 0.0715, 0.1365, 0.7037,0.0289, 0.0554]
    #
    # samples_num = 6
    # sampler1 = WeightedRandomSampler(samples_weight, 2948, replacement=True)
    Train_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\train\\datasets.txt",
                               transforms=train_transforms)
    Train_dataloader = DataLoader(Train_dataset, batch_size=10, shuffle=False, num_workers=3)
    # calculateWeights()
    # 2.加载测试数据集
    Test_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\test\\datasets.txt",
                              transforms=train_transforms)
    Test_dataloader = DataLoader(Test_dataset, batch_size=10, shuffle=False, num_workers=3)
    # 3.定义网络模型
    model = GoogLeNet_myself()
    # 4.定义损失函数
    criterion = torch.nn.MSELoss()
    # 5.定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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
                print('[%d, %5d] loss: %.3f' %
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
            path = "./myFirstModel.pth"
            torch.save(model.state_dict(), path)
            best_accuracy = accuracy
    print('Finished Training')
