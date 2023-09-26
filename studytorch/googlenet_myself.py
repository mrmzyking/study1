import torch
import torch.nn as nn
from torchvision import models
# from torchsummary import summary


class Google_Inception3A(nn.Module):
    """
    带池化层的模块
    """

    def __init__(self, in1_1, out1_1, out2_1, out2_2, out3_1, out3_2, out3_3, out4_1):
        super(Google_Inception3A, self).__init__()
        self.pool_lay = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.batch1_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out1_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out1_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch2_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out2_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out2_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out2_1, out_channels=out2_2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out2_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch3_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out3_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out3_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out3_1, out_channels=out3_2, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.BatchNorm2d(out3_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out3_2, out_channels=out3_3, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out3_3, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch4_lay = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(in_channels=in1_1, out_channels=out4_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out4_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )

    def forward(self, x):
        basicy = self.pool_lay(x)
        lay1num = self.batch1_lay(basicy)
        lay2num = self.batch2_lay(basicy)
        lay3num = self.batch3_lay(basicy)
        lay4num = self.batch4_lay(basicy)
        y = torch.cat((lay1num, lay2num, lay3num, lay4num), dim=1)
        return y


class Google_Inception3B(nn.Module):
    """
    不带池化层的模块
    """

    def __init__(self, in1_1, out1_1, out2_1, out2_2, out3_1, out3_2, out4_1):
        super(Google_Inception3B, self).__init__()
        self.batch1_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out1_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out1_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch2_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out2_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out2_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out2_1, out_channels=out2_2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out2_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch3_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out3_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out3_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out3_1, out_channels=out3_2, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.BatchNorm2d(out3_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch4_lay = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(in_channels=in1_1, out_channels=out4_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out4_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )

    def forward(self, x):
        lay1num = self.batch1_lay(x)
        lay2num = self.batch2_lay(x)
        lay3num = self.batch3_lay(x)
        lay4num = self.batch4_lay(x)
        y = torch.cat((lay1num, lay2num, lay3num, lay4num), dim=1)
        return y


class Google_Inception45A(nn.Module):
    """
    带池化层的模块
    """

    def __init__(self, in1_1, out1_1, out2_1, out2_2, out3_1, out3_2, out4_1):
        super(Google_Inception45A, self).__init__()
        self.poollay = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.batch1_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out1_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out1_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch2_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out2_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out2_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out2_1, out_channels=out2_2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out2_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch3_lay = nn.Sequential(
            nn.Conv2d(in_channels=in1_1, out_channels=out3_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out3_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out3_1, out_channels=out3_2, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.BatchNorm2d(out3_2, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )
        self.batch4_lay = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(in_channels=in1_1, out_channels=out4_1, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out4_1, eps=0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.poollay(x)
        lay1num = self.batch1_lay(x)
        lay2num = self.batch2_lay(x)
        lay3num = self.batch3_lay(x)
        lay4num = self.batch4_lay(x)
        y = torch.cat((lay1num, lay2num, lay3num, lay4num), dim=1)
        return y


class GoogLeNet_myself(nn.Module):
    def __init__(self):
        super(GoogLeNet_myself, self).__init__()
        self.googlenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, dilation=1),
            nn.BatchNorm2d(64, 0.001),
            nn.Sigmoid(),
            # nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(64, 0.001),
            nn.Sigmoid(),
            # nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(192, 0.001),
            nn.Sigmoid(),
            # nn.ReLU(),
            Google_Inception3A(in1_1=192, out1_1=64, out2_1=96, out2_2=128, out3_1=16, out3_2=32, out3_3=32,
                               out4_1=32),
            # 3a
            Google_Inception3B(in1_1=256, out1_1=128, out2_1=128, out2_2=192, out3_1=32, out3_2=96, out4_1=64),
            # 3b
            Google_Inception45A(in1_1=480, out1_1=192, out2_1=96, out2_2=208, out3_1=16, out3_2=48, out4_1=64),
            # 4a
            Google_Inception3B(in1_1=512, out1_1=160, out2_1=112, out2_2=224, out3_1=24, out3_2=64, out4_1=64),
            # 4b
            Google_Inception3B(in1_1=512, out1_1=128, out2_1=128, out2_2=256, out3_1=24, out3_2=64, out4_1=64),
            # 4c
            Google_Inception3B(in1_1=512, out1_1=112, out2_1=144, out2_2=288, out3_1=32, out3_2=64, out4_1=64),
            # 4d
            Google_Inception3B(in1_1=528, out1_1=256, out2_1=160, out2_2=320, out3_1=32, out3_2=128, out4_1=128),
            # 4e
            Google_Inception45A(in1_1=832, out1_1=256, out2_1=160, out2_2=320, out3_1=32, out3_2=128, out4_1=128),
            # 5a
            Google_Inception3B(in1_1=832, out1_1=384, out2_1=192, out2_2=384, out3_1=48, out3_2=128, out4_1=128),
            # 5b
            nn.AvgPool2d(kernel_size=4, stride=1, padding=0),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            # nn.Linear(in_features=1024, out_features=9),
            nn.Linear(in_features=1024, out_features=6),
            nn.Softmax(dim=1),
            # nn.Flatten(start_dim=0)
        )

    def forward(self, x):
        x = self.googlenet(x)
        # x = nn.Softmax(x)

        return x


if __name__ == '__main__':
    gnet = GoogLeNet_myself()
    # print(gnet)
    # summary(gnet, (3, 224, 224))
    x = torch.randn((5, 3, 244, 244))
    y = gnet(x)
    z = torch.squeeze(y)
    print(y)
    print(z)
