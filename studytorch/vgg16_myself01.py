import torch
import torch.nn as nn


class BasicModle1(nn.Module):
    def __init__(self, inchannels, outchannels, flag: bool):
        super(BasicModle1, self).__init__()
        self.flag = flag
        self.basicmodel = None
        if flag is not True:
            self.basicmodel = nn.Sequential(
                nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
        else:
            self.basicmodel = nn.Sequential(
                nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

    def forward(self, x):
        x = self.basicmodel(x)
        return x


VGG16_List = [[3, 64, False],
              [64, 128, False],
              [128, 256, True],
              [256, 512, True],
              [512, 512, True], ]


class VGG16_Myself1(nn.Module):
    """
    用模块化的思维实现VGG16模块
    """

    def __init__(self, vgg_list, basicmode):
        super(VGG16_Myself1, self).__init__()
        self.vgglist = vgg_list
        self.basicmodle = basicmode
        self.vggnet = None
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def Build_model(self):
        lay = []
        for i in self.vgglist:
            inchannels = i[0]
            outchannels = i[1]
            flag = i[2]
            lay.append(self.basicmodle(inchannels=inchannels, outchannels=outchannels, flag=flag))
        self.vggnet = nn.Sequential(*lay)

    def forward(self, x):
        x = self.vggnet(x)
        x = self.linear(x)
        return x


class VGG16_Myself(nn.Module):
    """
    用顺序方式实现VGG16
    """

    def __init__(self, ):
        super(VGG16_Myself, self).__init__()
        self.basicmodel_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.basicmodel_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.basicmodel_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.basicmodel_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.basicmodel_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.basicmodel_1(x)
        x = self.basicmodel_2(x)
        x = self.basicmodel_3(x)
        x = self.basicmodel_4(x)
        x = self.basicmodel_5(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    # net = VGG16_Myself()
    # print(net)
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(y)
    # net = VGG16_Myself1(vgg_list=VGG16_List,basicmode=BasicModle1)
    # net.Build_model()
    # print(net)
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(y)
    a = torch.randn((1, 64, 35, 35))
    b = torch.randn((1, 64, 35, 35))
    c = torch.randn((1, 96, 35, 35))
    d = torch.randn((1, 32, 35, 35))
    e = torch.cat((a, b, c, d),dim=1)
    print(e)
