from googlenet_myself import *
from SPP_NET import *
class Google_SPP_Net(nn.Module):
    def __init__(self):
        super(Google_SPP_Net, self).__init__()
        self.google_sppnet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, dilation=1),
            nn.BatchNorm2d(64, 0.001),
            nn.Sigmoid(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(64, 0.001),
            nn.Sigmoid(),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(192, 0.001),
            nn.Sigmoid(),

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
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            # nn.Linear(in_features=1024, out_features=9),
            nn.Linear(in_features=1024, out_features=4),
            nn.Softmax(dim=1),
        )

        # self.spp_net = nn.Sequential(
        #     SPPLayer(4),
        #     nn.Linear(in_features=30720, out_features=500),
        #     nn.Dropout(0.4),
        #     nn.Sigmoid(),
        #     nn.Linear(in_features=500, out_features=4),
        #     nn.Dropout(0.4),
        #     nn.Softmax(dim=1),
        # )

    def forward(self,x):
        x = self.google_sppnet1(x)
        # x = self.spp_net(x)
        return x



if __name__ == '__main__':
    net = Google_SPP_Net()
    x = torch.randn(size=(1, 3, 640, 512))
    y = net(x)
    print(y)
