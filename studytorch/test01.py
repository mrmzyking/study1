import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
        nn.Conv2d(  in_channels: int,
                    out_channels: int,
                    kernel_size: _size_2_t,
                    stride: _size_2_t = 1,
                    padding: Union[str, _size_2_t] = 0,
                    dilation: _size_2_t = 1,
                    groups: int = 1,
                    bias: bool = True,
                    padding_mode: str = 'zeros',  # TODO: refine this type
                    device=None,
                    dtype=None
                )
        
        """
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = self.conv1(x)  # 输出  （1，6，30，30）
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 输出（1，6，15，15）
        # x = F.max_pool2d(F.relu((self.conv2(x))),2)
        x = self.conv2(x)  # 输出（1，16，13，13）
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 输出（1，16，6，6）
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
# m = nn.MaxPool2d((3, 3), stride=(1, 1))
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01)
input = torch.randn(1, 1, 32, 32)
output = net(input)
y = torch.randn(1,10)
wucha = loss(output,y)
optimizer.zero_grad()
wucha.backward()
print(net.conv1.bias.grad)
optimizer.step()

