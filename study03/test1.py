import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
class simpleNet(nn.Module):

    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def get_name(self):
        return self.__class__.__name__

class activationNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(activationNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    def get_name(self):
        return self.__class__.__name__

class batchNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(batchNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim,n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
            )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True),
            )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



    def get_name(self):
        return self.__class__.__name__
def train(net,train_data,valid_data,num_epoches,optimizer,criterion):
    length = len(train_data)
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for iter,data in enumerate(train_data):
            im, label = data
            im = im.view(im.size()[0], -1)
            im = Variable(im)
            label = Variable(label)
            output = net(im)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred_label = torch.max(output.data,1)
            train_loss += loss.item()
            temp_loss = loss.item()
            train_acc += torch.sum(pred_label == label.data)

            temp_acc = (torch.sum(pred_label == label.data)) / label.size(0)
            if iter % 300 == 0 and iter > 0:
                print('Epoch {}/{},Iter {}/{} Loss: {:.4f},ACC:{:.4f}' \
                      .format(epoch, num_epoches - 1,iter,length,temp_loss,temp_acc))
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for data in valid_data:
                im,label = data
                im = im.view(im.size()[0], -1)
                im = Variable(im,volatile=True)
                label = Variable(label,volatile=True)
                output = net(im)
                _, pred_label = torch.max(output.data,1)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += torch.sum(pred_label == label.data)
            print('Epoch {}/{},complete! train_loss: {:.4f},train_acc:{:.4f}' \
                  .format(epoch, num_epoches - 1,train_loss, train_acc/60000),
                  'valid_loss: {:.4f},valid_acc:{:.4f}'.format(valid_loss,valid_acc/10000)
                  )
        else:
            pass
if __name__ == '__main__':
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
    )
    train_dataset = datasets.MNIST(
        root='../data', train=True, transform=data_tf, download=True
    )
    test_dataset = datasets.MNIST(
        root="../data", train=False, transform=data_tf
    )

    batch_size = 64
    learning_rate = 1e-2
    num_epoches = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    in_dim, n_hidden_1, n_hidden_2, out_dim = 28 * 28, 300, 100, 10

    model1 = simpleNet(in_dim, n_hidden_1, n_hidden_2, out_dim)
    model2 = activationNet(in_dim, n_hidden_1, n_hidden_2, out_dim)
    model3 = batchNet(in_dim, n_hidden_1, n_hidden_2, out_dim)

    for model in [model1]:
        print("the {} start traing...".format(model.get_name()))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
        train(model, train_loader, test_loader, 20, optimizer, criterion)
        print("the {} complete traing...".format(model.get_name()))