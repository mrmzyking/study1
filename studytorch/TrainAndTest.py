from studytorch.googlenet_myself import GoogLeNet_myself
from studytorch.MyDatasets import *
from torch.utils.data import DataLoader
import torch
# 1.加载训练数据集
Train_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\test\\datasets.txt",
                           transforms=train_transforms)
Train_dataloader = DataLoader(Train_dataset, batch_size=5, shuffle=True, num_workers=2)
# 2.加载测试数据集
Test_dataset = MyDataSets("D:\\python_prj\\study\\studytorch\\Datasets\\test\\datasets.txt",
                          transforms=train_transforms)
Test_dataloader = DataLoader(Test_dataset, batch_size=5, shuffle=False, num_workers=2)
# 3.定义网络模型
model = GoogLeNet_myself()
# 4.定义损失函数
criterion = torch.nn.MSELoss(reduction='sum')
# 5.定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in Test_dataloader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (outputs == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return (accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(Train_dataloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

if __name__ == '__main__':
    # Let's build our model
    train(5)
    print('Finished Training')


