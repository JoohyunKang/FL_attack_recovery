import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.act5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.act8 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(1 * 1 * 128, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x1 = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x1))
        x = self.act5(self.conv5(x) + x1)
        x = self.pool2(x)
        x2 = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x2))
        x = self.act8(self.conv8(x) + x2)
        x = self.pool3(x)
        x = x.view(-1, 128 * 1 * 1)
        x = self.fc1(x)
        return x


# # LeNet Model definition
# class MNIST_CNN(nn.Module):
#     def __init__(self):
#         super(MNIST_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# https://pytorch.org/tutorials/beginner/nn_tutorial.html
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


class ResNetBN(nn.Module):
    def __init__(self):
        super(ResNetBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(1 * 1 * 128, 10)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x1 = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x1)))
        x = self.act5(self.bn5(self.conv5(x) + x1))
        x = self.pool2(x)
        x2 = self.act6(self.bn6(self.conv6(x)))
        x = self.act7(self.bn7(self.conv7(x2)))
        x = self.act8(self.bn8(self.conv8(x) + x2))
        x = self.pool3(x)
        x = x.view(-1, 128 * 1 * 1)
        x = self.fc1(x)
        return x

class CNN10(nn.Module):
    def __init__(self):
        super(CNN10, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 5)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(8, 16, 5)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 5)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 2 * 2)
        x = self.fc1(x)
        return x


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.act5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.act8 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(128 * 1 * 1, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.pool2(x)
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.act8(self.conv8(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 1 * 1)
        x = self.fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)
        # self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # make inputs flat
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

