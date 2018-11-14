import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        pass
        # weight_shape = list(m.weight.data.size())
        # fan_in = np.prod(weight_shape[1:4])
        # fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        # w_bound = np.sqrt(6. / (fan_in + fan_out))
        # m.weight.data.uniform_(-w_bound, w_bound)
        # m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



class SModel(torch.nn.Module):
    def __init__(self):
        super(SModel, self).__init__()
        # self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.lrelu5 = nn.LeakyReLU(0.1)

        # self.cnn = models.vgg16_bn(pretrained=True).features
        self.fc1 = nn.Linear(1152, 1164)
        self.lrelu6 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(1164, 100)
        self.lrelu7 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(100, 50)
        self.lrelu8 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(50, 10)
        self.lrelu9 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(10, 1)

        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = x.view(x.size(0), -1)

        x = self.lrelu6(self.fc1(x))
        x = self.lrelu7(self.fc2(x))
        x = self.lrelu8(self.fc3(x))
        x = self.lrelu9(self.fc4(x))
        x = self.dropout1(x)
        x = self.fc5(x)
        return x
