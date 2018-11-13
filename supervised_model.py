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
        self.cnn = models.vgg16_bn(pretrained=True).features
        self.fc1 = nn.Linear(76800, 1024)
        self.lru1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 1)
        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.lru1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
