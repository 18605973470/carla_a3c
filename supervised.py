#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset
from supervised_model import SModel

prefix = "/home/r720/Data/imagedata/"
traindir = prefix + "train/"
valdir = prefix + "val/"


def train_label_transformer(data):
    return data * (1 + (np.random.randint(-5, 6) / 100) )


def fake_transformer(data):
    return data


def train_img_transformer(data):
    return data / 127.5 - 1


img_trans = transforms.Compose([
    # transforms.Resize(60, 200),
    transforms.ToTensor()
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

label_trans = transforms.Compose([
    train_label_transformer
])
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # print(data.shape, target.shape)
        target = target.reshape((target.shape[0], 1))
        target = target.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        # print(target)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def val(model, device, val_loader):
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = nn.MSELoss(output, target)
        print('Train Epoch: {} Val Loss: {:.7f}'.format(epoch, loss.item()))
        break

device = torch.device('cuda:0')
model = SModel()
model.load_state_dict(torch.load("nvidiasmodel/nvidia-200.dat"))
model.to(device)
model.train()

traindata = MyDataset(traindir, transform=img_trans, target_transform=label_trans)
valdata = MyDataset(valdir, transform=img_trans, target_transform=transforms.ToTensor)
train_loader = DataLoader(traindata, 64, True)
val_loader = DataLoader(valdata, 64, True)

epochs = 300
optimizer = optim.Adam(model.parameters(), lr=0.00001)
for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    if epoch % 100 == 0:
        # model.eval()
        # for images, labels in dataloader(valdir, valfiles, len(valfiles), "val"):
        #     ps = model(images)
        #     print("epoch {} val loss {}".format(i, loss_func(ps, labels)))
        state = model.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(state, 'nvidiasmodel/{0}-{1}.dat'.format("nvidia", epoch))