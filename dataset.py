import torch
import os
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None, target_transform=None, loader=None):
        # super(MyDataset, self).__init__()
        self.files = os.listdir(dir)
        self.dir = dir
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        # print(file)
        label = float(file[-12:-4])
        if file[-13] == file[-14]:
            label = -label
        with Image.open(self.dir + file) as img:
            img = img.convert('RGB')
        # img.show()
        img = self.transform(img)
        label = self.target_transform(label)
        # print(label)
        # import time
        # time.sleep(4)
        return img, label