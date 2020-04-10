#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import time

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # resize to 128 x ? or ? x 128
        height = img.shape[0]
        width = img.shape[1]
        rate = 128 / max(height, width)
        height = int(height * rate)
        width = int(width * rate)
        img = cv2.resize(img, (width, height))
        # pad black
        # from https://blog.csdn.net/qq_20622615/article/details/80929746
        W, H = 128, 128
        top = (H - height) // 2
        bottom = (H - height) // 2
        if top + bottom + height < H:
            bottom += 1
        left = (W - width) // 2
        right = (W - width) // 2
        if left + right + width < W:
            right += 1
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # to np array
        x[i, :, :] = img
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


# In[14]:


try:
    workspace_dir = sys.argv[1]
except:
    workspace_dir = './food-11'
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
    transforms.Normalize(
        [77.89311144813877 / 255, 102.3587941606983 / 255, 126.59376063616554 / 255],
        [72.80305392379675 / 255, 75.35438507973123 / 255, 79.31408066842762 / 255]
    )
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.PReLU(1),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(128, 100),
            nn.ReLU(),
            
            nn.Linear(100, 30),
            nn.PReLU(1),

            nn.Linear(30, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


model_best = torch.load('model.torch')

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

try:
    output_fpath = sys.argv[2]
except:
    output_fpath = "predict.csv"
di = os.path.dirname(output_fpath)
if di != '':
    os.makedirs(di, exist_ok=True)

with open(output_fpath, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

