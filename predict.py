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

IMAGE_SIZE = 112
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # resize to IMAGE_SIZE x ? or ? x IMAGE_SIZE
        height = img.shape[0]
        width = img.shape[1]
        rate = IMAGE_SIZE / max(height, width)
        height = int(height * rate)
        width = int(width * rate)
        img = cv2.resize(img, (width, height))
        # pad black
        # from https://blog.csdn.net/qq_20622615/article/details/80929746
        W, H = IMAGE_SIZE, IMAGE_SIZE
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


# In[4]:


# 分別將 training set、validation set、testing set 用 readfile 函式讀進來
try:
    workspace_dir = sys.argv[1]
except:
    workspace_dir = './food-11'
print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))


# # Dataset
# 在 PyTorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。
# 
# Dataset 需要 overload 兩個函數：\_\_len\_\_ 及 \_\_getitem\_\_
# 
# \_\_len\_\_ 必須要回傳 dataset 的大小，而 \_\_getitem\_\_ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
# 
# 實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
# 

# In[5]:


# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
#     transforms.Normalize(
#         [77.89311144813877 / 255, 102.3587941606983 / 255, 126.59376063616554 / 255],
#         [72.80305392379675 / 255, 75.35438507973123 / 255, 79.31408066842762 / 255]
#     )
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
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


# In[6]:


batch_size = 64
# # Model

# In[7]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, IMAGE_SIZE, IMAGE_SIZE]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
 

model_best = torch.load('model.torch')
print('model loaded')

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print('predicting')
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
print('predicted')

try:
    output_fpath = sys.argv[2]
except:
    output_fpath = "predict.csv"
di = os.path.dirname(output_fpath)
if di != '':
    os.makedirs(di, exist_ok=True)

with open(output_fpath, 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

