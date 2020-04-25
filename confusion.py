#!/usr/bin/env python
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



# 分別將 training set、validation set、testing set 用 readfile 函式讀進來
try:
    workspace_dir = sys.argv[1]
except:
    workspace_dir = './food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))


# training 時做 data augmentation
train_transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomChoice([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective()
    ]),
    transforms.RandomChoice([
        transforms.RandomAffine(10), # 隨機線性轉換
        transforms.RandomRotation(40)
    ]),
    transforms.ColorJitter(), # 隨機色溫等
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
#     transforms.Normalize(
#         [77.89311144813877 / 255, 102.3587941606983 / 255, 126.59376063616554 / 255],
#         [72.80305392379675 / 255, 75.35438507973123 / 255, 79.31408066842762 / 255]
#     )
])
train_transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomOrder([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective()
        ]),
        transforms.RandomAffine(30), # 隨機線性轉換
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0)), # 隨機子圖
    ]),
    transforms.RandomChoice([
        transforms.ColorJitter(), # 隨機色溫等
        transforms.RandomGrayscale(),
    ]),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.RandomErasing(0.2),
#     transforms.Normalize(
#         [77.89311144813877 / 255, 102.3587941606983 / 255, 126.59376063616554 / 255],
#         [72.80305392379675 / 255, 75.35438507973123 / 255, 79.31408066842762 / 255]
#     )
])
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
train_set = ConcatDataset([
    ImgDataset(train_x, train_y, train_transform1),
    ImgDataset(train_x, train_y, train_transform2)
])
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16)

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
    
# summary(Classifier().cuda(), (3, IMAGE_SIZE, IMAGE_SIZE))


model = Classifier().cuda()
# model = Classifier().cpu()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 350

# # use apex to optimize
# model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
#         train_pred = model(data[0].cpu())
#         batch_loss = loss(train_pred, data[1].cpu())
#         with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
#             scaled_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        batch_loss.backward() 
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
#             val_pred = model(data[0].cpu())
#             batch_loss = loss(val_pred, data[1].cpu())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %             (epoch + 1, num_epoch, time.time()-epoch_start_time,              train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))


torch.save(model, 'model_confusion.torch')

torch.cuda.empty_cache()
