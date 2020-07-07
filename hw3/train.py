"""
read training and validation set, 
train model,
save model at model_f
"""
import sys
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
# import time
# import matplotlib.pyplot as plt

model_f = "./model.pkl"
workspace_dir = sys.argv[1]

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomResizedCrop(size = 128, scale=(0.1, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to (-1, 1)
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to (-1, 1)
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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]


            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)       # [512, 4, 4]
        )
        self.fc = nn.Sequential(           
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

if __name__ == "__main__":
  # read file
  train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
  val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)

  # transform
  batch_size = 128 
  # train_set = ImgDataset(train_x, train_y, train_transform)
  # val_set = ImgDataset(val_x, val_y, test_transform)
  # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

  # combine train set and validation set
  train_val_x = np.concatenate((train_x, val_x), axis=0)
  train_val_y = np.concatenate((train_y, val_y), axis=0)
  train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
  train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)
  
  # train
  model_best = Classifier().cuda()
  loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
  optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
  num_epoch = 120

  for epoch in range(num_epoch):
      epoch_start_time = time.time()
      train_acc = 0.0
      train_loss = 0.0

      model_best.train()
      for i, data in enumerate(train_val_loader):
          optimizer.zero_grad()
          train_pred = model_best(data[0].cuda())
          batch_loss = loss(train_pred, data[1].cuda())
          batch_loss.backward()
          optimizer.step()

          train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
          train_loss += batch_loss.item()

          #將結果 print 出來
      # print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      #   (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      #   train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))
  
  # save model
  torch.save(model_best, model_f)

