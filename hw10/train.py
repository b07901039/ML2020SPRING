import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
import numpy as np

from model import fcn_autoencoder
from cnn2 import conv2_autoencoder


if __name__ == "__main__":

    data_pth = sys.argv[1]
    model_pth = sys.argv[2]

    if "baseline" in model_pth:
        model_type = "fcn"
    elif "best" in model_pth:
        model_type = "cnn"
    else:
        print("unknown model type")
 
    num_epochs = 1300
    batch_size = 128
    learning_rate = 1e-3

    x = np.load(data_pth, allow_pickle=True)

    if model_type == 'fcn':
        x = x.reshape(len(x), -1)
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv2_autoencoder()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    
    best_loss = np.inf
    model.train()
    for epoch in range(num_epochs):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)

            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            # ===================save====================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, model_pth)
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        


