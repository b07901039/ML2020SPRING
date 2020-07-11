# Basis Iterative Method
import os
import sys
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        # 存下攻擊後(或原本就判斷錯誤不用攻擊)的所有圖片 之後算succese rate和inf L-norm
        gen_imgs = []
        # 存下原本的所有圖片 之後算succese rate和inf L-norm
        raw_imgs = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;

            raw_img = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            raw_img = raw_img.squeeze().detach().cpu().numpy() 
            raw_imgs.append(raw_img)            
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                gen_img = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                gen_img = gen_img.squeeze().detach().cpu().numpy() 
                gen_imgs.append(gen_img)
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            gen_img = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            gen_img = gen_img.squeeze().detach().cpu().numpy() 
            gen_imgs.append(gen_img)
            
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                  adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                  data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  data_raw = data_raw.squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )        
        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t Success rate = {}\n"
        .format(epsilon, fail, len(self.loader), final_acc, 1 - final_acc))
        return adv_examples, final_acc, gen_imgs, raw_imgs

    def iter_attack(self, epsilon=0.1, epoch=5, boundary=9.5):
        # 每次根據gradient sign的方向更新圖片 再根據boundary fix圖片
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        # 存下攻擊後(或原本就判斷錯誤不用攻擊)的所有圖片 之後算succese rate和inf L-norm
        gen_imgs = []
        # 存下原本的所有圖片 之後算succese rate和inf L-norm
        raw_imgs = []
        wrong, fail, success = 0, 0, 0
        count = 0 # count data processed
        epoches = [] # 紀錄每張圖片更新到第幾個epoch時成功攻擊
        for (data, target) in self.loader:
            print("\rdata: [{} / 200], fail: {}".format(count, fail), end='')
            count += 1
            data, target = data.to(device), target.to(device)
            data_raw = data;

            raw_img = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            raw_img = raw_img.squeeze().detach().cpu().numpy() 
            raw_imgs.append(raw_img)            
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            # 存下前三高的機率和對應的類別 (report 第3題)
            output_copy = output.view(-1, 1).squeeze().detach().cpu().numpy()
            output_probs = np.exp(output_copy) / np.sum(np.exp(output_copy))
            idxs = np.argsort(output_probs)[::-1][:3]
            ori_prob = zip(idxs, output_probs[idxs])

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                gen_img = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                gen_img = gen_img.squeeze().detach().cpu().numpy() 
                gen_imgs.append(gen_img)
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            for i in range(epoch):
              loss = F.nll_loss(output, target)
              self.model.zero_grad()
              loss.backward()
              
              # .data 不會加到computational graph裡，避免"backward through the graph a second time"
              # data = data + epilson * loss.data
              data_grad = data.grad.data
              data = self.fgsm_attack(data, epsilon, data_grad)
              data = self.fix(data, data_raw, boundary)
              output = self.model(data)
              pred = output.max(1, keepdim=True)[1]
              if pred.item() != target.item():
                  # 攻擊成功
                  break
            epoches.append(i+1)
            perturbed_data = data
            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            gen_img = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            gen_img = gen_img.squeeze().detach().cpu().numpy() 
            gen_imgs.append(gen_img)
            
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                  output_copy = output.view(-1, 1).squeeze().detach().cpu().numpy()
                  output_probs = np.exp(output_copy) / np.sum(np.exp(output_copy))
                  idxs = np.argsort(output_probs)[::-1][:3]
                  adv_prob = zip(idxs, output_probs[idxs])
                  # ori_prob = None
                  # adv_prob = None
                  adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                  data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  data_raw = data_raw.squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex, ori_prob, adv_prob) )        
        final_acc = (fail / (wrong + success + fail))
        
        print("\nEpsilon: {}\tTest Accuracy = {} / {} = {}\t Success rate = {}\n"
        .format(epsilon, fail, len(self.loader), final_acc, 1 - final_acc))
        
        print("epoch count:")
        unique, counts = np.unique(epoches, return_counts=True)
        print(dict(zip(unique, counts)))
        return adv_examples, final_acc, gen_imgs, raw_imgs
    
    def fix(self, perturbed_data, raw_data, boundary):
      perturbed_data = torch.where(perturbed_data > raw_data + boundary, raw_data + boundary, perturbed_data)
      perturbed_data = torch.where(perturbed_data < raw_data + boundary, raw_data - boundary, perturbed_data)
      return perturbed_data.detach().requires_grad_()

    def test(self):
      wrong, right = 0, 0
      cnt = 0
      ex = None
      for (data, target) in self.loader:
        data, target = data.to(device), target.to(device)
        output = self.model(data)
        if cnt == 2:
          output_copy = output.view(-1, 1).squeeze().detach().cpu().numpy()
          output_probs = np.exp(output_copy) / np.sum(np.exp(output_copy))
          idxs = np.argsort(output_probs)[::-1][:3]
          prob = zip(idxs, output_probs[idxs])
          img = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
          img = img.squeeze().detach().cpu().numpy()
          ex = (img, prob)
        cnt += 1
        pred = output.max(1, keepdim=True)[1]
        if pred.item() != target.item():
          wrong += 1
        else:
          right += 1
      acc = right / (right + wrong)
      print("Test accuracy: {}, Success rate: {}".format(acc, 1-acc))
      return ex

      

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 讀入圖片相對應的 label
    df = pd.read_csv(os.path.join(input_dir, "labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(input_dir, "categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker(os.path.join(input_dir, 'images'), df)
    # 要嘗試的 epsilon
    epsilons = [0.1]

    accuracies, examples, gen_imgs_list, raw_imgs_list = [], [], [], []


    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        # ex, acc, gen_imgs, raw_imgs = attacker.attack(eps)
        ex, acc, gen_imgs, raw_imgs = attacker.iter_attack(epsilon=eps, epoch=5, boundary=20/255)
        # ex = attacker.test()
        accuracies.append(acc)
        examples.append(ex)
        gen_imgs_list.append(gen_imgs)
        raw_imgs_list.append(raw_imgs)

    # convert to 0..255
    new_genImgs_list = []
    new_rawImgs_list = []

    for gen_imgs in gen_imgs_list:
      new_imgs = []
      for i in range(200):
        # clip掉負數的部分 以免轉成uint8時變成255
        new_imgs.append(np.clip((gen_imgs[i]*255), 0, 255).astype(np.uint8))
      new_genImgs_list.append(new_imgs)

    for raw_imgs in raw_imgs_list:
      new_imgs = []
      for i in range(200):
        new_imgs.append(np.clip((raw_imgs[i]*255),0,255).astype(np.uint8))
      new_rawImgs_list.append(new_imgs)


    # 將攻擊產生的200張圖片輸出
    fnames=[]
    for i in range(200):
      fnames.append("{:03d}".format(i))
    cnt = 0
    for img in new_genImgs_list[0]:
      img = np.transpose(img, (1, 2, 0))
      img = Image.fromarray(img, 'RGB')
      img.save(os.path.join(output_dir,fnames[cnt] + '.png'))
      cnt += 1