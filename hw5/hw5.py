import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace

import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

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
    # 指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

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

class CAM_Classifier(nn.Module):
    def __init__(self, original_model):
        super(CAM_Classifier, self).__init__()
        self.cnn = nn.Sequential(
                    # stop at last ReLU
                    *list(original_model.cnn.children())[:-1]
                )
        self.max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.gradients = None
        self.fc = original_model.fc
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
   
    def forward(self, x):
        x = self.cnn(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.cnn(x)

def compute_saliency_maps(x, y, model):
  model.eval()
  x = x.cuda()
  
  # 告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
  x.requires_grad_()

  y_pred = model(x)
  loss_func = torch.nn.CrossEntropyLoss()
  loss = loss_func(y_pred, y.cuda())
  loss.backward()

  saliencies = x.grad.abs().detach().cpu() # saliencies: (batches, channels, height, weight)
  
  saliencies = torch.stack([normalize(item) for item in saliencies]) # normalize
  
  return saliencies

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

def testing(model, test_loader):

    model.eval()
    prediction = []
    standardized_raw_y = []
    raw_y = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            
            test_standardize = standardize(test_pred.cpu().data.numpy())
            test_value = np.max(test_standardize, axis=1)
            
            for i in range(len(test_label)):
                prediction.append(test_label[i])
                standardized_raw_y.append(test_value[i])
                raw_y.append(test_pred[i])
    return prediction, standardized_raw_y, raw_y

def standardize(x):
    std = np.std(x)
    if std != 0:
        return (x-np.mean(x))/std
    else:
        return x
def test(train_x, train_y, model, class_i):
    batch_size = 128
    num_of_class = [994, 709, 429, 1500, 986, 848, 1325, 440, 280, 855, 1500]
    if class_i == 0:
        img_indices = [i for i in range(994)]
    elif class_i == 10:
        img_indices = [i for i in range(994, 994+709)]
    else:
        start = sum(num_of_class[:class_i+1])
        img_indices = [i for i in range(start, start+num_of_class[class_i])]
    
    test_x, test_y = train_x.take(img_indices, axis = 0), train_y.take(img_indices, axis = 0)
    test_set = ImgDataset(test_x, test_y, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    pred_y, _, _ = testing(model, test_loader)

    correct = 0
    # check if train_y is correct
    for i in range(len(pred_y)):
        if pred_y[i] ==  test_y[i]:
            correct += 1
    print("class {}, accuracy: {}".format(class_i, correct/len(pred_y)))
    acc_imgs = [] # index of imgs with actual value high to low
    for i in np.argsort(actual_y)[::-1]:
        if pred_y[i] == class_i:
            acc_imgs.append(i)
    # show the most accurate image
    print("class {}, highest value: {}".format(class_i, actual_y[acc_imgs[0]]))
    plt.imshow(test_x[acc_imgs[0]][:,:,[2,1,0]])
    plt.savefig("./plot/acc2_{}.png".format(class_i))

    return acc_imgs[:10], correct/len(pred_y)

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output
  
  hook_handle = model.cnn[cnnid].register_forward_hook(hook)

  model(x.cuda())

  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  
  x = x.cuda()
  x.requires_grad_()
  optimizer = Adam([x], lr=lr)
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)
    
    objective = -layer_activations[:, filterid, :, :].sum()
    
    objective.backward()
    optimizer.step()
  filter_visualization = x.detach().cpu().squeeze()[0]

  hook_handle.remove()

  return filter_activations, filter_visualization

def predict(_input):
    # input: numpy array, (batches, height, width, channels)
    
    model.eval()                                                                                                                                                             
    _input = torch.FloatTensor(_input).permute(0, 3, 1, 2)                                                                                                            
    # (batches, channels, height, width)

    output = model(_input.cuda())                                                                                                                                             
    return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                             
def segmentation(_input):
    
    return slic(_input, n_segments=100, compactness=1, sigma=1)                                                                                                              


if __name__ == "__main__":

    model_f = "./model.pkl"
    workspace_dir = sys.argv[1]
    output_dir = sys.argv[2]


    print("loading model...")
    model = torch.load(model_f)
    
    # read file
    print("loading file...")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True) # 9866
    train_set = ImgDataset(train_x, train_y, test_transform)
    """
    # test and find high accuracy imgs for each class
    print("testing...")
    acc_dict = {}
    acc_list = []
    for i in [0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print("class {}...".format(i))
        index_list, acc = test(train_x, train_y, model, i)
        acc_dict[i] = index_list
        acc_list.append(acc)

    for k, v in acc_dict.items():
        print("class {}".format(k))
        print(v)

    print(acc_list)
    
    exit()
    """


    # Saliency map
    # class: [indices for imgs that has highest standardize predicted y of the class]
    print("saliency map...")
    img_indices_dict = {0: [345, 0, 181], 1: [1777, 2067, 2130], 2: [2152, 2362, 2236],
                        3: [4208, 4544, 3744], 4: [5414, 4778, 4732], 5: [6010, 5947, 6113],
                        6: [7205, 7168, 7202], 7: [7411, 7353, 7366], 8: [7762, 7583, 7623],
                        9: [8841, 8855, 8976], 10: [1469, 1302, 1618]}
    for i in range(11):
        # print("class {}...".format(i))
        img_indices = img_indices_dict[i]
        images, labels = train_set.getbatch(img_indices)
        original_imgs = train_x.take(img_indices, axis = 0)

        # print("computing saliency_map...")
        saliencies = compute_saliency_maps(images, labels, model)

        # plot
        # print("plotting...")
        fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
        for row, target in enumerate([images, saliencies]):
          for column, img in enumerate(target):
            if row == 0: # show original images
                original_img = original_imgs[column][:,:,[2,1,0]] # convert to RGB
                axs[row][column].imshow(original_img)
            else: # show saliency map
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
        fig.suptitle('class {}'.format(i), fontsize=16)
        plt.savefig(os.path.join(output_dir, 'saliency_map_class{}.png'.format(i)))
        plt.close()

    # Filter explaination
    print("Filter explanation...")
    img_indices = [83, 4218, 4707, 8598]
    images, labels = train_set.getbatch(img_indices)
    original_imgs = train_x.take(img_indices, axis = 0)

    # 指定filterid, cnnid的所有filter activations和filter visualization
    # 畫出 filter visualization
    filterid = 29
    cnnid = 4
    # print("cnn id: {}, filter id: {}".format(cnnid, filterid))
    
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=cnnid, filterid=filterid, iteration=100, lr=0.1)
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(output_dir, "fv_{}_{}.png".format(cnnid, filterid)))
    plt.clf()
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i in range(len(original_imgs)):
        original_img = original_imgs[i][:,:,[2,1,0]]
        axs[0][i].imshow(original_img)
    for i, img in enumerate(filter_activations):
      axs[1][i].imshow(normalize(img))
    plt.savefig(os.path.join(output_dir, "fa_{}_{}.png".format(cnnid, filterid)))
    plt.close("all")
    plt.clf()

    filterid = 48
    cnnid = 4
    # print("cnn id: {}, filter id: {}".format(cnnid, filterid))
    
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=cnnid, filterid=filterid, iteration=100, lr=0.1)
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(output_dir, "fv_{}_{}.png".format(cnnid, filterid)))
    plt.clf()
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i in range(len(original_imgs)):
        original_img = original_imgs[i][:,:,[2,1,0]]
        axs[0][i].imshow(original_img)
    for i, img in enumerate(filter_activations):
      axs[1][i].imshow(normalize(img))
    plt.savefig(os.path.join(output_dir, "fa_{}_{}.png".format(cnnid, filterid)))
    plt.close("all")
    plt.clf()

    
    # 每層 Conv2d 的前64個filter (input 4 張圖片，只留第一張的filter activation)
    # num_filter = {0: 64, 4: 128, 8: 256, 12: 512, 16: 512}
    # print("computing filter...")
    for cnnid in [0, 4, 8, 12, 16]:
        # print("cnn id: {}".format(cnnid))
        filter_activations_list = []
        filter_visualization_list = []
        for filterid in range(64):
            # print("filter id : {}".format(filterid), end='\r')
            filter_activations, filter_visualization = filter_explaination(images, model, cnnid=cnnid, filterid=filterid, iteration=100, lr=0.1)
            filter_activations_list.append(filter_activations[0,:,:])
            filter_visualization_list.append(filter_visualization)
            
        # 畫出 filter visualization
        # print("\nplotting...")
        fig, axis = plt.subplots(8, 8, figsize=(8,8))
        count = 0
        for row in range(8):
            for col in range(8):
                axis[row][col].imshow(normalize(filter_visualization_list[count].permute(1,2,0)))
                axis[row, col].axis('off')
                count += 1
        # plt.title("first 64 filters of {} Conv2d".format(cnnid))
        plt.suptitle("first 64 filters of {} Conv2d\n".format(cnnid))
        plt.savefig(os.path.join(output_dir, "fv_{}_all.png".format(cnnid)))
        plt.close("all")
        plt.clf()
        # 畫出 filter activations
        fig, axis = plt.subplots(8, 8, figsize=(8,8))
        count = 0
        for row in range(8):
            for col in range(8):
                axis[row][col].imshow(normalize(filter_activations_list[count]))
                axis[row, col].axis("off")
                count += 1
        # plt.title("output of first 64 filters of {} Conv2d\n".format(cnnid))
        plt.suptitle("output of first 64 filters of {} Conv2d\n".format(cnnid))
        plt.savefig(os.path.join(output_dir, "fa_{}_all.png".format(cnnid)))
        plt.close("all")
        plt.clf()
    
    
    # lime
    print("lime...")                                                                                                                                                              
    print("loading evaluataion file...")
    
    batch_size = 128
    val_x, val_y = readfile(os.path.join(workspace_dir, "evaluation"), True)
    val_set = ImgDataset(val_x, val_y, test_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    pred_y, actual_y, _ = testing(model, val_loader)

    img_indices_dict = {0: [234, 289], 10: [382, 437], 1: [685, 732, 671, 684], 
                        2: [940, 1096], 3: [1359, 1406, 1265], 4: [1620, 1631],
                        5: [2013, 2225], 6: [2447, 2423], 7: [2540, 2468],
                        8: [2632, 2610, 2769], 9: [2866, 2869]}
    mis_pred = {0: [4], 1: [0, 2, 5], 2: [0], 3: [0, 2], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0, 2], 9: [0], 10: [2]} # class: mis pred to classes
    class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

    
    # 和正確的label比較
    for classid in range(11):
        # print("class id: {}".format(classid))
        img_indices = img_indices_dict[classid]
        
        images, labels = val_set.getbatch(img_indices)
        original_imgs = val_x.take(img_indices, axis = 0)
        mis_labels = np.array(pred_y).take(img_indices, axis = 0)
        
        fig, axs = plt.subplots(3, len(img_indices), figsize=(15, 8))
        fig.suptitle('class {}'.format(class_names[classid], fontsize=16))
        axs[0][0].set_title(class_names[classid])
        for i in range(1, len(img_indices)):
            axs[0][i].set_title(class_names[mis_pred[classid][i-1]])
        
        np.random.seed(16)                                                                                                                                                       
        # 讓實驗 reproducible
        for col in range(len(img_indices)):
            original_img = original_imgs[col][:,:,[2,1,0]] # convert to RGB
            axs[0][col].imshow(original_img)
            axs[0, col].axis("off")

        # compare with val_y
        for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):                                                                                                                                              
            x = image.astype(np.double) # (128, 128, 3)
            # lime 這個套件要吃 numpy array

            explainer = lime_image.LimeImageExplainer()                                                                                                                              
            explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
            # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
            # classifier_fn 定義圖片如何經過 model 得到 prediction
            # segmentation_fn 定義如何把圖片做 segmentation
            # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

            lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                        label=label.item(),                                                                                                                           
                                        positive_only=False,                                                                                                                         
                                        hide_rest=False,                                                                                                                             
                                        num_features=11,                                                                                                                              
                                        min_weight=0.05                                                                                                                              
                                    )
            # 把 explainer 解釋的結果轉成圖片
            # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
            
            axs[1][idx].imshow(lime_img)
            axs[1, idx].axis("off")
        # compare with pred_y
        for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), mis_labels)):                                                                                                                                              
            x = image.astype(np.double) # (128, 128, 3)
            # lime 這個套件要吃 numpy array

            explainer = lime_image.LimeImageExplainer()                                                                                                                              
            explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)
            # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
            # classifier_fn 定義圖片如何經過 model 得到 prediction
            # segmentation_fn 定義如何把圖片做 segmentation
            # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

            lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                        label=label.item(),                                                                                                                           
                                        positive_only=False,                                                                                                                         
                                        hide_rest=False,                                                                                                                             
                                        num_features=11,                                                                                                                              
                                        min_weight=0.05                                                                                                                              
                                    )
            # 把 explainer 解釋的結果轉成圖片
            # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
            
            axs[2][idx].imshow(lime_img)
            axs[2, idx].axis("off")


        plt.savefig(os.path.join(output_dir, "lime_class{}".format(classid)))
        plt.close("all")


    

    
    # training confusion matrix
    # predict training set
    print("confusion matrix...")
    batch_size = 128
    test_set = ImgDataset(train_x, train_y, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    pred_y, _, _ = testing(model, test_loader)

    cm = confusion_matrix(np.array(train_y), np.array(pred_y))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
    class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
    df_cm = pd.DataFrame(cm, index = class_names, columns = class_names )
    plt.figure(figsize=(14,14))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout() # give x, y labels space
    plt.title("Training confusion matrix")
    plt.savefig(os.path.join(output_dir, "train_cm.png"))
    plt.close("all")

    
    # validation confusion matrix
    print("loading evalutaion file...")
    pred_y, _, _ = testing(model, val_loader)
    cm = confusion_matrix(np.array(val_y), np.array(pred_y))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
    class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
    df_cm = pd.DataFrame(cm, index = class_names, columns = class_names )
    plt.figure(figsize=(14,14))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout() # give x, y labels space
    plt.title("Validation confusion matrix")
    plt.savefig(os.path.join(output_dir, "val_cm.png"))
    plt.close("all")

    """
    # 找出validation set易誤判成指定類別的index
    batch_size=128
    val_x, val_y = readfile(os.path.join(workspace_dir, "evaluation"), True) 
    val_set = ImgDataset(val_x, val_y, test_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    pred_y, actual_y = testing(model, val_loader)

    mis_pred = {0: [4], 1: [0, 2, 5], 2: [0], 3: [0, 2], 4: [0], 5: [0],
                6: [0], 7: [0], 8: [0, 2], 9: [0], 10: [2]}
    mis_index = {}
    acc_index = {}
    for i in range(len(val_y)):
        if val_y[i] == pred_y[i]:
            if val_y[i] not in acc_index.keys():
                acc_index[val_y[i]] = i
            elif actual_y[i] > actual_y[acc_index[val_y[i]]]:
                    acc_index[val_y[i]] = i
        elif pred_y[i] in mis_pred[val_y[i]]:
            if val_y[i] not in mis_index.keys():
                mis_index[val_y[i]] = [None for i in range(len(mis_pred[val_y[i]]))]
                mis_index[val_y[i]][mis_pred[val_y[i]].index(pred_y[i])] = i
            elif mis_index[val_y[i]][mis_pred[val_y[i]].index(pred_y[i])] == None:
                mis_index[val_y[i]][mis_pred[val_y[i]].index(pred_y[i])] = i
            elif actual_y[i] > actual_y[mis_index[val_y[i]][mis_pred[val_y[i]].index(pred_y[i])]]:
                mis_index[val_y[i]][mis_pred[val_y[i]].index(pred_y[i])] = i
    print(mis_index)
    print(acc_index)
    """
    # grad-cam
    print("grad cam...")

    img_indices = [345, 1777, 2152, 4208, 5414, 6010, 7202, 7411, 7762, 8841, 1469]
    cam_cnn = CAM_Classifier(model).cuda() # cam_cnn has same architecture and weights as model
    cam_cnn.eval()
    for idx in range(len(img_indices)):
        # print("class {}".format(idx))
        images, labels = train_set.getbatch([img_indices[idx]])
        original_imgs = train_x.take([img_indices[idx]], axis = 0)
        pred_y = cam_cnn(images.cuda())
        pred_labels = np.argmax(model(images.cuda()).cpu().data.numpy(), axis=1)

        pred_y[:, pred_labels[0]].backward()
        gradients = cam_cnn.get_activations_gradient() # [1, 512, 8, 8]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) #[512]
        activations = cam_cnn.get_activations(images.cuda()).detach() # torch.Size([1, 512, 8, 8])
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap) # torch.Size([8, 8])

        # draw the heatmap
        # plt.matshow(heatmap.squeeze())

        # project on original images
        heatmap = cv2.resize(np.float32(heatmap), (128, 128))
        # print(heatmap.shape)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.8 + original_imgs[0][:,:,[2,1,0]] 

        cv2.imwrite(os.path.join(output_dir, "cam_class{}.png".format(idx)), superimposed_img)
        plt.close("all")

    
    # output original imgs for grad-cam
    img_indices = [345, 1777, 2152, 4208, 5414, 6010, 7202, 7411, 7762, 8841, 1469]
    original_imgs = train_x.take(img_indices, axis = 0)
    for i in range(11):
        plt.imshow(original_imgs[i][:,:,[2, 1, 0]])
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "original_class{}".format(i)))
        plt.close("all")
        plt.clf()
