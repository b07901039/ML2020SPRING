"""
用助教pretrained的resnet18當teacher net
Depthwise Separable Convolution + Knowledge Distillation
模型架構參考: https://github.com/marvis/pytorch-mobilenet/blob/master/main.py
transform:  randomcrop, randomhorizontalflip, randomrotation, color jitter, random perspective
用AdamW(lr=1e-3) train 120 epoch, 再用AdamW(lr=1e-4) train 90 epoch, 再用 SGD(lr=1e-3) train 90 epcoh (因為訓練中途有被中斷過)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

model_f = "./big_model.bin"
Tmodel_f = "./teacher_resnet18.bin"
workspace_dir = sys.argv[1]

class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        模型架構參考: https://github.com/marvis/pytorch-mobilenet/blob/master/main.py        
        '''
        super(StudentNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.cnn = nn.Sequential(

            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),

            nn.AdaptiveAvgPool2d((1, 1)),
            
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(256, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
        
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        if folderName == os.path.join(workspace_dir, 'training&validation'):
            for img_path in glob(os.path.join(workspace_dir, 'training') + '/*.jpg'):
                try:
                    # Get classIdx by parsing image path
                    class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
                except:
                    # if inference mode (there's no answer), class_idx default 0
                    class_idx = 0

                image = Image.open(img_path)
                # Get File Descriptor
                image_fp = image.fp
                image.load()
                # Close File Descriptor (or it'll reach OPEN_MAX)
                image_fp.close()

                self.data.append(image)
                self.label.append(class_idx)

            for img_path in glob(os.path.join(workspace_dir, 'validation') + '/*.jpg'):
                try:
                    # Get classIdx by parsing image path
                    class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
                except:
                    # if inference mode (there's no answer), class_idx default 0
                    class_idx = 0

                image = Image.open(img_path)
                # Get File Descriptor
                image_fp = image.fp
                image.load()
                # Close File Descriptor (or it'll reach OPEN_MAX)
                image_fp.close()

                self.data.append(image)
                self.label.append(class_idx)


        else: # training, validation, test
            for img_path in glob(folderName + '/*.jpg'):
                try:
                    # Get classIdx by parsing image path
                    class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
                except:
                    # if inference mode (there's no answer), class_idx default 0
                    class_idx = 0

                image = Image.open(img_path)
                # Get File Descriptor
                image_fp = image.fp
                image.load()
                # Close File Descriptor (or it'll reach OPEN_MAX)
                image_fp.close()

                self.data.append(image)
                self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)), # 參考同學分享
    transforms.RandomPerspective(distortion_scale=0.2,p=0.5), # 參考同學分享
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation', 'training&validation']

    dataset = MyDataset(
        os.path.join(workspace_dir, mode),
        transform=trainTransform if mode == 'training' or mode == 'training&validation' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training' or mode == 'training&validation'))

    return dataloader

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num




if __name__ == "__main__":
    
    batch_size = 32
    print("loading model...")
    teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
    student_net = StudentNet(base=16).cuda()
    teacher_net.load_state_dict(torch.load(Tmodel_f))
    optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)

    # train with training + validation set
    print("loading data...")
    train_val_dataloader = get_dataloader('training&validation', batch_size=batch_size)
    
    print("training...")
    # TeacherNet永遠都是Eval mode.
    teacher_net.eval()
    for epoch in range(300): #300
        
        if epoch == 120: # 120
            optimizer = optim.AdamW(student_net.parameters(), lr=1e-4)

        if epoch == 210: # 210
          optimizer = optim.SGD(student_net.parameters(), lr=1e-3, momentum=0.9)
        
        student_net.train()
        train_loss, train_acc = run_epoch(train_val_dataloader, update=True)
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} '.format(epoch, train_loss, train_acc))
        
    
    torch.save(student_net.state_dict(), model_f)

