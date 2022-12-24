import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


import cv2
import os
import numpy as np
import pandas as pd
from natsort import natsorted
import argparse
import glob
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
import math

train_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.3),
    transforms.RandomErasing(p=0.5, scale=(0.00125,0.02), ratio=(1,1), value=0),
    # transforms.Grayscale(1),
])
test_tfm = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(1),
])

class U_Dataset(Dataset):
    def __init__(self, basepath, train=True):
        self.input = natsorted(glob.glob(os.path.join(basepath, 'data', '*.png')))
        self.target = natsorted(glob.glob(os.path.join(basepath, 'label', '*.csv')))
        self.train = train
    def __getitem__(self, index):
        # open image
        img = cv2.imread(self.input[index])

        # open csv
        df = np.genfromtxt(self.target[index], delimiter=',')
        df = df[1:].astype(float)
        df = df[:,1:3].astype(float)
        
        if np.isnan(df).any():
            print("NaN exist!")
            print(self.target[index])
            exit()
        df = df.flatten()
        
        if self.train:
            img = train_tfm(img)
        else:
            img = test_tfm(img)

        return img, df

    def __len__(self):
        return len(self.input)

## Unet
class DoubleConv(nn.Module):
    # (convolution => [BN] => ReLU) * 2

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    # Downscaling with maxpool then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Upscaling then double conv

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.fc = nn.Sequential(
            nn.Linear(800*800, 1922),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = torch.flatten(x,1)
        logits = self.fc(x)
        return logits

"""
## Resnet
class Bottleneck(nn.Module):
    # __init__
    #     in_channel: 残差块输入通道数
    #     out_channel: 残差块输出通道数
    #     stride: 卷积步长
    #     downsample: 在_make_layer函数中赋值, 用于控制shortcut图片下采样 H/2 W/2
    # expansion = 4   # 残差块第3个卷积层的通道膨胀倍率
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  # H/2，W/2。C不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x    # 将原始输入暂存为shortcut的输出
        if self.downsample is not None:
            identity = self.downsample(x)   # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity     # 残差连接
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    # __init__
    #     block: 堆叠的基本模块
    #     block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
    #     num_classes: 全连接之后的分类特征维度
    # _make_layer
    #     block: 堆叠的基本模块
    #     channel: 每个stage中堆叠模块的第一个卷积的卷积核个数, 对resnet50分别是:64,128,256,512
    #     block_num: 当期stage堆叠block个数
    #     stride: 默认卷积步长
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64    # conv1的输出维度
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   # H,W不变。downsample控制的shortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)
        
        for m in self.modules():    # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None   # 用于控制shorcut路的
        if stride != 1 or self.in_channel != channel*block.expansion:   # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4，因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                nn.BatchNorm2d(num_features=channel*block.expansion))
            
        layers = []  # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
        self.in_channel = channel*block.expansion   # 在下一次调用_make_layer函数的时候，self.in_channel已经x4
        
        for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
            layers.append(block(in_channel=self.in_channel, out_channel=channel))
            
        return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
"""
## Resnest
from resnest.torch import resnest50



train_loss_list = []
def train(model, dataloader, num_epoch, device, criterion, optimizer, save_path):
    best_loss = 9999
    # cnt = 0
    model.train()
    print(f'Training...')
    
    for epoch in trange(num_epoch):
        running_loss = 0
        for  i, data in enumerate(dataloader[0]):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / (i+1)
        train_loss_list.append(train_loss)
        if (epoch + 1) % 50 == 0:
            print('Saving loss for plotting learning curve')
            print(f'epoch : {len(train_loss_list)}')
            np.save(os.path.join(save_path, 'train_loss.npy'), np.array(train_loss_list, dtype=np.float32), allow_pickle=True)
            print('Finish saving')


        print(f'Epoch {epoch}: [Train | avg_loss = {train_loss:.5f}]')
        
        valid_loss = valid(model, dataloader[1], device, criterion)

        if valid_loss < best_loss:
            best_loss = valid_loss
            # cnt = 0
            print("Save Model")
            torch.save(model.state_dict(), os.path.join(save_path, 'model.ckpt'), _use_new_zipfile_serialization=False)
        # else:
        #     cnt += 1
        #     if(cnt >= 100):
        #         print('no improvement, early stop')
        #         break

def valid(model, dataloader, device, criterion):
    model.eval()

    with torch.no_grad():
        running_loss = 0
        global best_loss
        for i, data in enumerate(dataloader):
            inputs, labels = data

            inputs = inputs.float()
            inputs = inputs.to(device)

            labels = labels.float()
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
        valid_loss = running_loss / (i+1)
        print(f"         [Valid | avg_loss = {valid_loss:.5f}]")
    return valid_loss

def test(model, dataloader, device, criterion, save_csv_dir, save_img_dir, save_plot_dir):
    print(f'Predicting...')

    model.eval()
    cnt = 1
    with torch.no_grad():
        running_loss = 0
        for data in dataloader:
            inputs, labels = data

            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            outputs_arr = np.array(outputs.cpu()).reshape(-1, 2)
            # save predicted labels and plotted img and points
            img = output_img(inputs)
            df = output_label(outputs_arr)
            df.to_csv(os.path.join(save_csv_dir, str(cnt) + '.csv'))
            
            out_img = np.zeros((512, 512))
            
            for corner in outputs_arr:
                out_img[int(corner[0]), int(corner[1])] = 255
                cv2.circle(img, (int(corner[1]), int(corner[0])), 2, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_img_dir, str(cnt) + '.png'), img)
            cv2.imwrite(os.path.join(save_plot_dir, str(cnt) + '.png'), out_img)
            
            # compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            print(f'loss of image {cnt}: {loss.item()}')
            cnt += 1
            
def output_img(input):
    img = input.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    plt.imshow(img)
    plt.show()
    return img

def output_label(input):
    df = pd.DataFrame(index=range(961), columns=range(10 * 2))
    for i, corner in enumerate(input):
        df.iloc[i, 0] = corner[0]
        df.iloc[i, 1] = corner[1]
    return df

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", help="Base path of train/testing data", type=str)
    parser.add_argument("model", choices=['Unet', 'Resnest'], help='which model architecture to use', type=str)
    # parser.add_argument("savedir", choose=['Unet_output', 'Resnet_output', 'Transformer_output'], help="Path to save state dict or predicted csvs",
    #                 type=Path)
    parser.add_argument("--ckpt", help="Path to ckpt file, eg. Unet/model.ckpt", type=str)
    parser.add_argument("--do_predict", help="Do predict",
                    action='store_true')

    # Hyperparameters
    parser.add_argument("--num_epoch", help="num_epoch",
                    type=int, default=10)
    parser.add_argument("--batch_size", help="batch_size",
                    type=int, default=1)
    parser.add_argument("--lr", help="lr",
                    type=float, default=1e-4)
    parser.add_argument("--weight_decay", help="weight_decay",
                    type=float, default=1e-4)
    
    args = parser.parse_args()

    data_path = args.basedir
    torch.manual_seed(2022)
    device = torch.device('cuda')

	# Init output directory
    """
    output file structure:
    [model_name]/
        -label/
        -img/
        -input/
        -model.ckpt
    """
    output_dir = args.model
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_csv_dir = os.path.join(output_dir, 'csv')
    if not os.path.exists(output_csv_dir):
        os.mkdir(output_csv_dir)
    output_img_dir = os.path.join(output_dir, 'img')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
    output_plot_dir = os.path.join(output_dir, 'input')
    if not os.path.exists(output_plot_dir):
        os.mkdir(output_plot_dir)

    ### Create dataset
    if not args.do_predict:
        print("Creating train & valid dataset...")
        train_set = U_Dataset(os.path.join(data_path, 'train'), train=True)
        valid_set = U_Dataset(os.path.join(data_path, 'valid'), train=False)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False
        )
        loaders = [train_loader, valid_loader]
        print('Done')
    else:
        print("Creating test dataset...")
        test_set = U_Dataset(os.path.join(data_path, 'test'), train=False)
        
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False
        )
        print('Done')
    
    # Create model
    models = {'Unet': UNet(), 'Resnest': resnest50(pretrained=False, dilated=True, num_classes=1922)} # define which model to use
    model = models[args.model]
    model.to(device)

    # Load pretrained model
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # Train
    if not args.do_predict:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.SmoothL1Loss()
        train(model, loaders, args.num_epoch, device, criterion, optimizer, output_dir)
        
        # saving train/valid loss for plotting learning curve
        print('Saving loss for plotting learning curve')
        print(f'epoch : {len(train_loss_list)}')
        np.save(os.path.join(output_dir, 'train_loss.npy'), np.array(train_loss_list, dtype=np.float32), allow_pickle=True)
        print('Finish saving')
    # Predict
    else:
        criterion = nn.SmoothL1Loss()
        test(model, test_loader, device, criterion, output_csv_dir, output_img_dir, output_plot_dir)
        

    
if __name__ == '__main__':
    main()
