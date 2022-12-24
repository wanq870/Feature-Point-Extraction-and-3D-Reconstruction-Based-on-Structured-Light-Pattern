import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import cv2
import os
import csv
import numpy as np
import pandas as pd
from natsort import natsorted
import argparse
import glob
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt

train_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.3),
    transforms.RandomErasing(p=0.5, scale=(0.00125,0.02), ratio=(1,1), value=0),
    transforms.Grayscale(1),
])
test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1),
])

class U_Dataset(Dataset):
    def __init__(self, basepath, train=True):
        self.input = natsorted(glob.glob(os.path.join(basepath, 'data', '*.png')))
        self.target = natsorted(glob.glob(os.path.join(basepath, 'label', '*.csv')))
        self.train = train
    def __getitem__(self, index):
        # open image
        img = cv2.imread(self.input[index])
        # img = img[720:1232, 720:1232]

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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

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
            nn.Linear(512*512, 1922),
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

def train(model, dataloader, num_epoch, device, criterion, optimizer, save_path):
    loss_list = []
    best_loss = 9999
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / (i+1)
        loss_list.append(train_loss)
        print(f'Epoch {epoch}: [Train | avg_loss = {train_loss:.5f}]')
        
        valid_loss = valid(model, dataloader[1], device, criterion)

        if valid_loss < best_loss:
            best_loss = valid_loss
            print("Save Model")
            torch.save(model.state_dict(), os.path.join(save_path, 'model.ckpt'), _use_new_zipfile_serialization=False)
    
    figure, axis = plt.subplots()
    axis.plot(range(1, num_epoch + 1), loss_list)
    axis.set_title("Learning curve of Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    plt.savefig(os.path.join(save_path, 'img/learning_curve.png'))

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

def test(model, dataloader, device, data_path, save_path):
    print(f'Predicting...')

    test_data_paths = natsorted(glob.glob(os.path.join(data_path,'test', 'data', '*.png')))
    model.eval()
    idx = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data

            inputs = inputs.float()
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs = np.reshape(np.array(outputs.cpu()), (-1, 2))
            img = cv2.imread('3dcv_dataset/test/data/72.png')
            for idx, corner in enumerate(outputs):
                df = pd.DataFrame(index=range(961), columns=range(10 * 2))
                for k in range(961):
                    df.iloc[k, 0] = outputs[k][0] * 550 / 512 + 740
                    df.iloc[k, 1] = outputs[k][1] * 550 / 512 + 700
                df.to_csv(os.path.join(save_path, 'csv/output.csv'))
                cv2.circle(img, (int(corner[1] * 550 / 512 + 700), int(corner[0] * 550 / 512 + 740)), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_path, 'img/output.png'), img)

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", help="Base path of train/testing data",
                    type=str)
    parser.add_argument("savedir", help="Path to save state dict or predicted csvs",
                    type=Path)
    parser.add_argument("--ckpt", help="Path to ckpt file",
                    type=str)
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
    save_path = args.savedir
    torch.manual_seed(2022)
    device = torch.device('cuda')

	# Init output directory 
    output_csv_dir = args.savedir / 'csv'
    output_img_dir = args.savedir / 'img'
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)

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
    model = UNet()
    model.to(device)

    # Load pretrained model
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # Train
    if not args.do_predict:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.SmoothL1Loss()
        train(model, loaders, args.num_epoch, device, criterion, optimizer, save_path)
    
    # Predict
    else:
        test(model, test_loader, device, data_path, save_path)
if __name__ == '__main__':
    main()
