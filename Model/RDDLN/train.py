import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
import torch.nn.functional as func
import torch.utils.data as data
import torchvision.models as models
import cv2 as cv
import numpy as np
import h5py
import os
import RDDLN

# 训练参数
batch_size = 28

num_epoch = 50

size_cut_edge = 23

size_image = 58

learning_rate = 3e-4

PATH = ".\\checkpoint\\RDDLN.pth"


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, sr, hr, vgg):
        # 损失比例
        a = 0.66
        l1 = torch.mean(torch.abs(hr.cpu() - sr.cpu()))

        # 3通道
        sr_3c = torch.cat([sr, sr, sr], 1)
        hr_3c = torch.cat([hr, hr, hr], 1)

        per_sr = vgg(sr_3c).cpu()
        per_hr = vgg(hr_3c).cpu()
        per_loss = torch.mean(torch.abs(per_hr - per_sr))

        return l1 * (1 - a) + per_loss * a


class DataFromH5File(data.Dataset):
    def __init__(self, filename):
        # 数据集
        h5File = h5py.File(filename, 'r')
        self.data_lr = h5File.get("lr")
        self.data_hr = h5File.get("hr")

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.data_lr[idx]).float()
        hr = torch.from_numpy(self.data_hr[idx]).float()
        return lr, hr

    def __len__(self):
        assert self.data_hr.shape[0] == self.data_lr.shape[
            0], "Wrong data length"
        return self.data_hr.shape[0]


# 实例化一个dataset
trainset = DataFromH5File(".\\checkpoint\\train_MRDN_2.h5")

# 建立一个dataloader
train_loader = data.DataLoader(dataset=trainset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=True)

# 部署模型
net = RDDLN.RDDLN()
# 部署VGG
vgg = models.vgg16(pretrained=True).features[:11]
vgg = vgg.eval()

# 定义loss和optimizer
criterion = Loss()

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 检查GPU是否可用
if torch.cuda.is_available():
    net = net.cuda()
    vgg = vgg.cuda()
    criterion = criterion.cuda()

net.train()

net.load_state_dict(torch.load(PATH))
net = net.eval()

for epoch in range(num_epoch):
    tr_loss = 0
    for step, data_train in enumerate(train_loader, 0):
        # 获取数据
        lr, hr = data_train
        lr, hr = lr.cuda(), hr.cuda()
        # 清除梯度
        optimizer.zero_grad()
        sr = net(lr)
        loss = criterion(sr, hr.permute(0, 3, 1, 2), vgg)
        loss.backward()
        optimizer.step()
        # print statistics
        tr_loss += loss.item()
        if step % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, step + 1, tr_loss / 100))
            tr_loss = 0.0
            torch.save(net.state_dict(), PATH)
            
    if epoch > 30:
        learning_rate = 2e-5
