import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
import torch.nn.functional as func
import torch.utils.data as data
import torchvision.models as models
import pytorch_ssim
import pytorch_colors as colors
import cv2 as cv
import numpy as np
import h5py
import os
import RDDLN_jpeg

# 训练参数
batch_size = 32

num_epoch = 50

learning_rate = 3e-4

PATH = ".\\checkpoint\\RDDLN_jpeg_lv2_photo.pth"


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.ssim = pytorch_ssim.SSIM()

    def forward(self, x1, x2, vgg):
        # YUV转RGB
        x1_rgb=colors.yuv_to_rgb(x1)
        x2_rgb=colors.yuv_to_rgb(x2)

        per_x1 = vgg(x1_rgb).cpu()
        per_x2 = vgg(x2_rgb).cpu()

        x1 = x1.cpu()
        x2 = x2.cpu()
        y1, u1, v1 = torch.chunk(x1, 3, dim=1)
        y2, u2, v2 = torch.chunk(x2, 3, dim=1)

        y_loss = torch.mean(torch.abs(y1 - y2))
        u_loss = torch.mean(torch.abs(u1 - u2))
        v_loss = torch.mean(torch.abs(v1 - v2))
        x_ssim = torch.mean(self.ssim(x1, x2))
        per_loss = torch.mean(torch.abs(per_x1 - per_x2))

        # 各损失权重
        w = [0.1, 0.3, 0.3, 0.3]
        return y_loss * w[0] + u_loss * w[1] + v_loss * w[2] + per_loss * w[3]


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
trainset = DataFromH5File(".\\checkpoint\\train_RDDLN_jpeg_lv2_photo.h5")

# 建立一个dataloader
train_loader = data.DataLoader(dataset=trainset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=True)

# 部署模型
net = RDDLN_jpeg.RDDLN()
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
net.eval()

for epoch in range(num_epoch):
    tr_loss = 0
    for step, data_train in enumerate(train_loader, 0):
        # 获取数据
        lr, hr = data_train
        lr, hr = lr.cuda(), hr.cuda()

        # 清除梯度
        optimizer.zero_grad()

        sr = net(lr.permute(0,3,1,2))
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
        learning_rate = 1e-5
        