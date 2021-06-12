import torch
import torch.nn as nn
import torch.nn.functional as func


class RDDLN(nn.Module):
    def __init__(self):
        super(RDDLN, self).__init__()
        """
        ===============网络参数===============
        """
        # 分辨率
        self.size_lr = 64
        self.size_hr = 124

        # 输入图片通道数
        self.num_input_channel = 1

        # 预处理卷积
        self.size_kernel_pre = 3
        self.num_filters_pre = 64

        # 残差密集块feature map数量
        self.num_filters_des = 64

        # 局部特征融合
        self.num_filters_lff = 64

        # -------

        # 预处理
        self.pre = nn.Conv2d(self.num_input_channel,
                             self.num_filters_pre,
                             self.size_kernel_pre,
                             padding=0)

        self.prelu = nn.PReLU(self.num_filters_pre)

        self.RDB1 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB2 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB3 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB4 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB5 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB6 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB7 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)
        self.RDB8 = ResDenseBlock(num_channels_in=self.num_filters_pre,
                                  num_layers_des=4)

        self.LFF1 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF2 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF3 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF4 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF5 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF6 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF7 = LocalFeatureFusion(num_channels_in=self.num_filters_des)
        self.LFF8 = LocalFeatureFusion(num_channels_in=self.num_filters_des)

        self.PS = PixelShuffle(num_channels_in=self.num_filters_lff)

    def forward(self, x):
        #x = x.permute(0,3,1,2)
        x = self.pre(x)
        x = self.prelu(x)
        rdb1 = self.RDB1(x)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb4 = self.RDB4(rdb3)
        rdb5 = self.RDB5(rdb4)
        rdb6 = self.RDB6(rdb5)
        lff1 = self.LFF1(torch.cat([x, rdb6], 1))
        lff2 = self.LFF2(torch.cat([lff1, rdb5], 1))
        lff3 = self.LFF3(torch.cat([lff2, rdb4], 1))
        lff4 = self.LFF4(torch.cat([lff3, rdb3], 1))
        lff5 = self.LFF5(torch.cat([lff4, rdb2], 1))
        lff6 = self.LFF6(torch.cat([lff5, rdb1], 1))
        ps = self.PS(lff6)
        return ps


class ResDenseBlock(nn.Module):
    def __init__(self, num_channels_in, num_layers_des):
        super(ResDenseBlock, self).__init__()
        self.num_channels_in = num_channels_in
        self.num_layers_des = num_layers_des
        # 残差密集块feature map数量
        self.num_filters_des = 64
        self.size_kernel_des = 3
        # 瓶颈层
        self.size_kernel_bl = 1

        self.modlist = nn.ModuleList([])

        # 密集连接卷积网络
        for layer in range(self.num_layers_des):
            # Bottleneck
            bl = nn.Conv2d(self.num_channels_in,
                           self.num_filters_des,
                           self.size_kernel_bl,
                           padding=0,
                           bias=False)

            # Convolution
            conv = nn.Conv2d(self.num_filters_des,
                             self.num_filters_des,
                             self.size_kernel_des,
                             padding=(self.size_kernel_des - 1) // 2)

            # ReLU
            relu = nn.ReLU()

            # 残差密集块特征图输入数量
            self.num_channels_in = self.num_channels_in + self.num_filters_des

            # 添加到模型列表
            self.modlist.append(bl)
            self.modlist.append(conv)
            self.modlist.append(relu)

        # Bottleneck
        self.bl_res = nn.Conv2d(self.num_channels_in,
                                self.num_filters_des,
                                self.size_kernel_bl,
                                padding=0,
                                bias=False)

    def forward(self, x):
        # 输出列表
        # 原始输入
        source = x
        # 每层输出放入列表
        list_des = [x]
        for des in range(0, self.num_layers_des * 3, 3):
            # Concatenation
            x = torch.cat(list_des, 1)
            x = self.modlist[des + 0](x)
            x = self.modlist[des + 1](x)
            x = self.modlist[des + 2](x)
            list_des.append(x)
        x = torch.cat(list_des, 1)
        x = self.bl_res(x)
        return source + x


class LocalFeatureFusion(nn.Module):
    def __init__(self, num_channels_in):
        super(LocalFeatureFusion, self).__init__()
        self.num_channels_in = num_channels_in
        self.size_kernel_bl = 1
        # 局部特征融合参数
        self.size_kernel_lff = 3
        self.num_filters_lff = 64

        # Bottleneck
        self.bl = nn.Conv2d(self.num_channels_in * 2,
                       self.num_filters_lff,
                       self.size_kernel_bl,
                       padding=0,
                       bias=False)

        self.lff = nn.Conv2d(self.num_filters_lff,
                        self.num_filters_lff,
                        self.size_kernel_lff,
                        padding=(self.size_kernel_lff - 1) // 2)

        # PReLU
        self.prelu = nn.PReLU(self.num_filters_lff)

    def forward(self, x):
        x = self.bl(x)
        x = self.lff(x)
        x = self.prelu(x)

        return x


class PixelShuffle(nn.Module):
    def __init__(self, num_channels_in):
        super(PixelShuffle, self).__init__()
        self.num_channels_in = num_channels_in
        # 超分缩放因子
        self.factor_resolution = 2
        self.size_kernel_bl = 1
        self.size_kernel_ps = 3
        self.num_filters_ps = 1

        # 亚像素卷积
        self.sub = nn.Conv2d(self.num_channels_in,
                        self.num_channels_in,
                        self.size_kernel_ps,
                        padding=0)

        # 像素重组卷积
        self.ps = nn.PixelShuffle(self.factor_resolution)

        # 像素重组后通道数
        self.num_channels_ps = self.num_channels_in // pow(self.factor_resolution,
                                                      2)

        # 卷积
        self.conv = nn.Conv2d(self.num_channels_ps,
                         self.num_filters_ps,
                         self.size_kernel_ps,
                         padding=(self.size_kernel_ps - 1) // 2)

    def forward(self, x):
        x = self.sub(x)
        x = self.ps(x)
        x = self.conv(x)

        return x
