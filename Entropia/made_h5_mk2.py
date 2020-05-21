from cv2 import cv2 as cv
import numpy as np
import h5py
import os

block_low = 33
block_high = 50

resolution = 2

stride = 71

#image_train_path = "D:\\SRCNN-Dataset\\anime\\"
image_train_path = "D:\\SRCNN-Dataset\\DIV2K_train_HR\\"

list_train_low = []
list_train_high = []

def ReduceImage(input_image, block_size):
    output_image = cv.resize(input_image, (block_size, block_size),
                             interpolation=cv.INTER_CUBIC)

    return output_image


for num in range(2, 801,2):

    # 读取图片
    image_train = cv.imread(image_train_path + str(num).zfill(4) + ".png")

    # 转化为YCrCb
    image_train = cv.cvtColor(image_train, cv.COLOR_BGR2YCrCb)

    # 通道分离
    YCrCb_channels = cv.split(image_train)

    # 提取Y通道
    image_train = YCrCb_channels[0]

    # 归一化
    image_train = image_train / 255

    # 图像高和宽
    height = image_train.shape[0]
    width = image_train.shape[1]

    # 裁剪单位图像数量
    rows = (height + stride - (block_low * resolution)) // stride
    cols = (width + stride - (block_low * resolution)) // stride

    # 裁剪图片
    for row in range(rows // 4, rows // 4 * 3):
        for col in range(cols // 4,cols // 4 * 3):

            # 坐标
            x0 = row * stride
            x1 = x0 + (block_low * resolution)
            y0 = col * stride
            y1 = y0 + (block_low * resolution)

            # 裁剪原图
            image_cut_temp = image_train[x0:x1, y0:y1]

            # 偏移量
            offset_high = (block_low * resolution - block_high) // 2

            # 缩小为低分辨率图像
            image_cut_low = ReduceImage(image_cut_temp, block_low)

            # 高分辨率图片坐标
            x0_h = offset_high
            x1_h = x0_h + block_high
            y0_h = offset_high
            y1_h = y0_h + block_high

            # 裁剪高分辨率图
            image_cut_high = image_cut_temp[x0_h:x1_h, y0_h:y1_h]

            # 重塑为3维数组
            image_cut_low = image_cut_low.reshape(block_low, block_low, 1)
            image_cut_high = image_cut_high.reshape(block_high, block_high, 1)

            # 传入数组
            list_train_low.append(image_cut_low)
            list_train_high.append(image_cut_high)

# 转换位numpy.array
data_train_low = np.array(list_train_low)
data_train_high = np.array(list_train_high)

# 数据打乱
data_size = data_train_low.shape[0]
np.random.seed(0)
permutation = np.random.permutation(data_size)
data_train_low_ooo = data_train_low[permutation, :, :, :]
data_train_high_ooo = data_train_high[permutation, :, :, :]

# 保存为h5文件
h5path = os.path.join(os.getcwd(), "checkpoint\\train_msrn.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset("lr", data=data_train_low_ooo)
    hf.create_dataset("hr", data=data_train_high_ooo)
