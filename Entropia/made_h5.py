from cv2 import cv2 as cv
import numpy as np
import h5py
import os

block_low = 33
block_high = 33

resolution = 2

stride = 71

#image_train_path = "D:\\SRCNN-Dataset\\anime\\"
image_train_path = "D:\\SRCNN-Dataset\\DIV2K_train_HR\\"

list_train_low = []
list_train_high = []


def ReducedResolution(divisor, input_image):
    height = input_image.shape[0]
    width = input_image.shape[1]

    # 缩小
    output_image = cv.resize(input_image,
                             (width // divisor, height // divisor),
                             interpolation=cv.INTER_CUBIC)

    # 放大
    output_image = cv.resize(output_image, (width, height),
                             interpolation=cv.INTER_CUBIC)

    return output_image


for num in range(1, 350):
    # 读取图片
    image_train = cv.imread(image_train_path + str(num).zfill(4) + ".png", cv.CV_8UC1)

    # 归一化
    image_train = image_train / 255

    # 图像高和宽
    height = image_train.shape[0]
    width = image_train.shape[1]

    # 裁剪单位图像数量
    rows = (height + stride - block_low) // stride
    cols = (width + stride - block_low) // stride

    # 裁剪图片
    for row in range(rows):
        for col in range(cols//2):

            # 坐标
            x0_low = row * stride
            x1_low = x0_low + block_low
            y0_low = col * stride
            y1_low = y0_low + block_low

            offset = (block_low - block_high) // 2

            x0_high = x0_low + offset
            x1_high = x0_high + block_high
            y0_high = y0_low + offset
            y1_high = y0_high + block_high

            # 裁剪
            image_cut_low = ReducedResolution(
                2, image_train[x0_low:x1_low, y0_low:y1_low])
            image_cut_high = image_train[x0_high:x1_high, y0_high:y1_high]

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
h5path = os.path.join(os.getcwd(), "checkpoint\\VDSR_train.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset("lr", data=data_train_low_ooo)
    hf.create_dataset("hr", data=data_train_high_ooo)
