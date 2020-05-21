from cv2 import cv2 as cv
import numpy as np
import h5py
import os

try:
    xrange
except:
    xrange = range

block_low = 33
block_high = 50

resolution = 2

stride = 29

# Set5
image_train_path = "D:\\SRCNN-Dataset\\Set5\\test\\"

# Set14
#image_train_path = "D:\\SRCNN-Dataset\\Set5\\test\\"

extension=".bmp"

list_train_low = []
list_train_high = []

def NumImages(path,extension_image):
    counter_images=0
    all_folds = os.listdir(path)   #解析出父文件夹中所有的文件名称，并以列表的格式输出
    for i in range(len(all_folds)):
        (filename, extension) = os.path.splitext(all_folds[i])
        if extension==extension_image:
            counter_images+=1

    return counter_images

def ReducedResolution(divisor, input_image):
    height = input_image.shape[0]
    width = input_image.shape[1]

    output_image = cv.resize(input_image,
                             (width // divisor, height // divisor),
                             interpolation=cv.INTER_CUBIC)
    '''
    output_image = cv.resize(output_image, (height, width),
                             interpolation=cv.INTER_CUBIC)
                             '''

    return output_image


def ReduceImage(input_image, block_size):
    output_image = cv.resize(input_image, (block_size, block_size),
                             interpolation=cv.INTER_CUBIC)

    return output_image


for num in range(1, NumImages(image_train_path,extension)):

    # 读取图片
    image_train = cv.imread(image_train_path + str(num) + extension)

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
    for row in xrange(rows):
        for col in xrange(cols):

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
h5path = os.path.join(os.getcwd(), "checkpoint\\train_set5.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset("lr", data=data_train_low)
    hf.create_dataset("hr", data=data_train_high)

