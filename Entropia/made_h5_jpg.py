from cv2 import cv2 as cv
import numpy as np
import h5py
import os

try:
    xrange
except:
    xrange = range

block_jpg = 33
block_hr = 25

stride = 22

#image_train_path = "D:\\SRCNN-Dataset\\anime\\"
path_image_train_jpg = "D:\\SRCNN-Dataset\\anime\\low\\"
path_image_train_hr = "D:\\SRCNN-Dataset\\anime\\"

list_train_jpg = []
list_train_hr = []

for num in range(1, 35):

    # 读取图片
    image_train_jpg = cv.imread(path_image_train_jpg + str(num) + ".jpg")
    image_train_hr = cv.imread(path_image_train_hr + str(num) + ".png")

    # 转化为YCrCb
    image_train_jpg = cv.cvtColor(image_train_jpg, cv.COLOR_BGR2YCrCb)
    image_train_hr = cv.cvtColor(image_train_hr, cv.COLOR_BGR2YCrCb)

    # 归一化
    image_train_jpg = image_train_jpg / 255
    image_train_hr = image_train_hr / 255

    # 图像高和宽
    height = image_train_hr.shape[0]
    width = image_train_hr.shape[1]

    # 裁剪单位图像数量
    rows = (height + stride - block_jpg) // stride
    cols = (width + stride - block_jpg) // stride

    # 裁剪图片
    for row in xrange(rows//4,rows//4*3):
        for col in xrange(cols//2):

            # 坐标
            x0_jpg = row * stride
            x1_jpg = x0_jpg + block_jpg
            y0_jpg = col * stride
            y1_jpg = y0_jpg + block_jpg

            # 偏移量
            offset = (block_jpg - block_hr) // 2

            x0_hr = x0_jpg + offset
            x1_hr = x0_hr + block_hr
            y0_hr = y0_jpg + offset
            y1_hr = y0_hr + block_hr

            # 裁剪原图
            image_cut_jpg = image_train_jpg[x0_jpg:x1_jpg, y0_jpg:y1_jpg]
            image_cut_hr = image_train_hr[x0_hr:x1_hr, y0_hr:y1_hr]

            # 传入数组
            list_train_jpg.append(image_cut_jpg)
            list_train_hr.append(image_cut_hr)

# 转换位numpy.array
data_train_jpg = np.array(list_train_jpg)
data_train_hr = np.array(list_train_hr)

# 数据打乱
data_size = data_train_jpg.shape[0]
np.random.seed(0)
permutation = np.random.permutation(data_size)
data_train_low_ooo = data_train_jpg[permutation, :, :, :]
data_train_high_ooo = data_train_hr[permutation, :, :, :]

# 保存为h5文件
h5path = os.path.join(os.getcwd(), "checkpoint\\train_jpg_lv2.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset("jpg", data=data_train_low_ooo)
    hf.create_dataset("hr", data=data_train_high_ooo)
