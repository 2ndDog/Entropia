import torch
from collections import deque
import numpy as np
import os
import cv2 as cv
import math
import sys
import RDDLN

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu

PATH = ".\\checkpoint\\RDDLN.pth"

size_cut_edge = 23

# 部署模型
net = RDDLN.RDDLN()

# 加载权重
net.load_state_dict(torch.load(PATH))
net.eval()

# 检查GPU是否可用
if torch.cuda.is_available():
    net = net.cuda()

# 读取图像路径,防止中文路径问题
def cv_imread(file_path=""):
    file_path_gbk = file_path.encode('gbk')  # unicode转gbk，字符串变为字节数组
    img_mat = cv.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
    return img_mat


def SuperResolution(path_image_input):
    low = 64

    high = 120

    stride = 56

    resolution = 2

    batch_size = 20

    pad = (low * resolution - high) // (resolution * 2)

    stride_high = stride * resolution

    edge = (high - stride_high) // 2

    # 读取图像
    image = cv_imread(path_image_input)

    # 转化为YUV再处理
    image = cv.cvtColor(image, cv.COLOR_BGR2YUV)

    # 归一化
    image = image / 255

    # 填充
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), "edge")

    height = image.shape[0]
    width = image.shape[1]

    # 记录高和宽的块数
    height_block_size = (height + stride - low) // stride
    width_block_size = (width + stride - low) // stride

    # 多一块用于边缘
    rows = height_block_size + 1  # 高几个图片
    cols = width_block_size + 1  # 宽几个图片

    # 通道分离
    list_image_channels = cv.split(image)

    # 用于存储子图像的列表
    list_sub_image_lr = []
    deque_sub_image_sr = deque()

    # 最终图像
    list_image_result = []


    for channel in range(len(list_image_channels)):  # 通道
        for row in range(rows):
            for col in range(cols):

                # 坐标
                x0 = row * stride
                x1 = x0 + low
                y0 = col * stride
                y1 = y0 + low

                # 边缘坐标
                if row == height_block_size:
                    x0 = height - low
                    x1 = height

                if col == width_block_size:
                    y0 = width - low
                    y1 = width

                # 裁剪子图像
                sub_image = list_image_channels[channel][x0:x1, y0:y1]

                sub_image = sub_image.reshape((1, low, low))

                # 子图像添加到列表
                list_sub_image_lr.append(sub_image)

                # 输出当前进度
                print("\r裁取子图像: %.2f%%" %
                      ((rows * cols * channel + row * cols + col) /
                       (rows * cols * len(list_image_channels)) * 100),
                      end="")

    print()
    # 按Batch size张图进行超分
    for sub in range(math.ceil(len(list_sub_image_lr) / batch_size)):
        start = sub * batch_size
        end = min((sub + 1) * batch_size, len(list_sub_image_lr))

        lr = (torch.Tensor(np.array(list_sub_image_lr[start:end]))).cuda()
        

        # 超分辨率
        sr = net(torch.tensor(lr).to(torch.float32)).permute(0,2,3,1).detach().cpu().numpy()

        # 把每张图添加到队列
        for btch in range(len(sr)):
            deque_sub_image_sr.append(sr[btch])

        print("\r超分辨率%.2f%%" %
              (sub / (math.ceil(len(list_sub_image_lr) / batch_size)) * 100),
              end="")

    print()

    # 重建SR图像
    for channel in range(len(list_image_channels)):  # 通道
        image_result = np.zeros(
            (height * resolution - (low * resolution - high),
             width * resolution - (low * resolution - high)))

        for row in range(rows):
            for col in range(cols):
                # 坐标
                x0 = row * stride
                x1 = x0 + low
                y0 = col * stride
                y1 = y0 + low

                # 边缘坐标
                if row == height_block_size:
                    x0 = height - low
                    x1 = height

                if col == width_block_size:
                    y0 = width - low
                    y1 = width

                # 取一张SR子图像
                result = deque_sub_image_sr.popleft()

                # 逆重塑图片
                result = result.reshape(high, high)

                # 是否在边缘
                if row == 0:
                    h_offset = 0
                else:
                    h_offset = edge

                if col == 0:
                    w_offset = 0
                else:
                    w_offset = edge

                # 重建像素位置
                result_x = row * stride_high
                result_y = col * stride_high
                # 如果重建位置在边缘
                if row == height_block_size:
                    result_x = image_result.shape[0] - high
                if col == width_block_size:
                    result_y = image_result.shape[1] - high

                # 拼接图像
                for h in range(h_offset, high):
                    for w in range(w_offset, high):
                        image_result[result_x + h][result_y + w] = result[h][w]

                #柔化拼接边缘
                for h in range(edge):
                    for w in range(high):
                        image_result[result_x + h][result_y + w] = (
                            image_result[result_x + h][result_y + w] *
                            (1 - (h / edge))) + (result[h][w] * (h / edge))

                for h in range(high):
                    for w in range(edge):
                        image_result[result_x + h][result_y + w] = (
                            image_result[result_x + h][result_y + w] *
                            (1 - (w / edge))) + (result[h][w] * (w / edge))

                # 输出进度
                print("\r图像重建%.2f%%" %
                      ((rows * cols * channel + row * cols + col) /
                       (rows * cols * len(list_image_channels)) * 100),
                      end="")

        list_image_result.append(image_result)

    image_result = cv.merge(list_image_result)

    image = image_result

    return image


def PostProcessing(image_input, path_image_output):
    # 后处理
    image = image_input * 255
    image = np.clip(image, 0, 255)
    image = np.around(image)
    image = image.astype(np.uint8)

    image = cv.cvtColor(image, cv.COLOR_YUV2BGR)

    cv.imshow("Super-Resolution", image)
    cv.waitKey(0)

    cv.imwrite(path_image_output, image)


path_image_input = ".\\.png"
path_image_output = ".\\s.png"

image = SuperResolution(path_image_input)
PostProcessing(image, path_image_output)
