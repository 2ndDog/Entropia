import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv
import global_variable as Gvar
import math
import sys

try:
    xrange
except:
    xrange = range


def Enlarge(path_image_input, path_image_output, multiple=None, size=None):
    # 初始化信号量
    # 检查运行状态
    if Gvar.SignalRuning() == 0:  # 运行状态未占用
        Gvar.Running(1)
        Gvar.setLogs("开始执行...", 0)
    else:
        Gvar.setLogs("执行失败,运行状态被占用,ID:" + str(Gvar.SignalRuning()), 2)
        Gvar.Ending()
        return 1

    # 模型路径
    module_path = "model\\Entropia-88"
    #module_path = "model\\MSRN-35"

    try:
        saver = tf.train.import_meta_graph(module_path + ".meta")
    except:
        Gvar.setLogs("找不到模型", 2)
        Gvar.Ending()
        return 1

    low = 33

    high = 54

    stride = 17

    resolution = 2

    pad = (low * resolution - high) // (resolution * 2)

    stride_high = stride * resolution

    edge = (high - stride * resolution) - 2

    if edge <= 0:
        Gvar.setLogs("Error edge!", 2)
        Gvar.Ending()
        return 1

    image = cv.imread(path_image_input)

    if image is None:
        Gvar.setLogs("图片加载失败", 2)
        Gvar.Ending()
        return 1

    # 归一化
    image = image / 255

    height_image_input = image.shape[0]
    width_image_input = image.shape[1]

    mode = 0
    # 处理模式
    if multiple != None:
        mode = 1
    if size != None:
        mode = 2

    if mode == 0:
        Gvar.setLogs("处理模式无效", 2)
        Gvar.Ending()
        return 1

    if mode == 1:
        iterations = math.ceil(multiple / 2)  # 迭代次数
    if mode == 2:
        iterations = math.ceil(
            max((size[0] / height_image_input),
                (size[1] / width_image_input)) / 2)

    input = tf.get_default_graph().get_tensor_by_name("low_r:0")
    output = tf.get_default_graph().get_tensor_by_name("output_image:0")

    for it in range(iterations):
        # 填充
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), "edge")

        height = image.shape[0]
        width = image.shape[1]

        # 记录高和宽的块数
        height_block_size = (height + stride - low) // stride
        width_block_size = (width + stride - low) // stride

        if (height_block_size == 0 or width_block_size == 0):
            Gvar.setLogs("图片太小", 2)
            Gvar.Ending()
            return 1
        # 重设大小
        # image = image[0:(height_block_size * stride + low - stride),
        #0:(width_block_size * stride + low - stride)]

        # 通道分离
        image_channels = cv.split(image)

        # 最终图像
        result_image_array = []

        with tf.Session() as sess:
            # 载入model
            saver.restore(sess, module_path)

            for channel in xrange(len(image_channels)):  # 通道
                result_image = np.zeros(
                    (height * resolution - (low * resolution - high),
                     width * resolution - (low * resolution - high)))

                for row in xrange(height_block_size + 1):  # 高几个图片
                    for col in xrange(width_block_size + 1):  # 宽几个图片

                        # 中止信号
                        if Gvar.SignalTermination():
                            Gvar.setLogs("取消", 0)
                            Gvar.Ending()
                            return 2

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

                        small_image = image_channels[channel][x0:x1, y0:y1]

                        # 重塑图片
                        small_image = small_image.reshape(1, low, low, 1)

                        # CNN
                        result = sess.run(output,
                                          feed_dict={
                                              input: small_image,
                                          })

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

                        if row == height_block_size:
                            result_x = result_image.shape[0] - high
                        if col == width_block_size:
                            result_y = result_image.shape[1] - high

                        for h in range(h_offset, high):
                            for w in range(w_offset, high):
                                result_image[result_x + h][result_y +
                                                           w] = result[h][w]

                        #柔化边缘
                        for h in xrange(edge):
                            for w in xrange(high):
                                result_image[result_x + h][result_y + w] = (
                                    result_image[result_x + h][result_y + w] *
                                    (1 - (h / edge))) + (result[h][w] *
                                                         (h / edge))

                        for h in xrange(high):
                            for w in xrange(edge):
                                result_image[result_x + h][result_y + w] = (
                                    result_image[result_x + h][result_y + w] *
                                    (1 - (w / edge))) + (result[h][w] *
                                                         (w / edge))

                        # 输出进度
                        print(
                            "\r%.2f%%" %
                            (((height_block_size * width_block_size *
                               len(image_channels) * it +
                               height_block_size * width_block_size * channel +
                               row * width_block_size + col) /
                              (height_block_size * width_block_size *
                               len(image_channels) * iterations)) * 100),
                            end="")

                        # 输出进度
                        Gvar.setProgress(
                            (height_block_size * width_block_size *
                             len(image_channels) * it +
                             height_block_size * width_block_size * channel +
                             row * width_block_size + col) /
                            (height_block_size * width_block_size *
                             len(image_channels) * iterations))

                result_image_array.append(result_image)

        result_image = cv.merge(result_image_array)

        image = result_image

    # 后处理
    image = image * 255
    image = np.maximum(image, 0)
    image = np.minimum(image, 255)
    image = np.around(image)
    image = image.astype(np.uint8)
    # 模式
    if mode == 1:
        result_image = cv.resize(image, (int(
            width_image_input * multiple), int(height_image_input * multiple)),
                                 interpolation=cv.INTER_CUBIC)
    if mode == 2:
        result_image = cv.resize(image, (size[1], size[0]),
                                 interpolation=cv.INTER_CUBIC)

    cv.imwrite(path_image_output, result_image)

    # 结束信号
    Gvar.Ending()
    Gvar.setLogs("转换完成", 0)
