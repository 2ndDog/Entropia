import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv
import global_variable as Gvar
import sys

try:
    xrange
except:
    xrange = range

# 读取图像路径,防止中文路径问题
def cv_imread(file_path = ""):
    file_path_gbk = file_path.encode('gbk')        # unicode转gbk，字符串变为字节数组
    img_mat = cv.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
    return img_mat

def NoiseCancelling(path_image_input, path_image_output, level):
    # 初始化信号量
    # 检查运行状态
    if Gvar.SignalRuning() == 0:  # 运行状态未占用
        Gvar.Running(2)
        Gvar.setLogs("开始执行...", 0)
    else:
        Gvar.setLogs("执行失败,运行状态被占用,ID:" + str(Gvar.SignalRuning()), 2)
        Gvar.Ending()
        return 1

    if level == 1:
        # 模型地址
        module_path = "model\\Entropia_jpg_lv1-150"
    if level == 2:
        module_path = "model\\Entropia_jpg_lv2-82"

    try:
        saver = tf.train.import_meta_graph(module_path + ".meta")
    except:
        Gvar.setLogs("找不到模型", 2)
        Gvar.Ending()
        return 1

    low = 33

    high = 25

    stride = 15

    pad = (low - high) // 2

    edge = (high - stride) - 2

    if edge <= 0:
        Gvar.setLogs("Error edge!", 2)
        Gvar.Ending()
        return 1

    image = cv_imread(path_image_input)

    if image is None:
        Gvar.setLogs("图片加载失败", 2)
        Gvar.Ending()
        return 1

    # 填充
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), "reflect")

    height = image.shape[0]
    width = image.shape[1]

    # 记录高和宽的块数
    height_block_size = (height + stride - low) // stride
    width_block_size = (width + stride - low) // stride

    if (height_block_size == 0 or width_block_size == 0):
        Gvar.setLogs("图片太小", 2)
        Gvar.Ending()
        return 1

    # 通道转换
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    # 归一化
    image = image / 255

    input_Y = tf.get_default_graph().get_tensor_by_name("y:0")
    input_U = tf.get_default_graph().get_tensor_by_name("u:0")
    input_V = tf.get_default_graph().get_tensor_by_name("v:0")
    output = tf.get_default_graph().get_tensor_by_name("output_image:0")

    with tf.Session() as sess:
        # 载入model
        saver.restore(sess, module_path)

        result_image = np.zeros(
            (height - (low - high), width - (low - high), 3))

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

                image_unit = image[x0:x1, y0:y1]

                # 通道分离
                image_channels = cv.split(image_unit)

                # 重塑图片
                image_Y = image_channels[0].reshape(1, low, low, 1)
                image_U = image_channels[1].reshape(1, low, low, 1)
                image_V = image_channels[2].reshape(1, low, low, 1)

                # CNN
                result = sess.run(output,
                                  feed_dict={
                                      input_Y: image_Y,
                                      input_U: image_U,
                                      input_V: image_V
                                  })

                # 逆重塑图片
                result = result.reshape(high, high, 3)

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
                result_x = row * stride
                result_y = col * stride

                if row == height_block_size:
                    result_x = result_image.shape[0] - high
                if col == width_block_size:
                    result_y = result_image.shape[1] - high

                for h in range(h_offset, high):
                    for w in range(w_offset, high):
                        result_image[result_x + h][result_y +
                                                   w][:] = result[h][w][:]

                #柔化边缘
                for h in xrange(edge):
                    for w in xrange(high):
                        result_image[result_x + h][result_y + w][:] = (
                            result_image[result_x + h][result_y + w][:] *
                            (1 - (h / edge))) + (result[h][w][:] * (h / edge))

                for h in xrange(high):
                    for w in xrange(edge):
                        result_image[result_x + h][result_y + w][:] = (
                            result_image[result_x + h][result_y + w][:] *
                            (1 - (w / edge))) + (result[h][w][:] * (w / edge))

                # 输出进度
                Gvar.setProgress((row * width_block_size + col) /
                                 (height_block_size * width_block_size))

    result_image = cv.cvtColor(result_image.astype(np.float32),
                               cv.COLOR_YCrCb2BGR)

    # 后处理
    result_image = result_image * 255
    result_image = np.maximum(result_image, 0)
    result_image = np.minimum(result_image, 255)
    result_image = np.around(result_image)
    result_image = result_image.astype(np.uint8)

    cv.imwrite(path_image_output, result_image)

    # 结束信号
    Gvar.Ending()
    Gvar.setLogs("转换完成", 0)