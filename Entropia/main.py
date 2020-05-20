import tensorflow as tf
import numpy as np
from cv2 import cv2 as cv

try:
    xrange
except:
    xrange = range

module_path = "model\\VDSR-97"

saver = tf.train.import_meta_graph(module_path + ".meta")

low = 33

high = 33

stride = 17

resolution = 1

stride_high = stride * resolution

edge = (high - stride * resolution) - 2

if edge <= 0:
    print("Error edge!")
    exit(1)

image = cv.imread("D:\\SRCNN-Dataset\\Set5\\test\\lr.bmp")

if image is None:
    print("图片加载失败!")
    exit(1)

height = image.shape[0]
width = image.shape[1]

# 记录高和宽的块数
height_block_size = (height + stride - low) // stride
width_block_size = (width + stride - low) // stride

# 重设大小
# image = image[0:(height_block_size * stride + low - stride),
              #0:(width_block_size * stride + low - stride)]

# 归一化
image = image / 255

# 通道分离
image_channels = cv.split(image)

# 图像分离后存储
image_data = []

# 最终图像
result_image_array = []

input = tf.get_default_graph().get_tensor_by_name("low_r:0")
output = tf.get_default_graph().get_tensor_by_name("output_image:0")

with tf.Session() as sess:
    # 载入model
    saver.restore(sess, module_path)

    for channel in xrange(len(image_channels)):  # 通道

        # 单通道结果图像
        '''result_image = np.zeros(
            (height_block_size * stride_high + high - stride_high,
             width_block_size * stride_high + high - stride_high))'''

        result_image = np.zeros(
            (height * resolution - (low * resolution - high), width * resolution - (low * resolution - high)))

        for row in xrange(height_block_size + 1):  # 高几个图片
            for col in xrange(width_block_size + 1):  # 宽几个图片

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
                result = sess.run(output, feed_dict={
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
                        result_image[result_x + h][result_y + w] = result[h][w]

                #柔化边缘
                for h in xrange(edge):
                    for w in xrange(high):
                        result_image[result_x + h][result_y + w] = (
                            result_image[result_x + h][result_y + w] *
                            (1 - (h / edge))) + (result[h][w] * (h / edge))

                for h in xrange(high):
                    for w in xrange(edge):
                        result_image[result_x + h][result_y + w] = (
                            result_image[result_x + h][result_y + w] *
                            (1 - (w / edge))) + (result[h][w] * (w / edge))

                # 输出进度
                print("\r%.2f%%" %
                      ((height_block_size * width_block_size * channel +
                        row * width_block_size + col) /
                       (height_block_size * width_block_size *
                        len(image_channels)) * 100),
                      end="")

        result_image_array.append(result_image)

result_image = cv.merge(result_image_array)

cv.imshow("ENTROPIA", result_image)

cv.waitKey(0)

result_image = result_image * 255
cv.imwrite("C:\\Users\\12984\\Desktop\\ENTROPIA.png", result_image)
