import tensorflow as tf
from cv2 import cv2 as cv
import numpy as np
import h5py

try:
    xrange
except:
    xrange = range

train_continue = 82
if train_continue != 0:
    checkpoint_path = "model\\Entropia_jpg_lv2-" + str(train_continue)
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta")

tf.reset_default_graph()


#--函数类----------------------------------------------------------------------------
# 定义PReLU
def PReLU(_x, name):
    """parametric ReLU activation"""
    with tf.variable_scope("PReLU", reuse=tf.AUTO_REUSE):
        _alpha = tf.get_variable(name=name + "prelu",
                                 shape=_x.get_shape()[-1],
                                 dtype=_x.dtype,
                                 initializer=tf.constant_initializer(0.1))
        pos = tf.nn.relu(_x)
        neg = _alpha * (_x - tf.abs(_x)) * 0.5
        return pos + neg


#--数据集----------------------------------------------------------------------------
with h5py.File("checkpoint\\train_jpg_lv2.h5", "r") as hf:
    data_train_low = np.array(hf.get("jpg"))
    data_train_high = np.array(hf.get("hr"))

batch_size = 32
dataset_size = len(data_train_low)

#--网络参数----------------------------------------------------------------------------
# 分辨率
size_low_image = 33
size_high_image = 25

# 输入数据通道数
num_channel_input_Y = 1
num_channel_input_U = 2
num_channel_input_V = 2

# 卷积神经网络
Y_kernel_size_conv = 5  # Y通道处理卷积
Y_num_filters_conv = 32  # Y通道处理深度

block_num_filters_conv=16

U_kernel_size_conv1 = 5  # Y通道处理卷积
U_num_filters_conv1 = 32  # Y通道处理深度

V_kernel_size_conv1 = 5  # V通道处理卷积
V_num_filters_conv1 = 32  # V通道处理深度

kernel_size_conv1 = 1  # 非线性
num_filters_conv1 = 32

U_kernel_size_conv2 = 1  # U通道非线性
U_num_filters_conv2 = 16

V_kernel_size_conv2 = 1  # V通道非线性
V_num_filters_conv2 = 16

dense_num = 3

merge_num_filters_conv2 = block_num_filters_conv * 7 + num_filters_conv1  # 最终合并层深度

kernel_size_conv3 = 1  # 瓶颈层
num_filters_conv3 = 64

merge_num_filters_conv_YUV = num_filters_conv3 + U_num_filters_conv2 + V_num_filters_conv2

kernel_size_conv4 = 3  # 第4次
num_filters_conv4 = 36

kernel_size_conv5 = 3  # 第5次
num_filters_conv5 = 3


# 残差密集网络
def ResNet(input_layer,
           input_num_filters_conv,
           kernel_size_conv,
           layers,
           name=None):

    global block_num_filters_conv

    des_kernel_size_conv = kernel_size_conv  #
    des_num_filters_conv = 16

    bk_kernel_size_conv = 1  # Bottleneck层

    # 获取Identity层,直接映射
    identity_weight_conv = tf.Variable(
        tf.random_normal([
            bk_kernel_size_conv, bk_kernel_size_conv, input_num_filters_conv,
            block_num_filters_conv
        ],
                         stddev=1e-3))
    identity_layer = tf.nn.conv2d(input_layer,
                                  identity_weight_conv,
                                  strides=[1, 1, 1, 1],
                                  padding="VALID")

    # 前一层数据
    before_num_filters_conv = input_num_filters_conv
    before_layer_conv = input_layer

    merge_conv = [input_layer]

    for l in range(layers):
        # 权值和偏置
        # Bottleneck
        bk_des_weight_conv = tf.Variable(
            tf.random_normal([
                bk_kernel_size_conv, bk_kernel_size_conv,
                before_num_filters_conv, des_num_filters_conv
            ],
                             stddev=1e-3))
        bk_des_bias_conv = tf.Variable(tf.zeros([des_num_filters_conv]))

        # 卷积层
        des_weight_conv = tf.Variable(
            tf.random_normal([
                des_kernel_size_conv, des_kernel_size_conv,
                des_num_filters_conv, des_num_filters_conv
            ],
                             stddev=1e-3))
        des_bias_conv = tf.Variable(tf.zeros([des_num_filters_conv]))

        # Bottleneck
        bk_des_layer_conv = tf.nn.conv2d(before_layer_conv,
                                         bk_des_weight_conv,
                                         strides=[1, 1, 1, 1],
                                         padding="SAME")
        bk_des_layer_conv = tf.nn.bias_add(bk_des_layer_conv, bk_des_bias_conv)

        # CNN
        des_layer_conv = tf.nn.conv2d(bk_des_layer_conv,
                                      des_weight_conv,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME")
        des_layer_conv = tf.nn.bias_add(des_layer_conv, des_bias_conv)
        des_layer_conv = PReLU(des_layer_conv, name)

        # 前一层添加到队列
        merge_conv.append(des_layer_conv)

        #
        before_layer_conv = tf.concat(merge_conv, -1)

        before_num_filters_conv += des_num_filters_conv

    # Bottleneck层
    bk_weight_conv = tf.Variable(
        tf.random_normal([
            bk_kernel_size_conv, bk_kernel_size_conv, before_num_filters_conv,
            block_num_filters_conv
        ],
                         stddev=1e-3))
    bk_bias_conv = tf.Variable(tf.zeros([block_num_filters_conv]))

    # Bottleneck
    bk_layer_conv = tf.nn.conv2d(before_layer_conv,
                                 bk_weight_conv,
                                 strides=[1, 1, 1, 1],
                                 padding="VALID")
    bk_layer_conv = tf.nn.bias_add(bk_layer_conv, bk_bias_conv)

    # Res
    res_layer_conv = tf.add(bk_layer_conv, identity_layer)

    return res_layer_conv


#--PlaceHolder----------------------------------------------------------------------------
image_Y = tf.placeholder(tf.float32, [None, size_low_image, size_low_image, 1],
                         name="y")

image_U = tf.placeholder(tf.float32, [None, size_low_image, size_low_image, 1],
                         name="u")

image_V = tf.placeholder(tf.float32, [None, size_low_image, size_low_image, 1],
                         name="v")

image_HR = tf.placeholder(tf.float32,
                          [None, size_high_image, size_high_image, 3],
                          name="high_r")

# 预处理卷积
Y_weight_conv = tf.Variable(tf.random_normal([
    Y_kernel_size_conv, Y_kernel_size_conv, num_channel_input_Y,
    Y_num_filters_conv
],
                                             stddev=1e-1),
                            name="W_Y")
Y_bias_conv = tf.Variable(tf.zeros([Y_num_filters_conv]), name="B_Y")

U_weight_conv1 = tf.Variable(
    tf.random_normal([
        U_kernel_size_conv1, U_kernel_size_conv1, num_channel_input_U,
        U_num_filters_conv1
    ],
                     stddev=1e-1))
U_bias_conv1 = tf.Variable(tf.zeros([U_num_filters_conv1]))

V_weight_conv1 = tf.Variable(
    tf.random_normal([
        V_kernel_size_conv1, V_kernel_size_conv1, num_channel_input_V,
        V_num_filters_conv1
    ],
                     stddev=1e-1))
V_bias_conv1 = tf.Variable(tf.zeros([V_num_filters_conv1]))

U_weight_conv2 = tf.Variable(
    tf.random_normal([
        U_kernel_size_conv2, U_kernel_size_conv2, U_num_filters_conv1,
        U_num_filters_conv2
    ],
                     stddev=1e-1))
U_bias_conv2 = tf.Variable(tf.zeros([U_num_filters_conv2]))

V_weight_conv2 = tf.Variable(
    tf.random_normal([
        V_kernel_size_conv2, V_kernel_size_conv2, V_num_filters_conv1,
        V_num_filters_conv2
    ],
                     stddev=1e-1))
V_bias_conv2 = tf.Variable(tf.zeros([V_num_filters_conv2]))

# 卷积层1
weight_conv1 = tf.Variable(
    tf.random_normal([
        kernel_size_conv1, kernel_size_conv1, Y_num_filters_conv,
        num_filters_conv1
    ],
                     stddev=1e-3))

bias_conv1 = tf.Variable(tf.zeros([num_filters_conv1]))

# 卷积层3
weight_conv3 = tf.Variable(tf.random_normal([
    kernel_size_conv3, kernel_size_conv3, merge_num_filters_conv2,
    num_filters_conv3
],
                                            stddev=1e-3),
                           name="W3")

bias_conv3 = tf.Variable(tf.zeros([num_filters_conv3]), name="B3")

# 卷积层4
weight_conv4 = tf.Variable(tf.random_normal([
    kernel_size_conv4, kernel_size_conv4, merge_num_filters_conv_YUV,
    num_filters_conv4
],
                                            stddev=1e-3),
                           name="W4")
bias_conv4 = tf.Variable(tf.zeros([num_filters_conv4]), name="B4")

# 卷积层5
weight_conv5 = tf.Variable(tf.random_normal([
    kernel_size_conv5, kernel_size_conv5, num_filters_conv4, num_filters_conv5
],
                                            stddev=1e-3),
                           name="W5")

bias_conv5 = tf.Variable(tf.zeros([num_filters_conv5]), name="B5")

#--卷积神经网络----------------------------------------------------------------------------
# Y_CNN
Y_layer_conv = tf.nn.conv2d(image_Y,
                            Y_weight_conv,
                            strides=[1, 1, 1, 1],
                            padding="VALID")  # 卷积
Y_layer_conv = tf.nn.bias_add(Y_layer_conv, Y_bias_conv)  # 添加偏置

image_merge_U = tf.concat([image_Y, image_U], -1)

U_layer_conv1 = tf.nn.conv2d(image_merge_U,
                             U_weight_conv1,
                             strides=[1, 1, 1, 1],
                             padding="VALID")  # 卷积
U_layer_conv1 = tf.nn.bias_add(U_layer_conv1, U_bias_conv1)  # 添加偏置
U_layer_conv1 = PReLU(U_layer_conv1, name="U1_")

image_merge_V = tf.concat([image_Y, image_V], -1)

V_layer_conv1 = tf.nn.conv2d(image_merge_V,
                             V_weight_conv1,
                             strides=[1, 1, 1, 1],
                             padding="VALID")  # 卷积
V_layer_conv1 = tf.nn.bias_add(V_layer_conv1, V_bias_conv1)  # 添加偏置
V_layer_conv1 = PReLU(V_layer_conv1, name="V1_")

U_layer_conv2 = tf.nn.conv2d(U_layer_conv1,
                             U_weight_conv2,
                             strides=[1, 1, 1, 1],
                             padding="VALID")  # 卷积
U_layer_conv2 = tf.nn.bias_add(U_layer_conv2, U_bias_conv2)  # 添加偏置
U_layer_conv2 = PReLU(U_layer_conv2, name="U2_")

V_layer_conv2 = tf.nn.conv2d(V_layer_conv1,
                             V_weight_conv2,
                             strides=[1, 1, 1, 1],
                             padding="VALID")  # 卷积
V_layer_conv2 = tf.nn.bias_add(V_layer_conv2, V_bias_conv2)  # 添加偏置
V_layer_conv2 = PReLU(V_layer_conv2, name="V2_")

# ResCNN
bp1_res_layer_conv1 = ResNet(Y_layer_conv,
                             Y_num_filters_conv,
                             5,
                             2,
                             name="bp1_res1_")

bp1_res_layer_conv2 = ResNet(bp1_res_layer_conv1,
                             block_num_filters_conv,
                             5,
                             2,
                             name="bp1_res2_")

bp1_res_layer_conv3 = ResNet(bp1_res_layer_conv2,
                             block_num_filters_conv,
                             5,
                             2,
                             name="bp1_res3_")

bp2_res_layer_conv1 = ResNet(Y_layer_conv,
                             Y_num_filters_conv,
                             3,
                             3,
                             name="bp2_res1_")

bp2_res_layer_conv2 = ResNet(bp2_res_layer_conv1,
                             block_num_filters_conv,
                             3,
                             3,
                             name="bp2_res2_")

bp2_res_layer_conv3 = ResNet(bp2_res_layer_conv2,
                             block_num_filters_conv,
                             3,
                             3,
                             name="bp2_res3_")

bp2_res_layer_conv4 = ResNet(bp2_res_layer_conv3,
                             block_num_filters_conv,
                             3,
                             3,
                             name="bp2_res4_")

# CNN1非线性
layer_conv1 = tf.nn.conv2d(Y_layer_conv,
                           weight_conv1,
                           strides=[1, 1, 1, 1],
                           padding="VALID")
layer_conv1 = tf.nn.bias_add(layer_conv1, bias_conv1)
layer_conv1=PReLU(layer_conv1,"Y1_")

# 合并层
layer_conv2_merge = tf.concat([
    layer_conv1,
    bp1_res_layer_conv1,
    bp1_res_layer_conv2,
    bp1_res_layer_conv3,
    bp2_res_layer_conv1,
    bp2_res_layer_conv2,
    bp2_res_layer_conv3,
    bp2_res_layer_conv4,
], -1)

# BottleneckCNN3
layer_conv3 = tf.nn.conv2d(layer_conv2_merge,
                           weight_conv3,
                           strides=[1, 1, 1, 1],
                           padding="VALID")
layer_conv3 = tf.nn.bias_add(layer_conv3, bias_conv3)

# 同都合并
layer_YUV_merge = tf.concat([layer_conv3, U_layer_conv2, V_layer_conv2], -1)

# CNN4
layer_conv4 = tf.nn.conv2d(layer_YUV_merge,
                           weight_conv4,
                           strides=[1, 1, 1, 1],
                           padding="VALID")
layer_conv4 = tf.nn.bias_add(layer_conv4, bias_conv4)

# CNN5
layer_final = tf.nn.conv2d(layer_conv4,
                           weight_conv5,
                           strides=[1, 1, 1, 1],
                           padding="VALID",
                           name="output_image")

#--损失----------------------------------------------------------------------------
# L1
l1 = tf.reduce_mean(tf.abs(image_HR - layer_final))

# SSIM
ssim = tf.reduce_mean(tf.image.ssim(image_HR, layer_final, max_val=1.0))

# PSNR
psnr = tf.reduce_mean(tf.image.psnr(image_HR, layer_final, max_val=1.0))

# 损失
alpha = 0.84
loss = alpha * (1 - ssim) + (1 - alpha) * l1

#--学习率----------------------------------------------------------------------------
global_step = tf.Variable(train_continue + 1)  #定义global_step 它会自动+1

learning_rate_continue = 1e-3 * pow(0.99, train_continue)
learning_rate = tf.train.exponential_decay(
    1e-3, global_step, 200, 0.98, staircase=True) + 1e-5  #生成学习率

optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

#--训练网络----------------------------------------------------------------------------
# 变量初始化
init_op = tf.global_variables_initializer()
# 保存参数
vl = [v for v in tf.global_variables() if "Adam" not in v.name]
saver = tf.train.Saver(var_list=vl)

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 恢复参数
    if train_continue != 0:
        saver.restore(sess, checkpoint_path)

    # 设定训练的轮数
    STEPS = 1000
    for ep in xrange(train_continue + 1, STEPS):
        batch_step = dataset_size // batch_size
        for i in xrange(batch_step):

            # 每次选取batch_size个样本进行训
            start = i * batch_size
            end = (i + 1) * batch_size

            feed_dict_train = {
                image_Y: data_train_low[start:end, :, :, 0:1],
                image_U: data_train_low[start:end, :, :, 1:2],
                image_V: data_train_low[start:end, :, :, 2:3],
                image_HR: data_train_high[start:end]
            }

            # 通过选取的样本进行训练神经网络并更新参数
            _, LOSS, SSIM, PSNR,rate = sess.run([optimizer1, loss, ssim, psnr,learning_rate],
                                           feed_dict=feed_dict_train)

            if i % 100 == 0:
                print(i)
                # 性能标准
                print("%d training epoch(s) \
                \nLOSS on dataset is %.8f \
                \nSSIM on dataset is %.8f \
                \nPSNR on dataset is %.6f \
                \nLearning Rate is %.6f" % (ep, LOSS, SSIM, PSNR, rate))

            if i == batch_step - 1:
                with open("log/entropia-jpg_lv2.txt", "a+") as f:
                    print("%d\t%.8f\t%.8f\t%.6f" % (ep, LOSS, SSIM, PSNR),
                          file=f)

                # 保存模型
                saver.save(sess, "model/Entropia_jpg_lv2", global_step=ep)

                # 保存模型
                #saver.save(sess, "model-anime/Entropia_anime_mk2", global_step=ep)
                '''
                # 查看中间结果
                feed_dict_temp = {
                    image_Y: data_train_low[0:1, :, :, 0:1],
                    image_U: data_train_low[0:1, :, :, 1:2],
                    image_V: data_train_low[0:1, :, :, 2:3],
                    image_HR: data_train_high[0:1]
                }

                layer_final_temp = sess.run(layer_final,
                                            feed_dict=feed_dict_temp).reshape(
                                                25, 25, 3)
                cv.imshow("TEMP of LR", data_train_low[0])
                cv.imshow("TEMP of HR", data_train_high[0])
                cv.imshow("TEMP of layer final", layer_final_temp)

                cv.waitKey(200)'''
