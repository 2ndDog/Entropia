import tensorflow as tf
from cv2 import cv2 as cv
import numpy as np
import h5py

try:
    xrange
except:
    xrange = range

train_continue = 45
if train_continue != 0:
    checkpoint_path = "model\\Entropia-" + str(train_continue)
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
with h5py.File("checkpoint\\train_test.h5", "r") as hf:
    data_train_low = np.array(hf.get("lr"))
    data_train_high = np.array(hf.get("hr"))

with h5py.File("checkpoint\\train_set5.h5", "r") as hf:
    data_vali_low = np.array(hf.get("lr"))
    data_vali_high = np.array(hf.get("hr"))

batch_size = 64
dataset_size = len(data_train_low)

vali_size = len(data_vali_low)

#--网络参数----------------------------------------------------------------------------
# 分辨率
size_low_image = 33
size_high_image = 54

# 放大倍数
multiple = 2

# 输入数据通道数
input_num_channel = 1

# 卷积神经网络
pre_kernel_size_conv = 5  # 预处理卷积
pre_num_filters_conv = 64  # 预处理深度

block_num_filters_conv = 32  # 每个卷积块的输出

bk_kernel_size_conv = 1  # 瓶颈层

merge1_num_filters_conv = block_num_filters_conv * 3
merge2_num_filters_conv = block_num_filters_conv * 4

lff_kernel_size_conv = 1  # 瓶颈层
lff_num_filters_conv = 64

dff_kernel_size_conv = 3  # 第4次
dff_num_filters_conv = block_num_filters_conv

gff_kernel_size_conv = 1  # 瓶颈层
gff_num_filters_conv = pre_num_filters_conv

num_filters_shuffle = gff_num_filters_conv // pow(multiple, 2)

kernel_size_conv5 = 5  # 第5次
num_filters_conv5 = 1


# 残差密集网络
def ResNet(input_layer,
           input_num_filters_conv,
           kernel_size_conv,
           layers,
           name=None):

    global block_num_filters_conv

    des_kernel_size_conv = kernel_size_conv  #
    des_num_filters_conv = 32

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
image_LR = tf.placeholder(tf.float32,
                          [None, size_low_image, size_low_image, 1],
                          name="low_r")

image_HR = tf.placeholder(tf.float32,
                          [None, size_high_image, size_high_image, 1],
                          name="high_r")

# 预处理卷积
pre_weight_conv = tf.Variable(tf.random_normal([
    pre_kernel_size_conv, pre_kernel_size_conv, input_num_channel,
    pre_num_filters_conv
],
                                               stddev=1e-1),
                              name="P0W0")
pre_bias_conv = tf.Variable(tf.zeros([pre_num_filters_conv]), name="P0B0")

# 局部特征融合1
lff1_weight_conv = tf.Variable(tf.random_normal([
    lff_kernel_size_conv, lff_kernel_size_conv, merge1_num_filters_conv,
    lff_num_filters_conv
],
                                                stddev=1e-3),
                               name="W3")

# 局部特征融合1
lff2_weight_conv = tf.Variable(tf.random_normal([
    lff_kernel_size_conv, lff_kernel_size_conv, merge2_num_filters_conv,
    lff_num_filters_conv
],
                                                stddev=1e-3),
                               name="W3")

# 密集特征融合
dff1_weight_conv = tf.Variable(tf.random_normal([
    dff_kernel_size_conv, dff_kernel_size_conv, lff_num_filters_conv,
    dff_num_filters_conv
],
                                                stddev=1e-3),
                               name="W3")
dff1_bias_conv = tf.Variable(tf.zeros([dff_num_filters_conv]))

# 密集特征融合
dff2_weight_conv = tf.Variable(tf.random_normal([
    dff_kernel_size_conv, dff_kernel_size_conv, lff_num_filters_conv,
    dff_num_filters_conv
],
                                                stddev=1e-3),
                               name="W3")
dff2_bias_conv = tf.Variable(tf.zeros([dff_num_filters_conv]))

# 全局特征融合
gff_weight_conv = tf.Variable(tf.random_normal([
    gff_kernel_size_conv, gff_kernel_size_conv, pre_num_filters_conv,
    pre_num_filters_conv
],
                                               stddev=1e-3),
                              name="W3")

# 卷积层5
weight_conv5 = tf.Variable(tf.random_normal([
    kernel_size_conv5, kernel_size_conv5, num_filters_shuffle,
    num_filters_conv5
],
                                            stddev=1e-3),
                           name="W5")

bias_conv5 = tf.Variable(tf.zeros([num_filters_conv5]), name="B5")

#--卷积神经网络----------------------------------------------------------------------------
# pre_CNN0
pre_layer_conv = tf.nn.conv2d(image_LR,
                              pre_weight_conv,
                              strides=[1, 1, 1, 1],
                              padding="VALID")  # 卷积
pre_layer_conv = tf.nn.bias_add(pre_layer_conv, pre_bias_conv)  # 添加偏置

# ResCNN
bp1_res_layer_conv1 = ResNet(pre_layer_conv,
                             pre_num_filters_conv,
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

bp2_res_layer_conv1 = ResNet(pre_layer_conv,
                             pre_num_filters_conv,
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

# 局部特征融合
merge_lff1_layer_conv = tf.concat(
    [bp1_res_layer_conv1, bp1_res_layer_conv2, bp1_res_layer_conv3], -1)

merge_lff2_layer_conv = tf.concat([
    bp2_res_layer_conv1, bp2_res_layer_conv2, bp2_res_layer_conv3,
    bp2_res_layer_conv4
], -1)

lff1_layer_conv = tf.nn.conv2d(merge_lff1_layer_conv,
                               lff1_weight_conv,
                               strides=[1, 1, 1, 1],
                               padding="VALID")

lff2_layer_conv = tf.nn.conv2d(merge_lff2_layer_conv,
                               lff2_weight_conv,
                               strides=[1, 1, 1, 1],
                               padding="VALID")

# 密集特征融合
dff1_layer_conv = tf.nn.conv2d(lff1_layer_conv,
                               dff1_weight_conv,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
dff1_layer_conv = tf.nn.bias_add(dff1_layer_conv, dff1_bias_conv)
dff1_layer_conv = PReLU(dff1_layer_conv, "DFF1")

dff2_layer_conv = tf.nn.conv2d(lff2_layer_conv,
                               dff2_weight_conv,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
dff2_layer_conv = tf.nn.bias_add(dff2_layer_conv, dff2_bias_conv)
dff2_layer_conv = PReLU(dff2_layer_conv, "DFF2")

# 全局特征融合
merge_gff_layer_conv = tf.concat([dff1_layer_conv, dff2_layer_conv], -1)

gff_layer_conv = tf.nn.conv2d(merge_gff_layer_conv,
                              gff_weight_conv,
                              strides=[1, 1, 1, 1],
                              padding="VALID")

# 全局残差
global_res_layer_conv = tf.add(gff_layer_conv, pre_layer_conv)

# PixelShuffle
layer_conv_shuffle = tf.depth_to_space(global_res_layer_conv, multiple)

# CNN5
layer_conv5 = tf.nn.conv2d(layer_conv_shuffle,
                           weight_conv5,
                           strides=[1, 1, 1, 1],
                           padding="VALID")
layer_final = tf.nn.bias_add(layer_conv5, bias_conv5, name="output_image")

#--损失----------------------------------------------------------------------------
# L1
l1 = tf.reduce_mean(tf.abs(tf.subtract(image_HR, layer_final)))

# L2
l2 = tf.reduce_mean(tf.square(tf.subtract(image_HR, layer_final)))

# SSIM
ssim = tf.reduce_mean(tf.image.ssim(image_HR, layer_final, max_val=1.0))

# PSNR
psnr = tf.reduce_mean(tf.image.psnr(image_HR, layer_final, max_val=1.0))

# 损失
alpha = 0.84
loss = alpha * (1 - ssim) + (1 - alpha) * l1

#--学习率----------------------------------------------------------------------------
global_step = tf.Variable(train_continue + 1)  #定义global_step 它会自动+1

learning_rate = tf.train.exponential_decay(
    1e-3, global_step, 100, 0.99, staircase=True) + 2e-5  #生成学习率

optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(
    l1, global_step=global_step)

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
        for i in xrange(0, batch_step):

            # 每次选取batch_size个样本进行训
            start = i * batch_size
            end = (i + 1) * batch_size

            # 训练集
            feed_dict_train = {
                image_LR: data_train_low[start:end],
                image_HR: data_train_high[start:end]
            }

            # 测试集
            feed_dict_vali = {
                image_LR: data_vali_low[0:min(32, vali_size)],
                image_HR: data_vali_high[0:min(32, vali_size)]
            }

            # 通过选取的样本进行训练神经网络并更新参数
            _, LOSS, SSIM, PSNR, rate = sess.run(
                [optimizer1, loss, ssim, psnr, learning_rate],
                feed_dict=feed_dict_train)

            if i % 10 == 0:
                #
                LOSS, SSIM, PSNR, rate = sess.run(
                    [loss, ssim, psnr, learning_rate],
                    feed_dict=feed_dict_vali)

                print(i)
                # 性能标准
                print("%d training epoch(s) \
                \nLOSS on dataset is %.8f \
                \nSSIM on dataset is %.8f \
                \nPSNR on dataset is %.6f \
                \nLearning Rate is %.6f" % (ep, LOSS, SSIM, PSNR, rate))

            if i == batch_step - 1:
                #
                LOSS, SSIM, PSNR, rate = sess.run(
                    [loss, ssim, psnr, learning_rate],
                    feed_dict=feed_dict_vali)

                with open("log/entropia.txt", "a+") as f:
                    print("%d\t%.8f\t%.8f\t%.6f" % (ep, LOSS, SSIM, PSNR),
                          file=f)

                # 保存模型
                saver.save(sess, "model/Entropia", global_step=ep)

                # 保存模型
                #saver.save(sess, "model-anime/Entropia_anime_mk2", global_step=ep)
                '''
                # 查看中间结果
                feed_dict_temp = {image_LR:data_train_low[ep % dataset_size:ep % dataset_size + 1],image_LR_insert:data_train_insert[ep % dataset_size:ep % dataset_siz+1],image_HR:data_train_high[ep % dataset_size:ep % dataset_size + 1]}

                image_LR_temp = sess.run(image_LR,feed_dict=feed_dict_temp).reshape(33,33,1)
                image_HR_temp = sess.run(image_HR,feed_dict=feed_dict_temp).reshape(19,19,1)
                layer_final_temp = sess.run(layer_final,feed_dict=feed_dict_temp).reshape(19,19,1)
                cv.imshow("TEMP of LR",image_LR_temp)
                cv.imshow("TEMP of HR",image_HR_temp)
                cv.imshow("TEMP of layer final",layer_final_temp)
                
                cv.waitKey(100)
                '''