import tensorflow as tf

"""
===============网络参数===============
"""
# 分辨率
size_lr = 65
size_hr = 118

# 输入通道数
num_input_channel = 1

# 放大倍数
multiple = 2

# 全局特征融合后层数
num_filters_gff = 64

# Bottleneck层参数
size_kernel_bk = 1

# 预处理卷积层
size_kernel_pre = 3
num_filters_pre = 32

# 密集块参数
size_kernel_des_b = 3
num_filters_des_b = 32
num_layers_des_b = 4

# 密集连接参数
num_layers_des = 4

# 卷积1
size_kernel_conv1 = 3
num_filters_conv1 = 32

# 卷积2
size_kernel_conv2 = 5
num_filters_conv2 = 1

"""
^^^^^^^^^^^^^^^网络参数^^^^^^^^^^^^^^^
"""


def DenseConvBlock(layer_input):
    # 在循环前 第一层为输入数据
    layer_before = layer_input

    # 用于存储输入前所有层数
    list_layer_des = [layer_input]

    # 密集连接卷积网络
    for layer in range(num_layers_des_b):
        # ==========定义密集卷积单元卷积层==========
        # Bottleneck
        conv_bk = tf.keras.layers.Conv2D(
            filters=num_filters_des_b, kernel_size=size_kernel_bk, strides=(1, 1), padding="same")

        # Convolution
        conv_des = tf.keras.layers.Conv2D(
            filters=num_filters_des_b, kernel_size=size_kernel_des_b, strides=(1, 1), padding="same", activation="relu")

        # Concatenate
        concat_des = tf.keras.layers.Concatenate(-1)

        # 执行卷积
        layer_bk = conv_bk(layer_before)
        layer_des = conv_des(layer_bk)

        # 当前输出添加到列表
        list_layer_des.append(layer_des)

        # Concat
        layer_before = concat_des(list_layer_des[:])

    # Bottleneck
    conv_bk = tf.keras.layers.Conv2D(
        filters=num_filters_des_b, kernel_size=size_kernel_bk, strides=(1, 1), padding="same")

    # 瓶颈层压缩特征图后输出
    layer_des_output = conv_bk(layer_before)

    return layer_des_output


def DenseConv(layer_input):
    # 定义预处理层
    conv_pre = tf.keras.layers.Conv2D(
        filters=num_filters_pre, kernel_size=size_kernel_pre, strides=(1, 1), padding="valid", activation=tf.keras.layers.PReLU())

    # 定义全局特征融合
    conv_gff = tf.keras.layers.Conv2D(
        filters=num_filters_gff, kernel_size=size_kernel_bk, strides=(1, 1), padding="same")

    # 定义全局特征融合后卷积1
    conv_1 = tf.keras.layers.Conv2D(filters=num_filters_conv1, kernel_size=size_kernel_conv1, strides=(
        1, 1), padding="valid", activation=tf.keras.layers.PReLU())

    # 定义亚像素卷积层
    conv_sub = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))

    # 定义像素重组后卷积2
    conv_2 = tf.keras.layers.Conv2D(filters=num_filters_conv2, kernel_size=size_kernel_conv2, strides=(
        1, 1), padding="valid")

    layer_pre = conv_pre(layer_input)

    # 在循环前 第一层为输入数据
    layer_before = layer_pre

    # 用于存储输入前所有层数
    list_layer_des = [layer_before]

    for layer in range(num_layers_des):
        # 定义Bottleneck
        conv_bk = tf.keras.layers.Conv2D(
            filters=num_filters_des_b, kernel_size=size_kernel_bk, strides=(1, 1), padding="same")

        # 定义Concatenate
        concat_des = tf.keras.layers.Concatenate(-1)

        # 压缩输入
        layer_bk = conv_bk(layer_before)

        # 输入到密集块
        layer_des = DenseConvBlock(layer_bk)

        # 密集联级
        list_layer_des.append(layer_des)
        layer_before = concat_des(list_layer_des[:])

    # 全局密集特征融合
    layer_gff = conv_gff(layer_before)
    # 卷积
    layer_conv1 = conv_1(layer_gff)
    # 像素重组
    layer_sub = conv_sub(layer_conv1)
    # 卷积
    layer_conv2 = conv_2(layer_sub)

    return layer_conv2


def DeployModel():
    # 定义输入层
    inputs = tf.keras.Input(shape=(size_lr, size_lr, num_input_channel))

    # 输出
    outputs = DenseConv(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="output")

    # 打印模型
    model.summary()

    return model
