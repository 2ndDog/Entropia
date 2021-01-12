import tensorflow as tf
"""
===============网络参数===============
"""
# 分辨率
size_lr = 65
size_hr = 114

# 输入通道数
num_input_channel = 1

# 放大倍数
multiple = 2

# Bottleneck层参数
size_kernel_bk = 1

# 残差块feature map数量
num_filters_des = 32

# 预处理卷积
size_kernel_pre = 5
num_filters_pre = 64

# 全局特征融合
size_kernel_gff = 3
num_filters_gff = 64

# 像素重组
size_kernel_ps = 5
num_filters_ps = 1


def DenseResBlock(input_layer, size_kernel_des, num_layers_des):
    # 用于保存最新的卷积层
    global layer_latest

    # 在循环前 第一层为输入数据
    layer_before = input_layer

    # 用于存储输入前所有层数
    list_layer_des = [input_layer]

    # 残差
    conv_res = tf.keras.layers.Add()

    # Identity
    conv_identity = tf.keras.layers.Conv2D(filters=num_filters_des,
                                           kernel_size=size_kernel_bk,
                                           strides=(1, 1),
                                           padding="same")

    # 密集连接卷积网络
    for layer in range(num_layers_des):
        # ==========定义密集卷积单元卷积层==========
        # Bottleneck
        conv_bk = tf.keras.layers.Conv2D(filters=num_filters_des,
                                         kernel_size=size_kernel_bk,
                                         strides=(1, 1),
                                         padding="same")

        # Convolution
        conv_des = tf.keras.layers.Conv2D(filters=num_filters_des,
                                          kernel_size=size_kernel_des,
                                          strides=(1, 1),
                                          padding="same",
                                          activation=tf.keras.layers.ReLU())

        # Concatenate
        concat_des = tf.keras.layers.Concatenate(-1)

        # 执行卷积
        layer_bk = conv_bk(layer_before)
        layer_des = conv_des(layer_bk)

        # 如果是最后一层,不需要concat
        if (layer + 1 == num_layers_des):
            layer_latest = layer_des

        # 当前输出添加到列表
        list_layer_des.append(layer_des)

        # Concat
        layer_before = concat_des(list_layer_des[:])

    #
    layer_identity = conv_identity(input_layer)

    # 残差后输出
    layer_res_out = conv_res([layer_latest, layer_identity])

    return layer_res_out


def GlobalFeatureFusion(local_features):
    conv_bk = tf.keras.layers.Conv2D(filters=num_filters_gff,
                                     kernel_size=size_kernel_bk,
                                     strides=(1, 1),
                                     padding="same")

    conv_gff = tf.keras.layers.Conv2D(filters=num_filters_gff,
                                      kernel_size=size_kernel_gff,
                                      strides=(1, 1),
                                      padding="valid",
                                      activation=tf.keras.layers.PReLU())

    # Concatenate
    concat_gff = tf.keras.layers.Concatenate(-1)

    # 局部特征联级
    layer_concat = concat_gff(local_features)

    # 全局特征融合
    layer_bk = conv_bk(layer_concat)

    layer_gff = conv_gff(layer_bk)

    return layer_gff


def PixelShuffle(input_features):
    # 定义亚像素卷积层
    conv_sub = tf.keras.layers.Lambda(
        lambda input: tf.nn.depth_to_space(input, multiple))

    # 像素重组后卷积
    conv_ps = tf.keras.layers.Conv2D(filters=num_filters_ps,
                                     kernel_size=size_kernel_ps,
                                     strides=(1, 1),
                                     padding="valid")

    layer_sub = conv_sub(input_features)

    layer_ps = conv_ps(layer_sub)

    return layer_ps


def MSDenseResidualNetwork(input_layer):
    conv_pre = tf.keras.layers.Conv2D(filters=num_filters_pre,
                                      kernel_size=size_kernel_pre,
                                      strides=(1, 1),
                                      padding="valid",
                                      activation=tf.keras.layers.PReLU())

    layer_pre = conv_pre(input_layer)

    layer_DRB3_1 = DenseResBlock(
        input_layer=layer_pre, size_kernel_des=3, num_layers_des=4)
    layer_DRB3_2 = DenseResBlock(
        input_layer=layer_DRB3_1, size_kernel_des=3, num_layers_des=4)
    layer_DRB3_3 = DenseResBlock(
        input_layer=layer_DRB3_2, size_kernel_des=3, num_layers_des=4)
    layer_DRB3_4 = DenseResBlock(
        input_layer=layer_DRB3_3, size_kernel_des=3, num_layers_des=4)

    layer_DRB5_1 = DenseResBlock(
        input_layer=layer_pre, size_kernel_des=5, num_layers_des=3)
    layer_DRB5_2 = DenseResBlock(
        input_layer=layer_DRB5_1, size_kernel_des=5, num_layers_des=3)
    layer_DRB5_3 = DenseResBlock(
        input_layer=layer_DRB5_2, size_kernel_des=5, num_layers_des=3)

    global_feature_fusion = GlobalFeatureFusion(
        [layer_pre, layer_DRB3_1, layer_DRB3_2, layer_DRB3_3, layer_DRB3_4, layer_DRB5_1, layer_DRB5_2, layer_DRB5_3])

    pixel_shuffle = PixelShuffle(global_feature_fusion)

    return pixel_shuffle


def DeployModel():
    # 定义输入层
    inputs = tf.keras.Input(shape=(size_lr, size_lr, num_input_channel))

    # 输出
    outputs = MSDenseResidualNetwork(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="output")

    # 打印模型
    model.summary()

    return model
