import tensorflow as tf
from cv2 import cv2 as cv
import numpy as np
import h5py
import os
import model_MRDN as Net

# 训练参数
batch_size = 16

num_epoch = 100

size_cut_edge = 23

size_image = 58

checkpoint_path = ".\\checkpoint\\MRDN\\mrdn.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

tf.compat.v1.disable_eager_execution()

#获取当前gpu设备列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#设置GPU按需分配方式
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.compat.v1.experimental.output_all_intermediates(True)

# 数据集
with h5py.File(".\\checkpoint\\train_MRDN.h5", "r") as hf:
    dataset_train_lr = np.array(hf.get("lr"))
    dataset_train_hr = np.array(hf.get("hr"))

# 测试集
with h5py.File(".\\checkpoint\\valid_MRDN.h5", "r") as hf:
    dataset_valid_lr = np.array(hf.get("lr"))
    dataset_valid_hr = np.array(hf.get("hr"))

# 打乱数据集
permutation = np.random.permutation(dataset_train_lr.shape[0])  # 记下第一维度的打乱顺序
dataset_train_lr = dataset_train_lr[permutation]  # 按照顺序索引
dataset_train_hr = dataset_train_hr[permutation]

size_dataset = len(dataset_train_lr)


def Loss(y_true, y_pred):
    # 设置分配权重
    alpha_loss = 0.84
    alpha_b = 0.4

    # 双三次插值缩小
    y_true_b = tf.image.resize(y_true, (size_image, size_image), method=tf.image.ResizeMethod.BICUBIC)
    y_pred_b = tf.image.resize(y_pred, (size_image, size_image), method=tf.image.ResizeMethod.BICUBIC)

    # 多尺度SSIM
    _MSSSIM_WEIGHTS = (0.44, 0.33, 0.23)
    ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0, power_factors=_MSSSIM_WEIGHTS)

    mea = tf.math.reduce_mean(tf.math.abs(y_true - y_pred))
    mae_b = tf.math.reduce_mean(tf.math.abs(y_true_b - y_pred_b))

    l1 = (mea * (1 - alpha_b)) + (mae_b * alpha_b)

    # 最终加权损失
    loss = alpha_loss * (1 - ssim) + (1 - alpha_loss) * l1

    return loss


def SSIM(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=5)
    return ssim


def PSNR(y_true, y_pred):
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr


# 部署模型
model = Net.DeployModel()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=Loss,
              metrics=[PSNR, SSIM])

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# 加载以前保存的权重
model.load_weights(latest)


#模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                          verbose=1,
                                          save_best_only = True, # 当设置为True，将只保留验证集上性能最好的模型
                                          save_weights_only = True,
                                          period =1)


model.fit(dataset_train_lr,
          dataset_train_hr,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data = (dataset_valid_lr, dataset_valid_hr),
          validation_freq = 1,
          callbacks=[ckpt])
