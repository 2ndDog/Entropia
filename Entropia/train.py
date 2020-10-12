import tensorflow as tf
from cv2 import cv2 as cv
import numpy as np
import h5py
import model_MRDN as Net

# 训练参数
batch_size = 16

num_epoch = 500

size_cut_edge = 23

size_image = 59

checkpoint_path = ".\\checkpoint\\MRDN"

# 数据集
with h5py.File("checkpoint\\train_MRDN.h5", "r") as hf:
    dataset_train_lr = np.array(hf.get("lr"))
    dataset_train_hr = np.array(hf.get("hr"))

size_dataset = len(dataset_train_lr)


def Loss(y_true, y_pred):
    # 定义裁剪张量取中心
    cut = tf.keras.layers.Cropping2D(cropping=((size_cut_edge, size_cut_edge),
                                               (size_cut_edge, size_cut_edge)))

    # 设置分配权重
    alpha_loss = 0.84
    alpha_cut = 0.4
    alpha_b = 0.6

    # 裁剪张量取中心
    y_true_cut = cut(y_true)
    y_pred_cut = cut(y_pred)

    # 双三次插值缩小
    y_true_b = tf.image.resize(y_true, (size_image, size_image), method=2)
    y_pred_b = tf.image.resize(y_pred, (size_image, size_image), method=2)

    mse_source = tf.keras.losses.MSE(y_true, y_pred)
    mse_cut = tf.keras.losses.MSE(y_true_cut, y_pred_cut)
    mse_b = tf.keras.losses.MSE(y_true_b, y_pred_b)

    # 多尺度SSIM
    ssim_mul = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
    ssim_mul_b = tf.image.ssim_multiscale(y_true_b, y_pred_b, max_val=1.0)

    # 加权平均L1损失
    l1 = (mse_source * (1 - alpha_b)) + (mse_b * alpha_b)
    ssim = (ssim_mul * (1 - alpha_b)) + (ssim_mul_b * alpha_b)

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

# 加载权重
model = tf.keras.models.load_model(checkpoint_path, custom_objects={
                                   "SSIM": SSIM, "PSNR": PSNR})

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.MAE,
              metrics=PSNR)

#模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    save_freq=100)

list_ckpt = [ckpt]

model.fit(dataset_train_lr,
          dataset_train_hr,
          epochs=num_epoch,
          batch_size=batch_size,
          callbacks=list_ckpt)
