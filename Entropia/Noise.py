import numpy as np
from PIL import Image
from cv2 import cv2 as cv

path_source = "D:\\SRCNN-Dataset\\anime\\"
path_jpg = "D:\\SRCNN-Dataset\\anime\\low\\"

filename_image = "3"

image_source = cv.imread(path_source + filename_image + ".png")
image_jpg = cv.imread(path_jpg + filename_image + ".jpg")

# 噪点
noise_source = noise_Y = Image.fromarray((image_source - image_jpg) + 128)

# 转化为YCrCb
image_source_YCrCb = cv.cvtColor(image_source, cv.COLOR_BGR2YCrCb)
image_jpg_YCrCb = cv.cvtColor(image_jpg, cv.COLOR_BGR2YCrCb)

# 通道分离
image_source_channels = cv.split(image_source_YCrCb)
image_jpg_channels = cv.split(image_jpg_YCrCb)

# 通道噪点
noise_Y = Image.fromarray((image_source_channels[0] - image_jpg_channels[0]) + 128)
noise_Cr = Image.fromarray((image_source_channels[1] - image_jpg_channels[1]) + 128)
noise_Cb = Image.fromarray((image_source_channels[2] - image_jpg_channels[2]) + 128)

noise_source.show()
noise_Y.show()
noise_Cr.show()
noise_Cb.show()

'''
cv.imshow("Source",noise_source)
cv.imshow("Y",noise_Y)
cv.imshow("Cr",noise_Cr)
cv.imshow("Cb",noise_Cb)

cv.waitKey(0)
'''