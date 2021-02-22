#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:22:25 2021

@author: darshit
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import ocr

#img = tf.io.read_file('/home/darshit/Desktop/Mask_RCNN-master/dataset/images/Cars33.png')
img = tf.io.read_file('/home/darshit/Desktop/Mask_RCNN-master/dataset/images/Cars201.png')
img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
img = tf.image.resize(img, (224,224))
if len(img.shape) == 4:
    img = tf.squeeze(img,0)
img = tf.expand_dims(img, axis=0)   

print(img)
#
model = keras.models.load_model("plate_model_2")

p = model.predict(img)

img = cv2.imread('/home/darshit/Desktop/Mask_RCNN-master/dataset/images/Cars201.png')
img = cv2.resize(img, (244,244))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

y_hat = p[0]*244
print(y_hat)
xt, yt = int(y_hat[0]), int(y_hat[1])
xb, yb = int(y_hat[2]), int(y_hat[3])
image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 3)
plt.imshow(image)
plt.show()
crop = img[yt:xb, xt:yb] #crop plate
plt.imshow(crop)
plt.show()

cv2.imwrite('crop.jpg', crop)
ocr.to_text('crop.jpg')




