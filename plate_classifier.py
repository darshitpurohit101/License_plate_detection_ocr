# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:38:30 2021

@author: 91820
"""
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import os
import cv2

df = pd.read_json("Indian_Number_plates.json", lines=True)
print(df.head())
print(df.columns)

dataset = dict()
dataset["image_name"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

counter = 0
for index, row in df.iterrows():
    path = 'cars/car{}.jpg'.format(counter)

    dataset["image_name"].append(path)
    
    data_points = row["annotation"]
    
    dataset["top_x"].append(data_points[0]["points"][0]["x"])
    dataset["top_y"].append(data_points[0]["points"][0]["y"])
    dataset["bottom_x"].append(data_points[0]["points"][1]["x"])
    dataset["bottom_y"].append(data_points[0]["points"][1]["y"])
    
    counter += 1
    
df_store = pd.DataFrame(dataset)
print(df_store.head())

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, (224,224))
    if len(img.shape) == 4:
        img = tf.squeeze(img,0)
    img = tf.expand_dims(img, axis=0)
    return img

# For traininig
imgs = []
labels = []
for index, row in df_store[:int(0.9*len(df_store))].iterrows():
    imgs.append(load_img(row['image_name']))
    labels.append(tf.constant([[row['top_x'],row['top_y'],row['bottom_x'],row['bottom_y']]]))
train_images = tf.concat(imgs,axis=0)
train_labels = tf.concat(labels,axis=0)
print("length of train: ",len(train_images))

# For testing
imgs = []
labels = []
for index, row in df_store[int(0.9*len(df_store)):].iterrows():
    imgs.append(load_img(row['image_name']))
    labels.append(tf.constant([[row['top_x'],row['top_y'],row['bottom_x'],row['bottom_y']]]))
test_images = tf.concat(imgs,axis=0)
test_labels = tf.concat(labels,axis=0)

print('train_images: {}'.format(train_images.shape))
print('train_labels: {}'.format(train_labels.shape))
print('test_images: {}'.format(test_images.shape))
print('test_labels: {}'.format(test_labels.shape))

def show_img_bbox(img, label):
    img = img.numpy()
#    print("################################################## ",label)
    y_hat = label.numpy()*224
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ",y_hat)
    xt, yt = int(y_hat[0]), int(y_hat[1])
    xb, yb = int(y_hat[2]), int(y_hat[3])
    image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 3)
    plt.imshow(image)
    plt.show()
    
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
#        print("accuracy>>>>>>>>>>>>>>> ",logs['accuracy'])
        if (logs['accuracy']>= 0.70):
            print("accuracy reached>>>>>>>>>>>>>>>>>>>")
#            keys = list(logs.keys())
#            print("End epoch {} of training; got log keys: {}".format(epoch, keys))


# Use the function
#show_img_bbox(train_images[0], train_labels[0])
#    
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).repeat().batch(30)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(10)

#print(train_ds)
#
for i,l in train_ds.take(2):
    show_img_bbox(i[0], l[0])
#    

tf.keras.backend.clear_session()

i = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(64, (5,5), activation='relu')(i)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, (5,5), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(256, (7,7), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(512, (7,7), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
o = tf.keras.layers.Dense(4,activation='sigmoid')(x)

model = tf.keras.Model(inputs=[i], outputs=[o])

model.summary()

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(train_ds, epochs=30, steps_per_epoch=1, callbacks=[CustomCallback()])

model.save('plate_model_3')