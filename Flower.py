# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:21:48 2021

@author: alons
"""

## Importing packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

## Data loading
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses','daisy','dandelion','sunflowers','tulips']

for cl in classes:
    img_path = os.path.join(base_dir,cl)
    images = glob.glob(img_path+'/*.jpg')
    print("{}: {} Images".format(cl,len(images)))
    train = images[:round(len(images)*0.8)]
    val = images[round(len(images)*0.8):]
    
    for t in train:
        if not os.path.exists(os.path.join(base_dir,'train',cl)):
            os.makedirs(os.path.join(base_dir,'train',cl))
        shutil.move(t,os.path.join(base_dir,'train',cl))
        
    for v in val:
        if not os.path.exists(os.path.join(base_dir,'val',cl)):
            os.makedirs(os.path.join(base_dir,'val',cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))


train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir,'val')

## Data Augmentation
batch_size = 100
img_shape = 150

# Apply Random Horizontal Flip
image_gen = ImageDataGenerator(rescale=(1./255), horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory = train_dir,
                                               shuffle=True,
                                               target_size = (img_shape,img_shape))

def plotImages(images_arr):
    fig, axes = plt.subplots(1,5, figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Apply Random Rotation
image_gen = ImageDataGenerator(rescale=(1./255), rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory = train_dir,
                                               shuffle=True,
                                               target_size = (img_shape,img_shape))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Apply Random Zoom
image_gen = ImageDataGenerator(rescale=(1./255), zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory = train_dir,
                                               shuffle=True,
                                               target_size = (img_shape,img_shape))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Put it all together
image_gen = ImageDataGenerator(rescale=(1./255),
                               rotation_range=45,
                               zoom_range=0.5,
                               horizontal_flip=(True),
                               width_shift_range=0.15,
                               height_shift_range=0.15
                               )
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory = train_dir,
                                               shuffle=True,
                                               target_size = (img_shape,img_shape),
                                               class_mode='sparse')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

## Create a Data Generator for the Validation Set
image_data_val = ImageDataGenerator(rescale=(1./255))
val_data_gen = image_data_val.flow_from_directory(batch_size=batch_size,
                                                  directory=val_dir,
                                                  target_size=(img_shape,img_shape),
                                                  class_mode='sparse')

## Create the CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,3, padding='same', activation='relu', input_shape=(img_shape,img_shape,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(32,3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(64,3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
    ])

model.summary()

## Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(True)),
              metrics=['accuracy'])

## Train the model
epochs = 75

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(val_data_gen.n / float(batch_size))
    )

## Plot training and validation graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')

plt.show()