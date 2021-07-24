# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:31:44 2021

@author: alons
"""

## Dogs vs Cats Image Classification Without Image Augmentation

## Importign Packages
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

## Data loading
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

zip_dir_base = os.path.dirname(zip_dir)
print(zip_dir_base)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

## Understanding our data
num_cats_tr, num_cats_val = len(os.listdir(train_cats_dir)), len(os.listdir(validation_cats_dir))
num_dogs_tr, num_dogs_val = len(os.listdir(train_dogs_dir)), len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_validation = num_cats_val + num_dogs_val

print("Total training cat images: ", num_cats_tr)
print("Total training dog images: ", num_dogs_tr)

print("Total validation cat images: ", num_cats_val)
print("Total validation dog images: ", num_cats_val)
print("--")
print("Total training images: ", total_train)
print("Total validation images: ", total_validation)

## Setting model parameters
batch_size = 100 # Number of training examples to process before updating our model variables
img_shape = 150 # Image width and height

## Data preparation
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size = (img_shape,img_shape),
                                                           class_mode = 'binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=validation_dir,
                                                           shuffle=False,
                                                           target_size = (img_shape,img_shape),
                                                           class_mode = 'binary')

## Visualizing training images
sample_training_image, _ = next(train_data_gen) # The next function returns a batch from the dataset
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(img_arr):
    fig, axes = plt.subplots(1,5, figsize=(20,20)) 
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_image[:5])

## Model creation
# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2), 
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
    ])

## Compile the model
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Model summary
model.summary()

## Train the model
epochs = 100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train/float(batch_size))),
    epochs = epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_validation / float(batch_size)))
    )

## Visualizing results of the training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc ='lower right') 
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
