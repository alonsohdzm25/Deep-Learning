# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:53:54 2021

@author: alons
"""

## Librerias
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import glob
import shutil
import matplotlib.pyplot as plt


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

## Directorio en el que se encuentran nuestras imagenes
base_dir = os.path.join('c:\\Users\\alons\\.keras\\datasets','plagas')
base_dir

classes = ['acaro_tostador','limon_sano','mancha_sectorial','piojo_harinoso','piojo_rojo']

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

## Separando el dataset en entrenamineto y validacion
train_dir = os.path.join(base_dir,'train')
train_dir
val_dir = os.path.join(base_dir,'val')
val_dir

img_size = 224
batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 
                                                                    shuffle = True,
                                                                    batch_size=batch_size,
                                                                    image_size = (img_size,img_size))

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_dir, 
                                                                    shuffle = True,
                                                                    batch_size=batch_size,
                                                                    image_size = (img_size,img_size))

class_names = train_dataset.class_names
num_training_examples = 0
num_validation_examples = 0

for cl in class_names:
    num_training_examples += len(glob.glob(os.path.join(train_dir, cl + '/*.jpg')))
    num_validation_examples += len(glob.glob(os.path.join(val_dir, cl + '/*.jpg')))


print('Total Number of Classes: {}'.format(len(class_names)))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))
    
for i, example in enumerate(train_dataset.take(5)):
  print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1][0]))

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )
train_data_gen = image_gen_train.flow_from_directory(batch_size = batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size = (img_size,img_size))

## Creating validation data generator
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                  directory=val_dir,
                                  target_size=(img_size,img_size))

## Aplicando aprendizaje por transferencia simple
URL ='https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
feature_extractor = hub.KerasLayer(URL, input_shape = (img_size,img_size,3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Flatten(),
    layers.Dense(len(class_names),activation='softmax')
    ])

model.summary()


## Entrenando Modelo
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

epochs = 10

history = model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen, shuffle=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label="Training Accuracy")
plt.plot(epochs_range,val_acc,label="Validation Accuracy")
plt.legend(loc = 'lower right')
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label="Training Loss")
plt.plot(epochs_range,val_loss,label="Validation Loss")
plt.legend(loc = 'upper right')
plt.title("Training and Validation Loss")
plt.show()

reloaded = tf.keras.models.load_model('R:\Cursos\Python\Machine Learning\plagas88_mn.h5',
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(reloaded)
tflite_model = converter.convert()

with open('model_plagas.tflite','wb') as f:
    f.write(tflite_model)