# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:01:13 2021

@author: alons
"""

## Tranfer Learning: Flowers in Inception V3
# Imports
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(training_set, validation_set), dataset_info = tfds.load('tf_flowers',
                                                         split = ['train[:70%]', 'train[70%:]'],
                                                         as_supervised=True,
                                                         with_info=True)

training_set

num_classes = dataset_info.features['label'].num_classes
num_training_examples = 0
num_validation_examples = 0

for example in training_set:
    num_training_examples += 1

for example in validation_set:
    num_validation_examples += 1

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))

for i, example in enumerate(training_set.take(10)):
    print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))
    

## Reformat Images and Create Batches
img_res = 299

def format_image(image,label):
    image = tf.image.resize(image, (img_res, img_res))/ 255
    return image,label

batch_size = 32

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)


## Do simple transfer learning with tensorflow hub
url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
feature_extractor = hub.KerasLayer(url, input_shape=(img_res,img_res, 3))

## Freeze the Pre-Trained Model
feature_extractor.trainable = False

## Attach a classification head
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(5, activation='softmax')
    ])

model.summary()

## Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(True)),
              metrics=['accuracy'])

## Train the model
epochs = 6

history = model.fit(train_batches,
                    epochs=epochs,
                    validation_data=validation_batches
                    )

## Plot Training and Validation Graphs
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

class_names = np.array(dataset_info.features['label'].names)
class_names

## Create an Image Batch and Make Predictions
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis = -1)
predicted_class_names = class_names[predicted_ids]

print("Label: ", label_batch)
print("Predicted Label: ", predicted_ids)

## Plot Model Predictions
plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5, n+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (Blue: Correct, Red: Incorrect)")
    
    
    
    
    
    

    
