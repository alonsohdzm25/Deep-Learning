# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:59:12 2021

@author: alons
"""

## TensorFlow Hub and Transfer Learning
## Import
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

## Part 1: Use a TensorFlow Hub MobileNet for prediction
# Download the classifier
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
img_res = 224

model = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=(img_res,img_res,3))
    ])

# Run it on a sigle image
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((img_res,img_res))
grace_hopper

grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

result = model.predict(grace_hopper[np.newaxis, ...])
result.shape

predicted_class = np.argmax(result[0], axis=-1)
predicted_class

## Decode the predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Predictions: "+ predicted_class_name.title())


## Part 2: Use a TensorFlow Hub models for the Cats vs. Dogs dataset
(train_examples, validation_examples), info = tfds.load('cats_vs_dogs',with_info=True, as_supervised=True,split=['train[:80%]', 'train[80%:]'],)

num_examples = info.splits['train'].num_examples
num_clases = info.features['label'].num_classes

for i, example_image in enumerate(train_examples.take(3)):
    print("Image {} shape: {}".format(i+1,example_image[0].shape))

def format_image(image, label):
    image = tf.image.resize(image, (img_res, img_res))/255.0
    return image, label

batch_size = 32
train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(batch_size).prefetch(1)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

plt.figure(figsize=(10,10))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")


## Part 3: Do simple transfer learning with TensorFlow Hub
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(img_res, img_res,3))

feature_batch = feature_extractor(image_batch)
print(feature_batch.shape)

feature_extractor.trainable = False

## Attach a classification head
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2)
])

model.summary()

## Trainn the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(True)),
              metrics=['accuracy'])

epochs = 6

history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()

## Check predictions
class_names = np.array(info.features['label'].names)
class_names

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis = -1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize = (10,10))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.subplots_adjust(hspace = 0.3)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
    