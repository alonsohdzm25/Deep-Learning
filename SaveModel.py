# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:15:27 2021

@author: alons
"""

## Saving and Loading Models
## Imports
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras import layers

## Part 1: Load the Cats vs Dog Dataset
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)


def format_image(image,label):
    image = tf.image.resize(image,(img_res,img_res))/ 255
    return image,label

num_examples = info.splits['train'].num_examples

img_res = 224
batch_size = 32

train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(batch_size).prefetch(1)

## Part 2: Transfer Learning with Tensorflow Hub
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(img_res,img_res,3))

feature_extractor.trainable = False

## Attach a clasification head
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2)
    ])

model.summary()

## Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=(True)),
              metrics=['accuracy'])

epochs = 1

history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches)

## Check Predictions
class_names = np.array(info.features['label'].names)
class_names

image_batch ,label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch,axis = -1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Label: ", label_batch)
print("Predicted label: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color = color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (Blue: Correct, Red: Incorrect)")


## Part 3: Save as Keras .h5 model
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)

## Part 4: Load the Keras .h5 Model
reloaded = tf.keras.models.load_model(
    export_path_keras,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

(abs(result_batch - reloaded_result_batch)).max()

## Keep Training
epochs = 3
history = reloaded.fit(train_batches, epochs=epochs,validation_data=validation_batches)

## Part 5: Export as SavedModel
t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

## Part 6: Load SavedModel
reloaded_sm = tf.saved_model.load(export_path_sm)

reloaded_sm_result_batch = reloaded_sm(image_batch, training = False).numpy()

(abs(result_batch - reloaded_sm_result_batch)).max()

## Part 7: Loading the SavedModel as a Keras Model
t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reload_sm_keras = tf.keras.models.load_model(
    export_path_sm,
    custom_objects={'KerasLayer' : hub.KerasLayer})

reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)

(abs(result_batch - reload_sm_keras_result_batch)).max()

## Part 8. Download your model
try:
    history.save(arguments['model'])








