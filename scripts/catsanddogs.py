import tensorflow as tf
from matplotlib import pyplot as plt
# import datasets 
import numpy as np
import os
import pathlib
import PIL.Image #abbreviation of python Image Library, a.k.a Pillow) import Image to load, manage, save image files in diverse formats
import IPython.display as display #to display images later on

# Change dataset directory to match yours
train_dir = pathlib.Path('/home/iris/catkin_ws/src/tensorflow_ex/datasets/train')
test_dir = pathlib.Path('/home/iris/catkin_ws/src/tensorflow_ex/datasets/test')

image_count = len(list(train_dir.glob('*/*.jpg')))
print(image_count) #25000 images for training
image_count = len(list(test_dir.glob('*/*.jpg')))
print(image_count) #12500 images for testing

#separate train and test data
train = list(train_dir.glob('dogs/*'))
PIL.Image.open(str(train[0]))
test = list(test_dir.glob('test1/*'))
PIL.Image.open(str(test[0]))

#create the dataset:
batch_size = 10
img_height = 180
img_width = 180


#we create a training subset of 80% of images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.5,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  #we create a validation subset of 20% of images
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.5,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2
model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  #tf.keras.layers.Conv2D(32, 3, activation='relu'),
  #tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1
)