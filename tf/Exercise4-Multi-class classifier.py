#!/usr/bin/python 
# -*- coding:utf-8 -*-

# %%
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd



# %%
def get_data(filename):
    with open(filename) as training_file:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        labels = data[:, 0].astype('int')
        images = data[:, 1:]
        # 转为三维数组
        images = images.astype('float').reshape(images.shape[0], 28, 28)
        data = None
        return images, labels


path_sign_mnist_train = f"{getcwd()}/tf/file/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/tf/file/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# %%
training_images = training_images.reshape(training_images.shape[0], training_images.shape[1], training_images.shape[2],
                                          1)
testing_images = testing_images.reshape(testing_images.shape[0], testing_images.shape[1], testing_images.shape[2], 1)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
print(training_images.shape)
print(testing_images.shape)

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

# %%
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_genertator = train_datagen.flow(training_images,
                                      training_labels,
                                      batch_size=25, )
validation_generator = validation_datagen.flow(testing_images,
                                               testing_labels,
                                               batch_size=25)

step_per_epoch = len(training_images) / 32
history = model.fit_generator(train_genertator, epochs=2, validation_data=validation_generator,
                              steps_per_epoch=step_per_epoch,
                              validation_steps=20, verbose=1)
model.evaluate(testing_images, testing_labels, verbose=0)

# %%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
