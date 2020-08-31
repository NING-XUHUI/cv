#!/usr/bin/python
#!/usr/bin/python
# -*- coding:utf-8 -*-

# %%
next_scissors = [os.path.join(scissors_dir, fname)
                 for fname in scissors_files[pic_index - 2:pic_index]]

for fname in scissors_files[pic_index - 2:pic_index]]

for i, img_path in enumerate(next_rock + next_paper + next_scissors):
    # print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()


# %%
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = os.path.join(base_dir, "rps")
training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=0.2,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

VALIDATION_DIR = os.path.join(base_dir, "rps-test-set")
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(150, 150),
                                                       class_mode="categorical",
                                                       batch_size=126)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              class_mode="categorical",
                                                              batch_size=126)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            self.model.stop_training = True


callbacks = myCallback()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                    validation_steps=3, callbacks=[callbacks])
model.save('rps.h5')

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
plt.legend(loc=0)
plt.figure()


plt.show()

# %%
#import numpy as np
from google.colab import files
#from keras.preprocessing import image

uploaded = files.upload()

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(fn)
#     print(classes)
#     print(classes)
# for fn in uploaded.keys():
    # predicting images
    #     path = fn
    #     img = image.load_img(path, target_size=(150, 150))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #
    #     images = np.vstack([x])
    #     classes = model.predict(images, batch_size=10)
    #     print(fn)
    #     print(classes)
