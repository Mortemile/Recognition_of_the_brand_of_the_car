import keras
from PIL import Image
from pathlib import Path

from keras.layers import MaxPooling2D, Conv2DTranspose, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

import os
import random

seed_value = 12
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.times = []


class Dataset:
    def __init__(self):

        self.prep_data = None
        self.base = 'car'
        self.sets = None
        self.classes = None

    def create_sets(self):
        '''
        Функция создания выборок
        '''
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # Gets a list of directories
        self.classes = sorted(os.listdir('middle_fmr'))
        counts = []

        for j, d in enumerate(self.classes):

            # Gets names of images for each class
            files = sorted(os.listdir(os.path.join('middle_fmr', d)))  # name of brand

            # separation parameter for set
            counts.append(len(files))
            count = counts[-1] * 0.8

            for i in range(len(files)):

                # loads img
                sample = np.array(image.load_img(os.path.join(  # split into pixels
                    'middle_fmr',
                    d,
                    files[i]), target_size=(54, 96)))

                # add img to test/train set
                if i < count:
                    x_train.append(sample)
                    y_train.append(j)
                else:
                    x_test.append(sample)
                    y_test.append(j)
        self.sets = (np.array(x_train) / 255., np.array(y_train)), (np.array(x_test) / 255., np.array(y_test))

        self.prep_data = 1

    def prepocessing_data(self, img_name):
        sample = np.array(image.load_img(img_name, target_size=(54, 96)))
        self.prep_data = np.array(sample) / 255.

        print('Тестовое изображение:')
        plt.imshow(self.prep_data)
        plt.axis('off')
        plt.show()


class Model:
    def __init__(self, trds):
        self.dat = None
        self.model = None
        self.trds = trds

    def train_model(self, epochs, use_callback=True):

        # Create model

        self.model = keras.Sequential()

        self.model.add(Conv2D(8, (3, 3), input_shape=(54, 96, 3), padding='same',
                              activation='relu'))  # двумерный слой, задаем сколько нейронов + ядро
        self.model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
        self.model.add(Flatten())  # выравнивание
        self.model.add(Dense(40, activation='relu'))  # полносвязный на 64 нейрона
        self.model.add(Dense(3, activation='softmax'))

        # Train

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

        accuracy_callback = AccuracyCallback()
        callbacks = []
        if use_callback:
            callbacks = [accuracy_callback]

        y_train = to_categorical(self.trds.sets[0][1], 3)
        x_train = self.trds.sets[0][0]
        x_test = self.trds.sets[1][0]
        y_test = to_categorical(self.trds.sets[1][1], 3)

        history = self.model.fit(x_train, y_train,
                                 batch_size=24,
                                 validation_data=(x_test, y_test),
                                 callbacks=callbacks,
                                 epochs=epochs,
                                 verbose=1)
        return history

    def test_model(self):
        sample = self.trds.prep_data
        # print(sample)
        pred = self.model.predict(sample[None, ...])[0]
        print()
        print('Результат предсказания модели:')
        for i in range(len(self.trds.classes)):
            print(
                f'Модель распознала класс «{self.trds.classes[i]}» на {round(100 * pred[i], 1)} %')


class BrendRecognition:
    def __init__(self):
        self.trds = Dataset()
        self.trmodel = None
        self.task_type = None

    def create_sets(self):
        self.trds.create_sets()

    def prepocessing_data(self, img_name):
        self.trds.prepocessing_data(img_name)

    def predic(self):
        self.trmodel = Model(self.trds)
        # self.trmodel.predic()

    def create_model(self, layers):
        self.trmodel = Model(self.trds)
        # self.trmodel.create_model(layers)

    def train_model(self, epochs):
        self.trmodel = Model(self.trds)
        self.trmodel.train_model(epochs)

    def test_model(self):
        self.trmodel.test_model()


car = BrendRecognition()

car.create_sets()
car.train_model(5)

car.prepocessing_data('ferr.png')
car.test_model()
print('Правильный ответ - Ferrari')

car.prepocessing_data('mercedes.png')
car.test_model()
print('Правильный ответ - Mercedes')

car.prepocessing_data('reno.png')
car.test_model()
print('Правильный ответ - Renault')
